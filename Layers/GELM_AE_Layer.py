import h5py
import numpy as np

from Optimizers.ELMOptimizer import ELMOptimizer
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf
from Resources.gram_schmidt import gram_schmidt


class GELM_AE_Layer:
    """
        Graph regularized ELM Autoencoder Layer.

        This layer implements a Generalized Extreme Learning Machine (ELM) autoencoder,
        which learns a feature representation of the input data through unsupervised learning.

        Parameters:
        -----------
        - number_neurons (int): Number of neurons in the hidden layer.
        - activation (str): default='tanh'
            The name of the activation function to be applied to the neurons that corresponds to the names of function
            in class Activation
        - act_params (dict): Parameters for the activation function.
            Additional parameters for the activation function (if needed - see implementation of particular function in
            class Activation).
        - C (float): Regularization parameter for the ELM.
            Regularization parameter to control the degree of regularization applied to the hidden layer.
        - beta_optimizer (ELMOptimizer): Optimizer for updating beta weights.
            An optimizer to optimize the output weights (beta) of the layer applied after the Moore-Penrose operation to
            finetune the beta parameter based on provided to optimizer loss function and optimization algorithm.
        - is_orthogonalized (bool): Whether to enforce orthogonality in the hidden layer.
            Indicates whether the input weights of the hidden neurons are orthogonalized, if yes the orthogonalization
            is performed (recommended to be applied for multilayer variants of ELM).
        - lam (float): Regularization parameter for the graph Laplacian.
        - K (float): Diagonal parameter for the graph Laplacian.

        Attributes:
        -----------
        - error_history (list): History of error values during training.
        - feature_map (tf.Tensor): Feature map learned by the layer.
        - name (str): Name of the layer.
        - beta (tf.Tensor): Beta weights learned by the layer.
        - bias (tf.Tensor): Bias vector learned by the layer.
        - alpha (tf.Tensor): Input-to-hidden weight matrix.
        - input (tf.Tensor): Input data tensor.
        - output (tf.Tensor): Output data tensor.
        - activation_name (str): Name of the activation function.
        - activation (function): Activation function.
        - C (float): Regularization parameter for the ELM.
        - denoising (str): Type of denoising method to be applied to input data.
        - denoising_param (float): Parameter for the denoising method.

        Examples:
        -----------
            Initialize Multilayer ELM model

            >>> model = ML_ELMModel(verbose=1, classification=False)

            Add ELM autoencoder layers for unsupervised learning and feature extraction

            >>> model.add(GELM_AE_Layer(number_neurons=100))
            >>> model.add(GELM_AE_Layer(number_neurons=1000))
            >>> model.add(GELM_AE_Layer(number_neurons=1000))
            >>> model.add(GELM_AE_Layer(number_neurons=100))

            Fit the model to the data

            >>> model.fit(X)
        -----------
    """
    def __init__(self,
                 number_neurons,
                 activation='tanh',
                 act_params=None,
                 C=1.0,
                 beta_optimizer: ELMOptimizer = None,
                 is_orthogonalized=False,
                 lam=0.5,
                 K=0.5,
                 **params):
        self.error_history = None
        self.feature_map = None
        self.name = "GELM_AE_Layer"
        self.beta = None
        self.bias = None
        self.alpha = None
        self.input = None
        self.output = None
        self.lam = lam
        self.K = K
        self.act_params = act_params
        self.beta_optimizer = beta_optimizer
        self.is_orthogonalized = is_orthogonalized
        if act_params is None:
            act = ActivationFunction(1.0)
        elif "act_param" in act_params and "act_param2" in act_params:
            act = ActivationFunction(act_param=act_params["act_param"], act_param2=act_params["act_param2"])
        elif "act_param" in act_params:
            act = ActivationFunction(act_param=act_params["act_param"])
        elif "knots" in act_params:
            act = ActivationFunction(knots=act_params["knots"])
        else:
            raise Exception("TypeError: Wrong specified activation function parameters")
        self.activation_name = activation
        self.activation = eval("act." + activation)
        self.number_neurons = number_neurons
        self.C = C

        if "beta" in params:
            self.beta = params.pop("beta")
        if "alpha" in params:
            self.alpha = params.pop("alpha")
        if "bias" in params:
            self.bias = params.pop("bias")
        if "denoising" in params:
            self.denoising = params.pop("denoising")
        else:
            self.denoising = None
        if "denoising_param" in params:
            self.denoising_param = params.pop("denoising_param")
        else:
            self.denoising_param = None

    def build(self, input_shape):
        """
            Build the layer by initializing weights and biases.

            Parameters:
            -----------
            - input_shape (tuple): Shape of the input data.
        """
        alpha = input_shape[-1]
        alpha_initializer = tf.random_uniform_initializer(-1, 1)
        self.alpha = tf.Variable(
            alpha_initializer(shape=(alpha, self.number_neurons)),
            dtype=tf.float32,
            trainable=False
        )
        bias_initializer = tf.random_uniform_initializer(0, 1)
        self.bias = tf.Variable(
            bias_initializer(shape=(self.number_neurons,)),
            dtype=tf.float32,
            trainable=False
        )
        if self.is_orthogonalized:
            self.alpha = gram_schmidt(self.alpha)
            self.bias = self.bias / tf.norm(self.bias)

    def fit(self, x):
        """
            Fit the layer to the input data.

            Parameters:
            -----------
            - x (tf.Tensor): Input data tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
        n, m = tf.shape(x)
        self.input = x

        # Laplacian Graph
        squared_norms = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        dot_product = tf.matmul(x, x, transpose_b=True)
        distances = squared_norms - 2 * dot_product + tf.transpose(squared_norms)
        distances = tf.maximum(distances, 0.0)
        sigma = 1.0
        W = tf.exp(-distances / (2.0 * sigma ** 2))
        D = tf.linalg.diag(tf.reduce_sum(W, axis=1))
        L_unnormalized = D - W
        D_sqrt_inv = tf.linalg.inv(tf.linalg.sqrtm(D))
        L = tf.matmul(tf.matmul(D_sqrt_inv, L_unnormalized), D_sqrt_inv)

        C = tf.linalg.diag(tf.fill([n], self.K))

        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)

        if x.shape[0] > self.number_neurons:
            HtCH = tf.matmul(tf.matmul(H, C, transpose_a=True), H)
            lHtLH = self.lam * tf.matmul(tf.matmul(H, L, transpose_a=True), H)
            i = tf.eye(tf.shape(HtCH)[0])
            inv = tf.linalg.inv(i + HtCH + lHtLH)
            beta = tf.matmul(tf.matmul(tf.matmul(inv, H, transpose_b=True), C), x)
        else:
            CHHt = tf.matmul(tf.matmul(C, H), H, transpose_b=True)
            lLHHt = self.lam * tf.matmul(tf.matmul(L, H), H, transpose_b=True)
            i = tf.eye(tf.shape(CHHt)[0])
            inv = tf.linalg.inv(i + CHHt + lLHHt)
            beta = tf.matmul(tf.matmul(tf.matmul(H, inv, transpose_a=True), C), x)

        if self.beta_optimizer is not None:
            self.beta, self.error_history = self.beta_optimizer.optimize(beta, H, x)
        else:
            self.beta = beta

        self.feature_map = H
        self.output = tf.matmul(H, self.beta)

    def predict(self, x):
        """
            Predicts the output for the given input data.

            Parameters:
            -----------
            - x (tf.Tensor): Input data tensor.

            Returns:
            -----------
            tf.Tensor: Predicted output tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)
        output = tf.matmul(H, self.beta)
        return output

    def predict_proba(self, x):
        """
            Predicts the probabilities output for the given input data upon application of the softmax funtion.

            Parameters:
            -----------
            - x (tf.Tensor): Input data tensor.

            Returns:
            -----------
            tf.Tensor: Predicted output tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        return tf.keras.activations.softmax(pred)

    def calc_output(self, x):
        x = tf.cast(x, dtype=tf.float32)
        """
            Calculates the output of the ELM layer for the given input data.
    
            Parameters:
            -----------
            - x (tf.Tensor): Input data tensor.
    
            Returns:
            -----------
            tf.Tensor: Output tensor.
        """
        out = self.activation(tf.matmul(x, self.beta, transpose_b=True))
        self.output = out
        return out

    def __str__(self):
        """
            Returns a string representation of the ELM layer.

            Returns:
            -----------
            str: String representation.
        """
        return f"{self.name}, neurons: {self.number_neurons}"

    def count_params(self):
        """
            Counts the number of trainable and non-trainable parameters in the ELM layer.

            Returns:
            -----------
            dict: Dictionary containing counts for trainable, non-trainable, and total parameters.
        """
        if self.beta is None:
            trainable = 0
        else:
            trainable = self.beta.shape[0] * self.beta.shape[1]
        if self.alpha is None or self.bias is None:
            non_trainable = 0
        else:
            non_trainable = self.alpha.shape[0] * self.alpha.shape[1] + self.bias.shape[0]
        return {'trainable': trainable, 'non_trainable': non_trainable, 'all': trainable + non_trainable}

    def to_dict(self):
        """
            Convert the layer attributes to a dictionary.

            Returns:
            -----------
            dict: Dictionary containing layer attributes.
        """
        attributes = {
            'name': 'GELM_AE_Layer',
            'number_neurons': self.number_neurons,
            'activation': self.activation_name,
            'act_params': self.act_params,
            'C': self.C,
            'is_orthogonalized': self.is_orthogonalized,
            "beta": self.beta,
            "alpha": self.alpha,
            "bias": self.bias,
            "denoising": self.denoising,
            "denoising_param": self.denoising_param,
            "lam": self.lam,
            "K": self.K
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
            Load a GELM_AE_Layer instance from a dictionary of attributes.

            Parameters:
            -----------
            - attributes (dict): Dictionary containing layer attributes.

            Returns:
            -----------
            GELM_AE_Layer: Loaded GELM_AE_Layer instance.
        """
        return cls(**attributes)
