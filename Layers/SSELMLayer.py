import numpy as np

from Optimizers.ELMOptimizer import ELMOptimizer
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf
from Resources.gram_schmidt import gram_schmidt


class SSELMLayer:
    """
        Semi-Supervised Extreme Learning Machine (SSELM) layer.

        Parameters:
        -----------
        - number_neurons (int): Number of neurons in the layer.
        - activation (str): Activation function to use. Defaults to 'tanh'.
        - act_params (dict or None): Parameters for the activation function.
        - C (float): Regularization parameter. Defaults to 1.0.
        - beta_optimizer (ELMOptimizer or None): Optimizer for beta parameters.
        - is_orthogonalized (bool): Whether to orthogonalize the alpha matrix. Defaults to False.
        - lam (float): Trade-off parameter between Laplacian graph regularization and output fitting. Defaults to 0.5.
        - params (dict): Additional parameters.

        Returns:
        -----------
        None

        Example:
        -----------
            Initializing the Semi-Supervised Extreme Learning Machine (SS-ELM) layer.
            Number of neurons: 1000, Regularization parameter: 0.001

            >>> layer = SSELMLayer(number_neurons=1000, lam=0.001)

            Initializing the SS-ELM model with the defined layer

            >>> model = SSELMModel(layer)
    """
    def __init__(self,
                 number_neurons,
                 activation='tanh',
                 act_params=None,
                 C=1.0,
                 beta_optimizer: ELMOptimizer = None,
                 is_orthogonalized=False,
                 lam=0.5,
                 **params):
        self.error_history = None
        self.feature_map = None
        self.name = "sselm"
        self.beta = None
        self.bias = None
        self.alpha = None
        self.input = None
        self.output = None
        self.lam = lam
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
            Builds the layer with the given input shape.

            Parameters:
            -----------
            - input_shape (tuple): Shape of the input data.

            Returns:
            -----------
            None

            Example:
            -----------
            >>> layer = SSELMLayer(number_neurons=1000, lam=0.001)
            >>> layer.build(x.shape)
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

    def fit(self, x_labeled, x_unlabeled, y_labeled, y_unlabeled):
        """
            Fits the layer to labeled and unlabeled data.

            Parameters:
            -----------
            - x_labeled (numpy.ndarray): Labeled input features.
            - x_unlabeled (numpy.ndarray): Unlabeled input features.
            - y_labeled (numpy.ndarray): Labeled target labels.
            - y_unlabeled (numpy.ndarray): Unlabeled target labels.

            Returns:
            -----------
            None

            Example:
            -----------
            >>> layer = SSELMLayer(number_neurons=1000, lam=0.001)
            >>> layer.build(x.shape)
            >>> layer.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
        """
        y_un_zero = np.zeros(shape=np.shape(y_unlabeled))
        n_labeled = np.shape(x_labeled)[0]
        X_combined = np.vstack([x_labeled, x_unlabeled])
        Y_combined = np.vstack([y_labeled, y_un_zero])

        X_combined = tf.cast(X_combined, dtype=tf.float32)
        Y_combined = tf.cast(Y_combined, dtype=tf.float32)

        self.input = X_combined

        # Laplacian Graph
        squared_norms = tf.reduce_sum(tf.square(X_combined), axis=1, keepdims=True)
        dot_product = tf.matmul(X_combined, X_combined, transpose_b=True)
        distances = squared_norms - 2 * dot_product + tf.transpose(squared_norms)
        distances = tf.maximum(distances, 0.0)
        sigma = 1.0
        W = tf.exp(-distances / (2.0 * sigma ** 2))
        D = tf.linalg.diag(tf.reduce_sum(W, axis=1))
        L_unnormalized = D - W
        D_sqrt_inv = tf.linalg.inv(tf.linalg.sqrtm(D))
        L = tf.matmul(tf.matmul(D_sqrt_inv, L_unnormalized), D_sqrt_inv)

        ni = self.C / tf.reduce_sum(Y_combined, axis=0)
        C = tf.linalg.diag(tf.reduce_sum(tf.multiply(Y_combined, ni), axis=1))
        m, n = tf.shape(C)[0], tf.shape(C)[1]
        mask = tf.math.logical_and(tf.range(m) > n_labeled, tf.range(n) > n_labeled)
        C = tf.where(mask, tf.zeros_like(C), C)

        H = tf.matmul(X_combined, self.alpha) + self.bias
        H = self.activation(H)

        if X_combined.shape[0] > self.number_neurons:
            HtCH = tf.matmul(tf.matmul(H, C, transpose_a=True), H)
            lHtLH = self.lam * tf.matmul(tf.matmul(H, L, transpose_a=True), H)
            i = tf.eye(tf.shape(HtCH)[0])
            inv = tf.linalg.inv(i + HtCH + lHtLH)
            beta = tf.matmul(tf.matmul(tf.matmul(inv, H, transpose_b=True), C), Y_combined)
        else:
            CHHt = tf.matmul(tf.matmul(C, H), H, transpose_b=True)
            lLHHt = self.lam * tf.matmul(tf.matmul(L, H), H, transpose_b=True)
            i = tf.eye(tf.shape(CHHt)[0])
            inv = tf.linalg.inv(i + CHHt + lLHHt)
            beta = tf.matmul(tf.matmul(tf.matmul(H, inv, transpose_a=True), C), Y_combined)

        if self.beta_optimizer is not None:
            self.beta, self.error_history = self.beta_optimizer.optimize(beta, H, Y_combined)
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

        Example:
        -----------
        >>> layer = SSELMLayer(number_neurons=1000, lam=0.001)
        >>> layer.build(x.shape)
        >>> layer.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
        >>> pred = layer.predict(test_data)
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

        Example:
        -----------
        >>> layer = SSELMLayer(number_neurons=1000, lam=0.001)
        >>> layer.build(x.shape)
        >>> layer.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
        >>> pred = layer.predict_proba(test_data)
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        return tf.keras.activations.softmax(pred)

    def calc_output(self, x):
        x = tf.cast(x, dtype=tf.float32)
        """
        Calculates the output of the SSELM layer for the given input data.

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
        Returns a string representation of the SSELM layer.

        Returns:
        -----------
        str: String representation.
        """
        return f"{self.name}, neurons: {self.number_neurons}"

    def count_params(self):
        """
        Counts the number of trainable and non-trainable parameters in the SSELM layer.

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
            Converts the layer's attributes to a dictionary.

            Returns:
            -----------
            dict: Dictionary containing the layer's attributes.
        """
        attributes = {
            'name': 'SSELMLayer',
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
            "lam": self.lam
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
            Loads an instance of the layer using the provided attributes.

            Parameters:
            -----------
            - attributes (dict): Dictionary containing the layer's attributes.

            Returns:
            -----------
            SSELMLayer: Instance of the layer.
        """
        return cls(**attributes)
