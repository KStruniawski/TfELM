from Optimizers import ELMOptimizer
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf


class OSELMLayer:
    """
        Online Sequential Extreme Learning Machine (OS-ELM) layer.

        This layer implements the Online Sequential Extreme Learning Machine (OS-ELM),
        a variant of Extreme Learning Machine (ELM) suitable for online learning scenarios.

        Parameters:
        -----------
        - number_neurons (int): The number of neurons in the ELM layer.
        - activation (str): Activation function for the ELM layer neurons.
        - act_params (dict): Dictionary containing activation function parameters.
        - C (float): Regularization parameter for the ELM layer.
        - beta_optimizer (ELMOptimizer): Optimizer for the beta coefficients.
        - is_orthogonalized (bool): If True, performs orthogonalization on the input weights.

        Attributes:
        -----------
        - y (tf.Tensor): Output labels.
        - P (tf.Tensor): P matrix used in sequential learning.
        - error_history (tf.Tensor): History of errors during optimization.
        - feature_map (tf.Tensor): Feature map generated during fitting.
        - beta (tf.Tensor): Beta coefficients learned by the layer.
        - bias (tf.Tensor): Bias term learned by the layer.
        - alpha (tf.Tensor): Input weights learned by the layer.
        - input (tf.Tensor): Input data tensor.
        - output (tf.Tensor): Output predictions tensor.
        - name (str): Name of the layer.

        Methods:
        -----------
        - build(input_shape): Build the layer based on the input shape.
        - fit_initialize(x, y): Initialize the layer and fit it to the initial data.
        - fit_seq(x, y): Fit the layer sequentially to new data.
        - predict(x): Predict the output for the given input data.
        - predict_proba(x): Predict the probabilities output for the given input data.
        - calc_output(x): Calculate the output for the given input data.
        - count_params(): Count the number of trainable and non-trainable parameters.
        - to_dict(): Convert the layer's attributes to a dictionary.
        - load(attributes): Load the layer from a dictionary of attributes.

        Example:
        -----------
        Initialize OSELMLayer with specified parameters

        >>> layer = OSELMLayer(1000, 'tanh')

        Initialize OSELMModel with the OSELMLayer and other parameters

        >>> model = OSELMModel(layer, prefetch_size=120, batch_size=64, verbose=0)
    """

    def __init__(self,
                 number_neurons,
                 activation='tanh',
                 act_params=None,
                 C=0.001,
                 beta_optimizer: ELMOptimizer = None,
                 is_orthogonalized=False,
                 **params):
        self.y = None
        self.P = None
        self.error_history = None
        self.feature_map = None
        self.beta = None
        self.bias = None
        self.alpha = None
        self.input = None
        self.output = None
        self.name = "OS-ELM"
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
        self.act_params = act_params
        self.activation_name = activation
        self.activation = eval("act." + activation)
        self.number_neurons = number_neurons
        self.C = C
        self.beta_optimizer = beta_optimizer

        if "beta" in params:
            self.beta = params.pop("beta")
        if "alpha" in params:
            self.alpha = params.pop("alpha")
        if "bias" in params:
            self.bias = params.pop("bias")

    def build(self, input_shape):
        """
            Build the layer based on the input shape.

            Parameters:
            -----------
            - input_shape: Shape of the input data.

            Returns:
            None
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
            self.alpha, _ = tf.linalg.qr(self.alpha)
            self.bias = self.bias / tf.norm(self.bias)

    def fit_initialize(self, x, y):
        """
            Initialize the layer and fit it to the initial data.

            Parameters:
            -----------
            - x (tf.Tensor): Input data tensor.
            - y (tf.Tensor): Output labels.

            Returns:
            None
        """
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)

        n = tf.shape(x)[0].numpy()

        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)

        if n < self.number_neurons:
            P = tf.matmul(H, H, transpose_b=True)
            if self.C is not None:
                P = tf.linalg.set_diag(P, tf.linalg.diag_part(P) + self.C)
            P = tf.linalg.inv(P)
            beta = tf.matmul(tf.matmul(H, P, transpose_a=True), y)
        else:
            P = tf.matmul(H, H, transpose_a=True)
            if self.C is not None:
                P = tf.linalg.set_diag(P, tf.linalg.diag_part(P) + self.C)
            P = tf.linalg.inv(P)
            beta = tf.matmul(tf.matmul(P, H, transpose_b=True), y)

        if self.beta_optimizer is not None:
            self.beta, self.error_history = self.beta_optimizer.optimize(beta, H, y)
        else:
            self.beta = beta
        self.y = y
        self.input = x
        self.P = P
        self.feature_map = H

    def fit_seq(self, x, y):
        """
            Fit the layer sequentially to new data.

            Parameters:
            -----------
            - x (tf.Tensor): Input data tensor.
            - y (tf.Tensor): Output labels.

            Returns:
            None
        """
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)

        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)

        p0 = tf.linalg.inv(tf.eye(tf.shape(H)[0]) + tf.matmul(tf.matmul(H, self.P), H, transpose_b=True))
        p1 = tf.matmul(tf.matmul(tf.matmul(tf.matmul(self.P, H, transpose_b=True), p0), H), self.P)
        self.P = self.P - p1
        beta = self.beta + tf.matmul(tf.matmul(self.P, H, transpose_b=True), y - tf.matmul(H, self.beta))

        if self.beta_optimizer is not None:
            self.beta, self.error_history = self.beta_optimizer.optimize(beta, H, y)
        else:
            self.beta = beta

        self.input = x
        self.feature_map = H

    def predict(self, x):
        """
            Predict the output for the given input data.

            Parameters:
            -----------
            - x (tf.Tensor): Input data tensor.

            Returns:
            tf.Tensor: Predicted output tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)
        output = tf.matmul(H, self.beta)
        self.output = output
        return output

    def predict_proba(self, x):
        """
            Predict the probabilities output for the given input data.

            Parameters:
            -----------
            - x (tf.Tensor): Input data tensor.

            Returns:
            tf.Tensor: Predicted probabilities tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        return tf.keras.activations.softmax(pred)

    def calc_output(self, x):
        """
            Calculate the output for the given input data.

            Parameters:
            -----------
            - x (tf.Tensor): Input data tensor.

            Returns:
            tf.Tensor: Calculated output tensor.
        """
        return tf.matmul(x, self.beta, transpose_b=True)

    def __str__(self):
        """
            Get the string representation of the layer.

            Returns:
            -----------
            str: String representation of the layer.
        """
        return self.name

    def count_params(self):
        """
            Count the number of trainable and non-trainable parameters.

            Returns:
            -----------
            dict: Dictionary containing the counts of trainable and non-trainable parameters.
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
           Convert the layer's attributes to a dictionary.

           Returns:
            -----------
           dict: Dictionary containing the layer's attributes.
       """
        attributes = {
            'name': 'OSELMLayer',
            'number_neurons': self.number_neurons,
            'activation': self.activation_name,
            'act_params': self.act_params,
            'C': self.C,
            'is_orthogonalized': self.is_orthogonalized,
            "beta": self.beta,
            "alpha": self.alpha,
            "bias": self.bias
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
            Load the layer from a dictionary of attributes.

            Parameters:
            -----------
            - attributes (dict): Dictionary containing the layer's attributes.

            Returns:
            -----------
            OSELMLayer: An instance of the OSELMLayer class.
        """
        return cls(attributes)
