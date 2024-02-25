import numpy as np

from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf

from Resources.Kernel import Kernel, CombinedSumKernel, CombinedProductKernel
from Resources.kernel_distances import calculate_pairwise_distances_vector, calculate_pairwise_distances


def proceed_kernel(attributes):
    if "kernel" in attributes:
        k_n = attributes.pop("kernel")
        k_p = attributes.pop("kernel_param")
        k_t = attributes.pop("kernel_type")
        if k_t == "Kernel":
            k = Kernel(kernel_name=k_n, param=k_p)
        elif k_t == "CombinedSumKernel":
            kernels = []
            for n, p in zip(k_n, k_p):
                kernels.append(Kernel(kernel_name=n.decode('utf-8'), param=p))
            k = CombinedSumKernel(kernels)
        else:
            kernels = []
            for n, p in zip(k_n, k_p):
                kernels.append(Kernel(kernel_name=n.decode('utf-8'), param=p))
            k = CombinedProductKernel(kernels)
    else:
        k = Kernel()
    return k


class KELMLayer:
    """
        Kernel Extreme Learning Machine (KELM) Layer.

        This class implements a single layer of the Kernel Extreme Learning Machine.

        Parameters:
        -----------
            kernel (object): Instance of a kernel function.
            activation (str, optional): Name of the activation function. Defaults to 'tanh'.
            act_params (dict, optional): Parameters for the activation function.
            C (float, optional): Regularization parameter. Defaults to 1.0.
            nystrom_approximation (bool, optional): Whether to use Nystrom approximation for large datasets.
                Defaults to False.
            landmark_selection_method (str, optional): Method for landmark selection if using Nystrom approximation.
                Defaults to 'random'.

        Attributes:
        -----------
            K (tensor): Kernel matrix.
            error_history (list): History of errors during training.
            feature_map (tensor): Feature map.
            name (str): Name of the layer.
            beta (tensor): Weights of the layer.
            input (tensor): Input data.
            output (tensor): Output data.
            nystrom_approximation (bool): Indicates whether Nystrom approximation is used.
            landmark_selection_method (str): Method for landmark selection.
            activation (function): Activation function.
            C (float): Regularization parameter.
            kernel (object): Instance of a kernel function.
            denoising (str, optional): Denoising method. Defaults to None.
            denoising_param (float, optional): Denoising parameter. Defaults to None.

        Methods:
        -----------
            build(input_shape): Build the layer with the given input shape.
            fit(x, y): Fit the layer to the input-output pairs.
            predict(x): Predict the output for the input data.
            predict_proba(x): Predict class probabilities for the input data.
            calc_output(x): Calculate the output for the input data.
            count_params(): Count the number of trainable and non-trainable parameters.
            to_dict(): Convert the layer attributes to a dictionary.
            load(attributes): Load the layer from a dictionary of attributes.

        Examples:
        -----------
        Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)

        >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

        Initialize a Kernel Extreme Learning Machine (KELM) layer

        >>> layer = KELMLayer(kernel, 'mish')

        Initialize a Kernel Extreme Learning Machine (KELM) layer with Nystrom kernel matrix approximation

        >>> layer = KELMLayer(kernel, 'mish', nystrom_approximation=True)

        Initialize a Kernel Extreme Learning Machine (KELM) model

        >>> model = KELMModel(layer)
        """
    def __init__(self, kernel: Kernel, activation='tanh', act_params=None, C=1.0,
                 nystrom_approximation=False, landmark_selection_method='random', **params):
        self.K = None
        self.error_history = None
        self.feature_map = None
        self.name = "kelm"
        self.beta = None
        self.input = None
        self.output = None
        self.nystrom_approximation = nystrom_approximation
        self.landmark_selection_method = landmark_selection_method
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
        self.activation = eval("act." + activation)
        self.C = C

        if "beta" in params:
            self.beta = params.pop("beta")
        if "input" in params:
            self.input = params.pop("input")
        if "K" in params:
            self.K = params.pop("K")

        if "denoising" in params:
            self.denoising = params.pop("denoising")
        else:
            self.denoising = None

        if "denoising_param" in params:
            self.denoising_param = params.pop("denoising_param")
        else:
            self.denoising_param = None

        if "kernel_param" in params:
            params.update({'kernel': kernel})
            self.kernel = proceed_kernel(params)
        else:
            self.kernel = kernel

    def build(self, input_shape):
        """
            Build the layer with the given input shape.

            This method initializes the layer by creating a kernel matrix placeholder
            of appropriate dimensions based on the input shape.

            Args:
            -----------
                input_shape (tuple): Shape of the input data.

            Example:
            -----------
                >>> kelm = KELMLayer(number_neurons=1000, activation='mish')
                >>> kelm.build(x.shape)
        """
        observations = input_shape[0]
        self.K = tf.Variable(
            tf.zeros(shape=(observations, observations)),
            dtype=tf.float32,
            trainable=False
        )

    def fit(self, x, y):
        """
            Fit the layer to the input-output pairs.

            This method fits the layer to the given input-output pairs by calculating
            the kernel matrix, applying regularization, and computing the weights.

            Args:
            -----------
                x (tensor): Input data.
                y (tensor): Target values.

            Example:
            -----------
                >>> kelm = KELMLayer(number_neurons=1000, activation='mish')
                >>> kelm.build(x.shape)
                >>> kelm.fit(train_data, train_targets)
        """
        x = tf.cast(x, dtype=tf.float32)
        self.input = x

        if self.nystrom_approximation:
            num_rows = tf.shape(x)[0]
            shuffled_indices = tf.random.shuffle(tf.range(num_rows))
            selected_indices = shuffled_indices[:100]
            L = tf.gather(x, selected_indices)
            C = calculate_pairwise_distances_vector(x, L, self.kernel.ev)
            W = calculate_pairwise_distances(L, self.kernel.ev)
            self.K = tf.matmul(tf.matmul(C, tf.linalg.inv(W)), C, transpose_b=True)
        else:
            self.K = calculate_pairwise_distances(x, self.kernel.ev)

        if self.C is not None:
            self.K = tf.linalg.set_diag(self.K, tf.linalg.diag_part(self.K) + self.C)
        else:
            self.K = tf.linalg.set_diag(self.K, tf.linalg.diag_part(self.K))

        self.K = tf.linalg.pinv(self.K)
        self.beta = tf.matmul(self.K, y)
        # self.output = self.activation(tf.matmul(x, self.beta, transpose_b=True))

    def predict(self, x):
        """
            Predict the output for the input data.

            This method predicts the output for the given input data.

            Args:
            -----------
                x (tensor): Input data.

            Returns:
            -----------
                tensor: Predicted output.

            Example:
            -----------
                >>> kelm = KELMLayer(number_neurons=1000, activation='mish')
                >>> kelm.build(x.shape)
                >>> kelm.fit(train_data, train_targets)
                >>> pred = kelm.predict(test_data)
        """
        x = tf.cast(x, dtype=tf.float32)
        k = calculate_pairwise_distances_vector(x, self.input, self.kernel.ev)
        kpKT = tf.matmul(k, self.K)
        output = tf.matmul(kpKT, self.beta)
        self.output = output
        return output

    def predict_proba(self, x):
        """
            Predict class probabilities for the input data.

            This method predicts class probabilities for the given input data.

            Args:
            -----------
                x (tensor): Input data.

            Returns:
            -----------
                numpy.ndarray: Predicted class probabilities.

            Example:
            -----------
            >>> elm = ELMLayer(number_neurons=1000, activation='mish')
            >>> elm.build(x.shape)
            >>> elm.fit(train_data, train_targets)
            >>> pred = elm.predict_proba(test_data)
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        return tf.keras.activations.softmax(pred).numpy()

    def calc_output(self, x):
        """
            Calculate the output for the input data.

            This method calculates the output for the given input data.

            Args:
            -----------
                x (tensor): Input data.

            Returns:
            -----------
                tensor: Calculated output.
        """
        x = tf.cast(x, dtype=tf.float32)
        k = calculate_pairwise_distances_vector(x, self.input, self.kernel.ev)
        out = self.activation(tf.matmul(k, self.beta))
        self.output = out
        return out

    def __str__(self):
        return self.name

    def count_params(self):
        """
            Count the number of trainable and non-trainable parameters.

            This method counts the number of trainable and non-trainable parameters
            in the layer.

            Returns:
            -----------
                dict: Dictionary containing the counts of trainable, non-trainable,
                    and total parameters.
        """
        if self.beta is None:
            trainable = 0
        else:
            trainable = self.beta.shape[0]*self.beta.shape[1]
        non_trainable = 0
        return {'trainable': trainable, 'non_trainable': non_trainable, 'all': trainable+non_trainable}

    def to_dict(self):
        """
            Convert the layer attributes to a dictionary.

            This method converts the layer's attributes to a dictionary.

            Returns:
            -----------
                dict: Dictionary containing the layer attributes.
        """
        attributes = {
            'name': 'KELMLayer',
            'C': self.C,
            "beta": self.beta,
            "kernel": self.kernel.kernel_name,
            "kernel_param": self.kernel.kernel_param,
            "kernel_type": self.kernel.__class__.__name__,
            "nystrom_approximation": self.nystrom_approximation,
            "landmark_selection_method": self.landmark_selection_method,
            "input": self.input,
            "K": self.K,
            "denoising": self.denoising,
            "denoising_param": self.denoising_param
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
            Load the layer from a dictionary of attributes.

            This class method loads the layer from a dictionary of attributes.

            Args:
            -----------
                attributes (dict): Dictionary containing the layer attributes.

            Returns:
            -----------
                object: Instance of the loaded layer.
        """
        k = proceed_kernel(attributes)
        attributes.update({"kernel": k})
        layer = cls(**attributes)
        return layer



