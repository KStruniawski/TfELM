from sklearn.cluster import KMeans
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf

from Resources.Kernel import Kernel, CombinedSumKernel, CombinedProductKernel
from Resources.kernel_distances import calculate_pairwise_distances_vector, calculate_pairwise_distances


class USKELMLayer:
    """
        Unsupervised Kernel ELM (USKELM) Layer.

        This class represents an Unsupervised Kernel Extreme Learning Machine (USKELM) layer. It encapsulates the
        properties and functionality required for building, training, and making predictions with a USKELM model.

        Parameters:
        - kernel: The kernel function to be used.
        - embedding_size (int): The size of the embedded feature space.
        - activation (str): The activation function to be used. Defaults to 'tanh'.
        - act_params (dict): Additional parameters for the activation function. Defaults to None.
        - C (float): Regularization parameter C. Defaults to 1.0.
        - nystrom_approximation (bool): Whether to use the Nystrom approximation method. Defaults to False.
        - landmark_selection_method (str): Method for selecting landmarks. Defaults to 'random'.
        - lam (float): Lambda parameter for Laplacian regularization. Defaults to 0.5.
        - **params: Additional parameters.

        Attributes:
        - error_history: History of errors during training, initialized to None.
        - feature_map: Feature map matrix, initialized to None.
        - name (str): Name of the layer, set to "elm".
        - beta: Beta weights matrix, initialized to None.
        - input: Input data, initialized to None.
        - output: Output data, initialized to None.
        - lam (float): Lambda parameter for Laplacian regularization.
        - act_params (dict): Additional parameters for the activation function.
        - embedding_size (int): The size of the embedded feature space.
        - activation_name (str): Name of the activation function.
        - activation: Activation function.
        - C (float): Regularization parameter C.
        - kernel: The kernel function used for feature mapping.
        - nystrom_approximation (bool): Whether Nystrom approximation is used.
        - landmark_selection_method (str): Method for selecting landmarks.
        - K: Kernel matrix, initialized to None.
        - denoising: Denoising parameter, initialized to None.
        - denoising_param: Denoising parameter, initialized to None.

        Methods:
        - build(input_shape): Builds the layer.
        - fit(x): Fits the layer to the input data.
        - predict(x, clustering=False, k=None): Makes predictions based on the input data.
        - calc_output(x): Calculates the output based on the input data.
        - __str__(): Returns a string representation of the layer.
        - count_params(): Counts the number of trainable and non-trainable parameters in the layer.
        - to_dict(): Converts the layer to a dictionary representation.
        - load(attributes): Loads the layer from a dictionary of attributes.

        Example:
        -----------
        >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
        >>> layer = USKELMLayer(number_neurons=kernel, embedding_size=3, lam=0.001)
        >>> layer.build(X_train.shape)
        >>> model = USKELMModel(layer=layer)
    """
    def __init__(self,
                 kernel,
                 embedding_size,
                 activation='tanh',
                 act_params=None,
                 C=1.0,
                 nystrom_approximation=False,
                 landmark_selection_method='random',
                 lam=0.5,
                 **params):
        self.error_history = None
        self.feature_map = None
        self.name = "elm"
        self.beta = None
        self.input = None
        self.output = None
        self.lam = lam
        self.act_params = act_params
        self.embedding_size = embedding_size
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
        self.C = C
        self.kernel = kernel
        self.nystrom_approximation = nystrom_approximation
        self.landmark_selection_method = landmark_selection_method

        if "K" in params:
            self.K = params.pop("K")
        if "input" in params:
            self.input = params.pop("input")
        if "beta" in params:
            self.beta = params.pop("beta")

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
            Builds the USKELM layer.

            Parameters:
            -----------
            - input_shape: Shape of the input data.

            Example:
            -----------
            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
            >>> layer = USKELMLayer(number_neurons=kernel, embedding_size=3, lam=0.001)
            >>> layer.build(X_train.shape)
        """
        observations = input_shape[0]
        self.K = tf.Variable(
            tf.zeros(shape=(observations, observations)),
            dtype=tf.float32,
            trainable=False
        )

    def fit(self, x):
        """
            Fits the USKELM layer to the input data.

            Parameters:
            -----------
            - x: Input data.

            Example:
            -----------
            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
            >>> layer = USKELMLayer(number_neurons=kernel, embedding_size=3, lam=0.001)
            >>> layer.build(X_train.shape)
            >>> layer.fit(X_train)
        """
        x = tf.cast(x, dtype=tf.float32)
        d = tf.shape(x)[-1]
        self.input = x

        if self.embedding_size > d + 1:
            raise Exception

        # Laplacian Graph
        squared_norms = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        dot_product = tf.matmul(x, x, transpose_b=True)
        distances = squared_norms - 2 * dot_product + tf.transpose(squared_norms)
        distances = tf.maximum(distances, 0.0)
        # distances = tf.reduce_sum(tf.square(x[:, tf.newaxis, :] - x[tf.newaxis, :, :]), axis=-1)
        sigma = 1.0
        W = tf.exp(-distances / (2.0 * sigma ** 2))
        D = tf.linalg.diag(tf.reduce_sum(W, axis=1))
        L_unnormalized = D - W
        D_sqrt_inv = tf.linalg.inv(tf.linalg.sqrtm(D))
        L = tf.matmul(tf.matmul(D_sqrt_inv, L_unnormalized), D_sqrt_inv)

        if self.nystrom_approximation:
            num_rows = tf.shape(x)[0]
            shuffled_indices = tf.random.shuffle(tf.range(num_rows))
            selected_indices = shuffled_indices[:100]
            L = tf.gather(x, selected_indices)
            C = calculate_pairwise_distances_vector(x, L, self.kernel.ev)
            W = calculate_pairwise_distances(L, self.kernel.ev)
            K = tf.matmul(tf.matmul(C, tf.linalg.inv(W)), C, transpose_b=True)
        else:
            K = calculate_pairwise_distances(x, self.kernel.ev)

        eq = tf.eye(tf.shape(K)[0]) + self.lam * tf.matmul(tf.matmul(L, K), K, transpose_b=True)
        e, v = tf.linalg.eigh(eq)
        sorted_indices = tf.argsort(e)
        v_sorted = tf.gather(v, sorted_indices, axis=1)
        v_trimmed = v_sorted[:, 1:self.embedding_size+1]
        norm_factor = tf.norm(tf.matmul(tf.matmul(K, tf.transpose(K)), v_trimmed), axis=0)
        v_trimmed = v_trimmed / norm_factor
        beta = tf.matmul(K, v_trimmed, transpose_a=True)

        self.beta = beta
        self.K = K

    def predict(self, x, clustering=False, k=None):
        """
            Makes predictions based on the input data.

            Parameters:
            -----------
            - x: Input data.
            - clustering (bool): Whether to perform clustering. Defaults to False.
            - k: Number of clusters if clustering is True.

            Returns:
            -----------
            - Predicted values.

            Example:
            -----------
            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
            >>> layer = USKELMLayer(number_neurons=kernel, embedding_size=3, lam=0.001)
            >>> layer.build(X_train.shape)
            >>> layer.fit(X_train)
            >>> pred = layer.predict(X_test)
        """
        x = tf.cast(x, dtype=tf.float32)
        k = calculate_pairwise_distances_vector(x, self.input, self.kernel.ev)
        kpKT = tf.matmul(k, self.K)
        output = tf.matmul(kpKT, self.beta)
        self.output = output

        if not clustering:
            return output
        else:
            output = output.numpy()
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(output)
            cluster_labels = kmeans.labels_
            return output, cluster_labels

    def calc_output(self, x):
        """
            Calculates the output based on the input data.

            Parameters:
            -----------
            - x: Input data.

            Returns:
            -----------
            - Output data.
        """
        x = tf.cast(x, dtype=tf.float32)
        k = calculate_pairwise_distances_vector(x, self.input, self.kernel.ev)
        out = self.activation(tf.matmul(k, self.beta))
        self.output = out
        return out

    def __str__(self):
        """
            Returns a string representation of the USKELM layer.

            Returns:
            -----------
            - String representation.
        """
        return f"{self.name}, kernel: {self.kernel.__class__.__name__}"

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

        non_trainable = 0
        return {'trainable': trainable, 'non_trainable': non_trainable, 'all': trainable + non_trainable}

    def to_dict(self):
        """
            Converts the layer to a dictionary representation.

            Returns:
            -----------
            - Dictionary containing layer attributes.
        """
        attributes = {
            'name': 'USKELMLayer',
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
            "denoising_param": self.denoising_param,
            "lam": self.lam,
            "embedding_size": self.embedding_size
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
            Loads the layer from a dictionary of attributes.

            Parameters:
            -----------
            - attributes: Dictionary containing layer attributes.

            Returns:
            -----------
            - Loaded USKELMLayer instance.
        """
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

        attributes.update({"kernel": k})
        layer = cls(**attributes)
        return layer