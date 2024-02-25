from sklearn.cluster import KMeans
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf
from Resources.gram_schmidt import gram_schmidt


class USELMLayer:
    """
        Unsupervised Extreme Learning Machine (USELM) Layer.

        This layer implements an unsupervised Extreme Learning Machine (ELM) for data embedding tasks.

        Parameters:
        -----------
        - number_neurons (int): Number of neurons in the hidden layer.
        - embedding_size (int): Size of the embedding space.
        - activation (str): Activation function to be used in the hidden layer.
        - act_params (dict): Parameters for the activation function.
        - C (float): Regularization parameter for the ELM.
        - is_orthogonalized (bool): Whether to enforce orthogonality in the hidden layer.
        - lam (float): Regularization parameter for the graph Laplacian.

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
        - lam (float): Regularization parameter for the graph Laplacian.
        - act_params (dict): Parameters for the activation function.
        - is_orthogonalized (bool): Whether to enforce orthogonality in the hidden layer.
        - embedding_size (int): Size of the embedding space.

        Example:
        -----------
        >>> layer = USELMLayer(number_neurons=5000, embedding_size=3, lam=0.001)
        >>> model = USELMModel(layer)
    """

    def __init__(self,
                 number_neurons,
                 embedding_size,
                 activation='tanh',
                 act_params=None,
                 C=1.0,
                 is_orthogonalized=False,
                 lam=0.5,
                 **params):
        self.error_history = None
        self.feature_map = None
        self.name = "elm"
        self.beta = None
        self.bias = None
        self.alpha = None
        self.input = None
        self.output = None
        self.lam = lam
        self.act_params = act_params
        self.is_orthogonalized = is_orthogonalized
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

            Example:
            -----------
            >>> layer = USELMLayer(number_neurons=5000, embedding_size=3, lam=0.001)
            >>> layer.build(X.shape)
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

            Example:
            -----------
            >>> layer = USELMLayer(number_neurons=5000, embedding_size=3, lam=0.001)
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

        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)

        if x.shape[0] > self.number_neurons:
            eq = tf.eye(tf.shape(H)[1]) + self.lam * tf.matmul(tf.matmul(H, L, transpose_a=True), H)
            e, v = tf.linalg.eigh(eq)
            sorted_indices = tf.argsort(e)
            v_sorted = tf.gather(v, sorted_indices, axis=1)
            v_trimmed = v_sorted[:, 1:self.embedding_size + 1]
            norm_factor = tf.norm(tf.matmul(H, v_trimmed), axis=0)
            v_trimmed = v_trimmed / norm_factor
            beta = v_trimmed
        else:
            eq = tf.eye(tf.shape(H)[0]) + self.lam * tf.matmul(tf.matmul(L, H), H, transpose_b=True)
            e, v = tf.linalg.eigh(eq)
            sorted_indices = tf.argsort(e)
            v_sorted = tf.gather(v, sorted_indices, axis=1)
            v_trimmed = v_sorted[:, 1:self.embedding_size+1]
            norm_factor = tf.norm(tf.matmul(tf.matmul(H, tf.transpose(H)), v_trimmed), axis=0)
            v_trimmed = v_trimmed / norm_factor
            beta = tf.matmul(H, v_trimmed, transpose_a=True)

        self.beta = beta
        self.feature_map = H
        self.output = tf.matmul(H, self.beta)

    def predict(self, x, clustering=False, k=None):
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
            >>> layer = USELMLayer(number_neurons=5000, embedding_size=3, lam=0.001)
            >>> layer.build(X_train.shape)
            >>> layer.fit(X_train)
            >>> pred = layer.predict(X_test)
        """
        x = tf.cast(x, dtype=tf.float32)
        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)
        output = tf.matmul(H, self.beta)

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
            Calculates the output of the USELM layer for the given input data.

            Parameters:
            -----------
            - x (tf.Tensor): Input data tensor.

            Returns:
            -----------
            tf.Tensor: Output tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
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
            'name': 'USELMLayer',
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
            "embedding_size": self.embedding_size
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
            Load a USELMLayer instance from a dictionary of attributes.

            Parameters:
            -----------
            - attributes (dict): Dictionary containing layer attributes.

            Returns:
            -----------
            USELMLayer: Loaded USELMLayer instance.
        """
        return cls(**attributes)
