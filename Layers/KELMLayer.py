import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors

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
    def __init__(self, kernel: Kernel, activation='tanh', act_params=None, C=0.0,
                 nystrom_approximation=False, landmark_selection_method='random',
                 random_pct=0.1, **params):
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

        self.random_pct = random_pct

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

        n_samples = int(self.random_pct * x.shape[0])

        if self.nystrom_approximation:
            if self.landmark_selection_method != "stratified" and \
                    self.landmark_selection_method != "information_gain_based":
                L = eval(f"{self.landmark_selection_method}_sampling(x, n_samples)")
            else:
                y_new = tf.argmax(y, axis=1)
                y_new = tf.cast(y_new, dtype=tf.int32)
                L = eval(f"{self.landmark_selection_method}_sampling(x, y_new, n_samples)")
            C = calculate_pairwise_distances_vector(x, L, self.kernel.ev)
            W = calculate_pairwise_distances(L, self.kernel.ev)
            diagonal = tf.linalg.diag_part(W)
            diagonal_with_small_value = diagonal + 0.00001
            W = tf.linalg.set_diag(W, diagonal_with_small_value)
            K = tf.matmul(tf.matmul(C, tf.linalg.inv(W)), C, transpose_b=True)
        else:
            K = calculate_pairwise_distances(x, self.kernel.ev)

        diagonal = tf.linalg.diag_part(K)
        diagonal_with_small_value = diagonal + 0.1
        K = tf.linalg.set_diag(K, diagonal_with_small_value)
        self.K = tf.linalg.inv(K)
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


# Random Sampling
def random_sampling(data, n_samples):
    num_rows = tf.shape(data)[0]
    selected_indices = tf.random.shuffle(tf.range(num_rows))[:n_samples]
    sampled_data = tf.gather(data, selected_indices)
    return sampled_data


# Uniform Sampling
def uniform_sampling(data, n_samples):
    num_rows = tf.shape(data)[0]
    indices = tf.range(num_rows)
    shuffled_indices = tf.random.shuffle(indices)
    selected_indices = shuffled_indices[:n_samples]
    sampled_data = tf.gather(data, selected_indices)
    return sampled_data


# K-Means Clustering Sampling
def kmeans_sampling(data, n_samples):
    data_np = data.numpy()
    kmeans = KMeans(n_clusters=n_samples, random_state=0)
    kmeans.fit(data_np)
    centroids = kmeans.cluster_centers_
    centroids_tensor = tf.convert_to_tensor(centroids, dtype=tf.float32)
    return centroids_tensor


def stratified_sampling(data, labels, n_samples_per_class):
    # Get unique class labels
    unique_labels, _ = tf.unique(labels)
    n_samples_per_class = int(n_samples_per_class / len(unique_labels))

    # Initialize list to store sampled landmarks
    sampled_landmarks = []

    # Sample landmarks from each stratum (class)
    for label in unique_labels:
        # Get indices of data points with the current label
        indices = tf.where(tf.equal(labels, label))[:, 0]

        # Sample landmarks from the current stratum
        sampled_indices = tf.random.shuffle(indices)[:n_samples_per_class]

        # Add sampled data points to the list
        sampled_landmarks.extend(tf.gather(data, sampled_indices))

    # Convert sampled landmarks to tensor
    sampled_landmarks_tensor = tf.convert_to_tensor(sampled_landmarks)

    return sampled_landmarks_tensor


def greedy_sampling(data, n_samples):
    # Initialize list to store sampled landmarks
    sampled_landmarks = []

    # Compute the leverage scores
    q, _ = tf.linalg.qr(data, full_matrices=False)
    leverage_scores = tf.reduce_sum(tf.square(q), axis=1)

    # Greedily select points with highest leverage scores
    for _ in range(n_samples):
        max_index = tf.argmax(leverage_scores)
        sampled_landmarks.append(data[max_index])
        leverage_scores = tf.tensor_scatter_nd_update(leverage_scores, [[max_index]], [0.])

    # Convert sampled landmarks to tensor
    sampled_landmarks_tensor = tf.convert_to_tensor(sampled_landmarks)

    return sampled_landmarks_tensor


def farthest_first_traversal_sampling(data, n_samples):
    # Initialize list to store sampled landmarks
    sampled_landmarks = []

    # Randomly select the first landmark
    initial_index = tf.random.uniform((), maxval=tf.shape(data)[0], dtype=tf.int32)
    sampled_landmarks.append(data[initial_index])

    # Compute pairwise distances between data points and the selected landmarks
    distances = tf.norm(data - sampled_landmarks[0], axis=1)

    for _ in range(1, n_samples):
        # Find the data point farthest from the selected landmarks
        farthest_index = tf.argmax(distances)
        farthest_point = data[farthest_index]

        # Update sampled landmarks and distances
        sampled_landmarks.append(farthest_point)
        distances = tf.minimum(distances, tf.norm(data - farthest_point, axis=1))

    # Convert sampled landmarks to tensor
    sampled_landmarks_tensor = tf.convert_to_tensor(sampled_landmarks)

    return sampled_landmarks_tensor


def spectral_sampling(data, n_samples):
    # Compute the kernel matrix using a Gaussian kernel
    kernel_matrix = tf.exp(-tf.norm(data[:, None] - data[None, :], axis=-1) ** 2)

    # Compute the eigenvectors and eigenvalues of the kernel matrix
    eigenvalues, eigenvectors = tf.linalg.eigh(kernel_matrix)

    # Select points corresponding to the top eigenvectors as landmarks
    indices = tf.argsort(eigenvalues, direction='DESCENDING')[:n_samples]
    sampled_landmarks = tf.gather(data, indices)

    return sampled_landmarks


def density_based_sampling(data, n_samples):
    # Convert data to TensorFlow tensor
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)

    # Apply DBSCAN to identify dense regions
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(data)

    # Find the indices of samples that belong to dense regions
    dense_indices = np.where(dbscan.labels_ != -1)[0]

    # Convert dense indices to TensorFlow tensor
    dense_indices_tensor = tf.constant(dense_indices, dtype=tf.int32)

    # Randomly select samples from dense regions
    sampled_indices = tf.random.shuffle(dense_indices_tensor)[:n_samples]

    # Extract sampled landmarks
    sampled_landmarks = tf.gather(data_tensor, sampled_indices)

    return sampled_landmarks


def hierarchical_clustering_sampling(data, n_samples):
    # Convert data to TensorFlow tensor
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)

    # Apply Agglomerative Clustering to build hierarchical clusters
    clustering = AgglomerativeClustering(n_clusters=n_samples, linkage='ward')
    clustering.fit(data)

    # Get the indices of cluster centers
    cluster_centers_indices = np.unique(clustering.labels_)

    # Randomly select cluster centers as landmarks
    sampled_cluster_centers = np.random.choice(cluster_centers_indices, size=n_samples, replace=False)

    # Convert sampled cluster centers to TensorFlow tensor
    sampled_cluster_centers_tensor = tf.constant(sampled_cluster_centers, dtype=tf.int32)

    # Extract sampled landmarks
    sampled_landmarks = tf.gather(data_tensor, sampled_cluster_centers_tensor)

    return sampled_landmarks


def entropy_based_sampling(data, n_samples):
    # Convert data to TensorFlow tensor
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)

    # Compute entropy for each data point
    entropy = -tf.reduce_sum(data_tensor * tf.math.log(data_tensor), axis=1)

    # Get indices of top entropy points
    top_indices = tf.argsort(entropy, direction='DESCENDING')[:n_samples]

    # Extract sampled landmarks
    sampled_landmarks = tf.gather(data_tensor, top_indices)

    return sampled_landmarks


def mutual_information_based_sampling(data, n_samples):
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)

    # Calculate mutual information between each pair of data points
    mutual_info = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            # Convert data slices to numpy arrays for mutual_info_regression
            mi_data_i = data[:, i].numpy().reshape(-1, 1)
            mi_data_j = data[:, j].numpy().reshape(-1, 1)
            mutual_info[:, i] += mutual_info_regression(mi_data_i, mi_data_j.ravel())  # Use ravel() here

    # Sum mutual information across features
    mutual_info_sum = tf.reduce_sum(mutual_info, axis=1)

    # Get indices of top mutual information points
    top_indices = tf.argsort(mutual_info_sum, direction='DESCENDING')[:n_samples]

    # Extract sampled landmarks
    sampled_landmarks = tf.gather(data_tensor, top_indices)

    return sampled_landmarks


def conditional_mutual_information_based_sampling(data, n_samples, n_neighbors=5):
    # Convert data to TensorFlow tensor
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)

    # Initialize NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)

    # Find nearest neighbors for each data point
    distances, indices = nbrs.kneighbors(data)

    # Calculate conditional mutual information
    cmi_values = []
    for i in range(data.shape[0]):
        # Calculate conditional entropy H(X|Y,Z) for each data point
        conditional_entropy = []
        for j in range(data.shape[0]):
            if j != i:  # Exclude the current data point
                # Calculate distances to neighbors excluding the current data point
                distances_j = distances[j][1:]  # Exclude the first element (distance to self)
                mean_distance_j = tf.reduce_mean(distances_j)

                # Find nearest neighbors of data point i excluding itself
                neighbors_i = nbrs.kneighbors([data[j]], return_distance=False)[0][
                              1:]  # Exclude the first element (index of self)

                # Calculate mean distance from data point i to its neighbors, conditioned on data point j
                mean_distance_i_given_j = tf.reduce_mean(tf.gather(distances[i], neighbors_i))

                # Calculate conditional entropy H(X|Y,Z)
                conditional_entropy.append(tf.math.log(mean_distance_i_given_j / mean_distance_j))

        # Calculate conditional mutual information I(X;Y|Z)
        cmi_values.append(tf.reduce_sum(conditional_entropy))

    # Get indices of top conditional mutual information points
    top_indices = tf.argsort(cmi_values, direction='DESCENDING')[:n_samples]

    # Extract sampled entries
    sampled_entries = tf.gather(data_tensor, top_indices)

    return sampled_entries


def joint_entropy_based_sampling(data, n_samples, subset_size):
    # Convert data to TensorFlow tensor
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)

    # Calculate number of subsets
    num_subsets = data.shape[0] - subset_size + 1

    # Calculate joint entropy for each subset
    joint_entropies = []
    for i in range(num_subsets):
        subset = data_tensor[i:i + subset_size]  # Extract subset
        # Compute joint entropy of the subset
        subset_entropy = -tf.reduce_sum(subset * tf.math.log(subset + 1e-10))  # Add a small epsilon to prevent log(0)
        joint_entropies.append(subset_entropy)

    # Find subsets with high joint entropy
    high_entropy_indices = tf.argsort(joint_entropies, direction='DESCENDING')[:n_samples]

    # Sample entries from high-entropy subsets
    sampled_entries = tf.gather(data_tensor, high_entropy_indices)

    return sampled_entries


def compute_entropy(labels):
    # Ensure labels is a 1D vector
    labels = tf.reshape(labels, [-1])
    unique_labels, _ = tf.unique(labels)
    label_counts = tf.math.bincount(unique_labels)
    probabilities = tf.cast(label_counts, tf.float32) / tf.cast(tf.size(labels), tf.float32)
    entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities))
    return entropy


def compute_information_gain(data_point, labels):
    # Assuming binary classification, modify as needed for other tasks
    # Split data point into two subsets based on a threshold (e.g., median value)
    threshold = tf.reduce_mean(data_point)
    subset1_indices = tf.where(data_point < threshold)
    subset2_indices = tf.where(data_point >= threshold)

    if tf.size(subset1_indices) == 0 or tf.size(subset2_indices) == 0:
        # If one of the subsets is empty, return 0 information gain
        return tf.constant(0.0)

    subset1_labels = tf.gather(tf.reshape(labels, [-1]), subset1_indices)
    subset2_labels = tf.gather(tf.reshape(labels, [-1]), subset2_indices)

    # Calculate entropy for original labels
    original_entropy = compute_entropy(labels)

    # Calculate weighted average of entropies of subsets
    subset1_entropy = compute_entropy(subset1_labels)
    subset2_entropy = compute_entropy(subset2_labels)

    scale1 = (tf.size(subset1_labels) / tf.size(labels))
    scale2 = (tf.size(subset2_labels) / tf.size(labels))
    # Calculate information gain (entropy reduction)
    information_gain = original_entropy - tf.cast(scale1, dtype=tf.float32) * subset1_entropy \
                       - tf.cast(scale2, dtype=tf.float32) * subset2_entropy

    return information_gain


def information_gain_based_sampling(data, labels, n_samples):
    # Calculate information gain for each data point
    information_gains = []
    for i in range(data.shape[0]):
        point = data[i]  # Extract data point
        information_gain = compute_information_gain(point, labels)
        # Append information gain to the list
        information_gains.append(information_gain)

    # Find data points with high information gain
    high_gain_indices = tf.argsort(information_gains, direction='DESCENDING')[:n_samples]

    return tf.gather(data, high_gain_indices)