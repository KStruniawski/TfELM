import tensorflow as tf


def calculate_pairwise_distances(X, kernel_func):
    """
        Calculate pairwise distances between samples in the input data using the specified kernel function.

        This function calculates pairwise distances between samples in the input data `X` by calling
        the `calculate_pairwise_distances_vector` function with `X` as both inputs.

        Parameters:
        -----------
            X (tf.Tensor): Input data of shape (N, d) where N is the number of samples and d is the number of features.
            kernel_func (callable): Kernel function to be used for distance calculation.

        Returns:
        -----------
            tf.Tensor: Pairwise distances between samples in the input data.
    """
    return calculate_pairwise_distances_vector(X, X, kernel_func)


def calculate_pairwise_distances_vector(X1, X2, kernel_func):
    """
        Calculate pairwise distances between samples in two sets of input data using the specified kernel function.

        This function calculates pairwise distances between samples in two sets of input data `X1` and `X2` by applying
        the specified kernel function.

        Parameters:
        -----------
            X1 (tf.Tensor): Input data of shape (N1, d) where N1 is the number of samples in the first set and d is the number of features.
            X2 (tf.Tensor): Input data of shape (N2, d) where N2 is the number of samples in the second set and d is the number of features.
            kernel_func (callable): Kernel function to be used for distance calculation.

        Returns:
        -----------
            tf.Tensor: Pairwise distances between samples in the first set and the second set of input data.
    """
    # Calculate pairwise differences
    X1_exp = tf.expand_dims(X1, axis=1)  # Shape: (N1, 1, d1)
    X2_exp = tf.expand_dims(X2, axis=0)  # Shape: (1, N2, d2)

    # Apply kernel function to calculate distances
    distances = kernel_func(X1_exp, X2_exp)  # Shape: (N1, N2)

    return distances
