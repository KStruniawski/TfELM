import numpy as np


def generate_random_filters(filter_size, num_filters, num_input_channels):
    """
        Generate random orthogonal filters for convolutional neural networks.

        This function generates random filters with a given size and number of input channels,
        and then orthogonalizes them using singular value decomposition (SVD).

        Parameters:
        -----------
            filter_size (int): The size of the filter (height and width).
            num_filters (int): The number of filters to generate.
            num_input_channels (int): The number of input channels.

        Returns:
        -----------
            numpy.ndarray: A 4D array containing the generated orthogonal filters.
                The shape of the array is (filter_size, filter_size, num_input_channels, num_filters).
    """
    random_filters = np.random.normal(loc=0, scale=1, size=(filter_size, filter_size, num_input_channels, num_filters))
    flattened_filters = np.reshape(random_filters, (-1, num_filters))
    U, _, Vt = np.linalg.svd(flattened_filters, full_matrices=False)
    orthogonal_filters = np.dot(U, Vt)
    orthogonal_filters = np.reshape(orthogonal_filters, random_filters.shape)
    return orthogonal_filters
