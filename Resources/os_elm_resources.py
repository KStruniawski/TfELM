import tensorflow as tf


def proceed_features(batch):
    """
        Process features in a batch by stacking them into a matrix.

        This function takes a batch of features represented as a dictionary of tensor values and stacks them along the second
        axis to form a matrix.

        Parameters:
        -----------
            batch (Dict[str, tf.Tensor]): A dictionary containing feature tensors where the keys are feature names.

        Returns:
        -----------
            tf.Tensor: A matrix containing the stacked feature tensors.
    """
    tensors = list(batch.values())
    tensors = [tf.dtypes.cast(tensor, tf.float32) for tensor in tensors]
    # Stack the tensors along the second axis (axis=1) to form a matrix
    tensor_matrix = tf.stack(tensors, axis=1)
    return tensor_matrix


def split_dataset(dataset: tf.data.Dataset, test_data_fraction: float):
    """
        Split a dataset into training and validation datasets.

        This function splits a dataset into training and validation datasets using the provided fraction for validation.

        Parameters:
        -----------
            dataset (tf.data.Dataset): The input dataset to split.
            test_data_fraction (float): The fraction of the validation data as a float between 0 and 1.

        Returns:
        -----------
            Tuple[tf.data.Dataset, tf.data.Dataset]: A tuple containing the training and validation datasets.
    """

    test_data_percent = round(test_data_fraction * 100)
    if not (0 <= test_data_percent <= 100):
        raise ValueError("validation data fraction must be ∈ [0,1]")

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 > test_data_percent)
    test_dataset = dataset.filter(lambda f, data: f % 100 <= test_data_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    test_dataset = test_dataset.map(lambda f, data: data)

    return train_dataset, test_dataset


def normalize_with_moments(x, axes=[0, 1], epsilon=1e-8):
    """
        Normalize input tensor using mean and variance calculated along specified axes.

        This function normalizes the input tensor using mean and variance calculated along the specified axes.

        Parameters:
        -----------
            x (tf.Tensor): The input tensor to normalize.
            axes (List[int]): The axes along which to calculate the mean and variance. Default is [0, 1].
            epsilon (float): Small constant to avoid division by zero. Default is 1e-8.

        Returns:
        -----------
            tf.Tensor: The normalized tensor.
    """
    mean, variance = tf.nn.moments(x, axes=axes)
    x_normed = (x - mean) / tf.sqrt(variance + epsilon) # epsilon to avoid dividing by zero
    return x_normed