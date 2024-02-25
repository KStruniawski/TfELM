import numpy as np


def labelize_images(X, y):
    """
        Convert labels into one-hot encoded vectors.

        This function takes an array of images (X) and their corresponding labels (y),
        and converts the labels into one-hot encoded vectors.

        Parameters:
        -----------
        - X (numpy.ndarray): Array of images with shape (num_samples, image_size).
        - y (numpy.ndarray): Array of labels with shape (num_samples,).

        Returns:
        -----------
        numpy.ndarray: Modified array of images with one-hot encoded labels.
    """
    num_samples = X.shape[0]
    image_size = X.shape[1]
    num_classes = np.max(y) + 1

    modified_X = np.copy(X)

    for i in range(num_samples):
        label = y[i]
        # Switch first d pixels to 0
        modified_X[i, :num_classes] = 0
        # Switch the pixel corresponding to the class to 1
        modified_X[i, label] = 1
    return modified_X


def unlabelize_images(X, num_classes):
    """
        Convert one-hot encoded vectors back into labels.

        This function takes an array of images with one-hot encoded labels and converts
        them back into their original labels.

        Parameters:
        -----------
        - X (numpy.ndarray): Array of images with one-hot encoded labels.
        - num_classes (int): Number of classes.

        Returns:
        -----------
        numpy.ndarray: Array of labels.
    """
    y = np.zeros(X.shape[0], dtype=np.int32)

    for i in range(X.shape[0]):
        # Find the index of the first pixel that corresponds to each class
        class_index = np.argmax(X[i, :num_classes])
        y[i] = class_index
    return y
