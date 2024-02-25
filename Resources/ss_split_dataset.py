import random


def ss_split_dataset(X, y, labeled_size, val_size, test_size, random_state=None):
    """
        Split a dataset into labeled, validation, test, and unlabeled subsets.

        Parameters:
        -----------
        - X (numpy.ndarray): Input features of the dataset.
        - y (numpy.ndarray): Target labels of the dataset.
        - labeled_size (int): Number of samples to include in the labeled subset.
        - val_size (int): Number of samples to include in the validation subset.
        - test_size (int): Number of samples to include in the test subset.
        - random_state (int or None): Seed for the random number generator.

        Returns:
        -----------
        tuple: A tuple containing the following arrays in the order specified:
            - X_labeled (numpy.ndarray): Features of the labeled subset.
            - X_val (numpy.ndarray): Features of the validation subset.
            - X_test (numpy.ndarray): Features of the test subset.
            - X_unlabeled (numpy.ndarray): Features of the unlabeled subset.
            - y_labeled (numpy.ndarray): Labels of the labeled subset.
            - y_val (numpy.ndarray): Labels of the validation subset.
            - y_test (numpy.ndarray): Labels of the test subset.
            - y_unlabeled (numpy.ndarray): Labels of the unlabeled subset.
    """
    total_size = len(X)

    indices = list(range(total_size))
    if random_state is not None:
        random.seed(random_state)
    random.shuffle(indices)

    labeled_indices = indices[:labeled_size]
    val_indices = indices[labeled_size:(labeled_size + val_size)]
    test_indices = indices[(labeled_size + val_size):(labeled_size + val_size + test_size)]
    unlabeled_indices = indices[(labeled_size + val_size + test_size):]

    X_labeled, y_labeled = X[labeled_indices], y[labeled_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    X_unlabeled, y_unlabeled = X[unlabeled_indices], y[unlabeled_indices]

    return X_labeled, X_val, X_test, X_unlabeled, y_labeled, y_val, y_test, y_unlabeled