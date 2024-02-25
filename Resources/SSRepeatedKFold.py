from Resources.ss_split_dataset import ss_split_dataset


class SSRepeatedKFold:
    """
        Repeated semi-supervised k-fold cross-validation splitter.

        This class implements a repeated k-fold cross-validation strategy specifically designed for semi-supervised
        learning scenarios. It repeatedly splits the data into labeled, unlabeled, validation, and test sets according
        to the specified sizes.

        Parameters:
        -----------
        - n_splits (tuple): Tuple containing the sizes of labeled, unlabeled, validation, and test sets, respectively.
        - n_repeats (int): Number of times cross-validation should be repeated.

        Attributes:
        -----------
        - labeled_size (int): Size of the labeled set in each split.
        - unlabeled_size (int): Size of the unlabeled set in each split.
        - val_size (int): Size of the validation set in each split.
        - test_size (int): Size of the test set in each split.
        - n_repeats (int): Number of times cross-validation should be repeated.

        Methods:
        -----------
        - split(X, y): Splits the input data into labeled, unlabeled, validation, and test sets.

        Notes:
        -----------
        - This splitter is specifically designed for semi-supervised learning scenarios.
    """
    def __init__(self, n_splits, n_repeats):
        self.labeled_size = n_splits[0]
        self.unlabeled_size = n_splits[1]
        self.val_size = n_splits[2]
        self.test_size = n_splits[3]
        self.n_repeats = n_repeats

    def split(self, X, y):
        """
            Generate indices to split data into labeled, unlabeled, validation, and test sets.

            Parameters:
            -----------
            - X (array-like): Feature matrix.
            - y (array-like): Target labels.

            Returns:
            -----------
            - tuple: A tuple containing indices for labeled, unlabeled, validation, and test sets.
        """
        return ss_split_dataset(X, y, self.labeled_size, self.unlabeled_size, self.val_size, self.test_size)

