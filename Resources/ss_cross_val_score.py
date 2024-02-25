from sklearn.metrics import check_scoring

from Resources import SSRepeatedKFold


def ss_cross_val_score(model, X, y, cv: SSRepeatedKFold, scoring='accuracy'):
    """
        Perform semi-supervised cross-validation with repeated k-fold splits and evaluate the model performance.

        This function performs semi-supervised cross-validation using the provided model and data. It splits the data
        into labeled, unlabeled, validation, and test sets according to the specified cross-validation strategy. The
        model is trained on the labeled and unlabeled data and evaluated on the validation and test sets using the
        specified scoring metric.

        Parameters:
        -----------
        - model: The machine learning model to be evaluated.
        - X (array-like): Feature matrix.
        - y (array-like): Target labels.
        - cv (SSRepeatedKFold): Cross-validation splitter.
        - scoring (str or callable): The scoring metric to use for evaluation. Default is 'accuracy'.

        Returns:
        -----------
        - tuple: A tuple containing lists of validation and test scores obtained from each cross-validation repeat.

        Notes:
        -----------
        - This function is specifically designed for semi-supervised learning scenarios.
    """
    val_scores = []
    test_scores = []

    scorer = check_scoring(model, scoring=scoring)

    for _ in range(cv.n_repeats):
        X_labeled, X_val, X_test, X_unlabeled, y_labeled, y_val, y_test, y_unlabeled = cv.split(X, y)
        model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
        val_pred = scorer(model, X_val, y_val)
        test_pred = scorer(model, X_test, y_test)
        val_scores.append(val_pred)
        test_scores.append(test_pred)

    return val_scores, test_scores
