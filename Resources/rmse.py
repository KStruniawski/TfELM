import numpy as np


def calculate_rmse(y_true, y_pred):
    """
        Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

        The RMSE is a measure of the differences between values predicted by a model and the actual observed values.

        Parameters:
        -----------
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.

        Returns:
        -----------
            float: The RMSE value.
    """

    # Compute the squared differences
    squared_diff = (y_true - y_pred) ** 2
    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)
    # Take the square root to obtain the RMSE
    rmse = np.sqrt(mean_squared_diff)
    return rmse