from Optimizers.ELMOptimizer import ELMOptimizer
import tensorflow as tf
from Resources.proximal_operator import proximal_operator


class FISTADELMOptimizer(ELMOptimizer):
    """
        Fast Iterative Soft Thresholding Algorithm (FISTA) optimizer for ELM with given regularization.

        This optimizer applies the FISTA algorithm for optimizing the beta weights with given regularization.

        Attributes:
        -----------
        - num_iter (int): Number of iterations. Defaults to 100.
        - alpha (float): Learning rate. Defaults to 1e-03.
        - optimizer_loss_name (str): Name of the optimizer loss function.
        - optimizer_loss (function): Optimizer loss function based on the specified loss type.

        Methods:
        -----------
        - optimize(beta, H, y): Optimizes the beta weights using the FISTA algorithm.

        Note:
        -----------
        Inherits from ELMOptimizer and implements the abstract method optimize.

        Examples:
        -----------
        Initialize optimizer (l1 norm)

        >>> optimizer = FISTAELMOptimizer(optimizer_loss='l1', optimizer_loss_reg=[0.01])

        Initialize a Regularized Extreme Learning Machine (ELM) layer with optimizer

        >>> elm = ELMLayer(number_neurons=num_neurons, activation='mish', beta_optimizer=optimizer)
        >>> model = ELMModel(elm)

        Fit the ELM model to the entire dataset

        >>> model.fit(X, y)
    """
    def __init__(self, num_iter=100, alpha=1e-03, optimizer_loss=None, optimizer_loss_reg=None):
        if optimizer_loss_reg is None:
            optimizer_loss_reg = [1.0]
        self.num_iter = num_iter
        self.alpha = alpha
        self.optimizer_loss_name = optimizer_loss
        if optimizer_loss == 'l1':
            self.optimizer_loss = lambda x: ELMOptimizer.l1_loss(x, optimizer_loss_reg)
        elif optimizer_loss == 'l2':
            self.optimizer_loss = lambda x: ELMOptimizer.l2_loss(x, optimizer_loss_reg)
        else:
            self.optimizer_loss = lambda x: ELMOptimizer.l12_loss(x, optimizer_loss_reg[0], optimizer_loss_reg[1])

    def optimize(self, beta, H, y):
        """
        Fast Iterative Soft Thresholding Algorithm (FISTA) for given regularization.

        Parameters:
        -----------
        - beta (tf.Tensor): Initial weights.
        - H (tf.Tensor): Input feature map.
        - y (tf.Tensor): Output data.

        Returns:
        -----------
        tuple: Final weights and a list of error history during optimization.

        Examples:
        -----------
        Initialize optimizer (l1 norm)

        >>> optimizer = FISTAELMOptimizer(optimizer_loss='l1', optimizer_loss_reg=[0.01])
        """
        HtH = tf.matmul(H, H, transpose_a=True)
        L = tf.linalg.eigvalsh(HtH)
        L = tf.reduce_max(L)
        beta_j = beta
        errors_history = []

        t_k = 1.0
        t_k_prev = 1.0

        for j in range(1, self.num_iter):
            gradient = -2 / L * tf.matmul(H, tf.matmul(H, beta_j) - y, transpose_a=True)
            beta_new = proximal_operator(beta_j - self.alpha * gradient, self.alpha)
            t_k = (1 + tf.sqrt(1 + 4 * t_k ** 2)) / 2
            beta_j = beta_new + ((t_k_prev - 1) / t_k) * (beta_new - beta_j)
            t_k_prev = t_k

            err = self.optimizer_loss(beta_j)
            errors_history.append(err.numpy())

        return beta_j, errors_history

