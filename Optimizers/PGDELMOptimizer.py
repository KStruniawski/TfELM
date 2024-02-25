from Optimizers.ELMOptimizer import ELMOptimizer
import tensorflow as tf
from Resources.proximal_operator import proximal_operator


class PGDELMOptimizer(ELMOptimizer):
    """
        Proximal Gradient Descent (PGD) optimizer for ELM with given regularization.

        This optimizer applies the (PGD) algorithm for optimizing the beta weights with given regularization.

        Attributes:
        -----------
        - num_iter (int): Number of iterations. Defaults to 100.
        - alpha (float): Learning rate. Defaults to 1e-03.
        - optimizer_loss_name (str): Name of the optimizer loss function.
        - optimizer_loss (function): Optimizer loss function based on the specified loss type.

        Methods:
        -----------
        - optimize(beta, H, y): Optimizes the beta weights using the (PGD) algorithm.

        Note:
        -----------
        Inherits from ELMOptimizer and implements the abstract method optimize.

        Examples:
        -----------
        Initialize optimizer (l1 norm)

        >>> optimizer = PGDELMOptimizer(optimizer_loss='l1', optimizer_loss_reg=[0.01])
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
        Proximal Gradient Descent (PGD) for given regularization.

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

        >>> optimizer = PGDELMOptimizer(optimizer_loss='l1', optimizer_loss_reg=[0.01])
        """
        HtH = tf.matmul(H, H, transpose_a=True)
        L = tf.linalg.eigvalsh(HtH)
        L = tf.reduce_max(L)
        beta_j = beta
        errors_history = []

        for j in range(1, self.num_iter):
            gradient = -2 / L * tf.matmul(H, tf.matmul(H, beta_j) - y, transpose_a=True)
            beta_j = proximal_operator(beta_j - self.alpha * gradient, self.alpha)
            err = self.optimizer_loss(beta_j)
            errors_history.append(err.numpy())

        return beta_j, errors_history



