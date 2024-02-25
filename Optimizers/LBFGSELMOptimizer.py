import numpy as np
from scipy.optimize import minimize
from Optimizers.ELMOptimizer import ELMOptimizer
import tensorflow as tf


class LBFGSELMOptimizer(ELMOptimizer):
    """
        Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) optimizer for ELM with given regularization.

        This optimizer applies the (L-BFGS) algorithm for optimizing the beta weights with given regularization.

        Attributes:
        -----------
        - num_iter (int): Number of iterations. Defaults to 100.
        - alpha (float): Learning rate. Defaults to 1e-03.
        - optimizer_loss_name (str): Name of the optimizer loss function.
        - optimizer_loss (function): Optimizer loss function based on the specified loss type.

        Methods:
        -----------
        - optimize(beta, H, y): Optimizes the beta weights using the (L-BFGS) algorithm.

        Note:
        -----------
        Inherits from ELMOptimizer and implements the abstract method optimize.

        Examples:
        -----------
        Initialize optimizer (l1 norm)

        >>> optimizer = LBFGSELMOptimizer(optimizer_loss='l1', optimizer_loss_reg=[0.01])
    """
    def __init__(self, num_iter=100, alpha=1e-03, optimizer_loss=None, optimizer_loss_reg=None):
        if optimizer_loss_reg is None:
            optimizer_loss_reg = [1.0]
        self.num_iter = num_iter
        self.alpha = alpha
        self.optimizer_loss_name = optimizer_loss
        if optimizer_loss_reg is None:
            optimizer_loss_reg = [1.0]
        if optimizer_loss == 'l1':
            self.optimizer_loss = lambda x: ELMOptimizer.l1_loss(x, optimizer_loss_reg)
        elif optimizer_loss == 'l2':
            self.optimizer_loss = lambda x: ELMOptimizer.l2_loss(x, optimizer_loss_reg)
        else:
            self.optimizer_loss = lambda x: ELMOptimizer.l12_loss(x, optimizer_loss_reg[0], optimizer_loss_reg[1])

    def optimize(self, beta, H, y):
        """
        Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) for given regularization.

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

        >>> optimizer = LBFGSELMOptimizer(optimizer_loss='l1', optimizer_loss_reg=[0.01])
        """
        HtH = tf.matmul(H, H, transpose_a=True)
        L = tf.linalg.eigvalsh(HtH)
        L = tf.reduce_max(L)
        errors_history = []
        beta_shape = beta.numpy().shape

        def scipy_objective(beta_np):
            return np.float64(self.optimizer_loss(beta_np).numpy()[0])

        def scipy_gradient(beta_np):
            beta_tf = tf.convert_to_tensor(beta_np.reshape(beta_shape), dtype=tf.float32)
            gradient_tf = -2 / L * tf.matmul(H, tf.matmul(H, beta_tf) - y, transpose_a=True)
            return np.float64(gradient_tf.numpy().flatten())

        result = minimize(scipy_objective, beta.numpy().flatten(), jac=scipy_gradient, method='L-BFGS-B')

        beta_optimized = tf.convert_to_tensor(result.x.reshape(beta_shape), dtype=tf.float32)
        errors_history.append(result.fun)

        return beta_optimized, errors_history