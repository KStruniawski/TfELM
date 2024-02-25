from abc import ABC, abstractmethod
import tensorflow as tf


class ELMOptimizer(ABC):
    """
        Abstract base class for ELM optimizers.

        This class defines common methods for ELM optimizers.

        Methods:
        -----------
        - l1_loss(x, reg=1.0): Computes the L1 loss.
        - l2_loss(x, reg=1.0): Computes the L2 loss.
        - l12_loss(x, reg_l1=1.0, reg_l2=1.0): Computes the combined L1 and L2 loss.
        - optimize(beta, H, y): Optimizes the beta weights.

        Note:
        -----------
        Subclasses must implement the optimize method.

        Examples:
        -----------
        Initialize optimizer (l1 norm)

        >>> optimizer = ISTAELMOptimizer(optimizer_loss='l1', optimizer_loss_reg=[0.01])

        Initialize a Regularized Extreme Learning Machine (ELM) layer with optimizer

        >>> elm = ELMLayer(number_neurons=num_neurons, activation='mish', beta_optimizer=optimizer)
        >>> model = ELMModel(elm)

        Fit the ELM model to the entire dataset

        >>> model.fit(X, y)
    """
    @staticmethod
    def l1_loss(x, reg=1.0):
        """
            Computes the L1 loss.

            Parameters:
            -----------
            - x: Input tensor.
            - reg (float): Regularization parameter. Defaults to 1.0.

            Returns:
            -----------
            - L1 loss.

            Examples:
            -----------
            Initialize optimizer (l1 norm)

            >>> optimizer = ISTAELMOptimizer(optimizer_loss='l1', optimizer_loss_reg=[0.01])
        """
        return reg * tf.reduce_sum(tf.abs(x))

    @staticmethod
    def l2_loss(x, reg=1.0):
        """
            Computes the L2 loss.

            Parameters:
            -----------
            - x: Input tensor.
            - reg (float): Regularization parameter. Defaults to 1.0.

            Returns:
            -----------
            - L2 loss.

            Examples:
            -----------
            Initialize optimizer (l2 norm)

            >>> optimizer = ISTAELMOptimizer(optimizer_loss='l2', optimizer_loss_reg=[0.01])
        """
        return reg * tf.reduce_sum(tf.abs(x))**2

    @staticmethod
    def l12_loss(x, reg_l1=1.0, reg_l2=1.0):
        """
            Computes the combined L1 and L2 loss.

            Parameters:
            -----------
            - x: Input tensor.
            - reg_l1 (float): L1 regularization parameter. Defaults to 1.0.
            - reg_l2 (float): L2 regularization parameter. Defaults to 1.0.

            Returns:
            -----------
            - Combined L1 and L2 loss.

            Examples:
            -----------
            Initialize optimizer (l2 norm)

            >>> optimizer = ISTAELMOptimizer(optimizer_loss='l2', optimizer_loss_reg=[0.01, 0.05])
        """
        return ELMOptimizer.l1_loss(x, reg_l1) + ELMOptimizer.l2_loss(x, reg_l2)

    @abstractmethod
    def optimize(self, beta, H, y):
        """
            Optimizes the beta weights.

            Parameters:
            - beta: Beta weights tensor.
            - H: Feature map tensor.
            - y: Target tensor.

            Returns:
            - Optimized beta weights tensor.
        """
        pass
