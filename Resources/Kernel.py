import numpy as np
import tensorflow as tf


class Kernel:
    def __init__(self, kernel_name='rbf', param=1.0):
        """
            Initialize a kernel function object.

            Parameters:
            -----------
            - kernel_name (str): Name of the kernel function. Default is 'rbf'.
            - param (float): Parameter value for the kernel function. Default is 1.0.
        """
        self.kernel_name = kernel_name
        self.kernel_param = param
        self.ev = eval(f"lambda x1,x2: Kernel.{kernel_name}(x1, x2, {param})")

    @staticmethod
    def rbf(x, y, alpha=1.0):
        """
            Radial Basis Function (RBF) kernel.

            .. math::
                K(x, y) = \\exp\\left(-\\frac{{||x - y||^2}}{{2 \\alpha^2}}\\right)

            Parameters:
            -----------
            - x (tf.Tensor): Input tensor.
            - y (tf.Tensor): Input tensor.
            - alpha (float): Parameter value for the RBF kernel. Default is 1.0.

            Returns:
            tf.Tensor: Kernel values.
        """
        pairwise_diff = tf.subtract(x, y)
        squared_diff = tf.reduce_sum(tf.square(pairwise_diff), axis=-1)
        kernel_values = tf.exp(-squared_diff / (2 * alpha ** 2))
        return kernel_values

    @staticmethod
    def laplacian(x, y, alpha=1.0):
        """
            Laplacian kernel.

            .. math::
                K(x, y) = \\exp\\left(-\\frac{{||x - y||_1}}{{\\alpha}}\\right)

            Parameters:
            -----------
            - x (tf.Tensor): Input tensor.
            - y (tf.Tensor): Input tensor.
            - alpha (float): Parameter value for the Laplacian kernel. Default is 1.0.

            Returns:
            -----------
            tf.Tensor: Kernel values.
        """
        pairwise_diff = tf.subtract(x, y)
        abs_diff = tf.reduce_sum(tf.abs(pairwise_diff), axis=-1)
        kernel_values = tf.exp(-abs_diff / alpha)
        return kernel_values

    @staticmethod
    def sigmoid(x, y, alpha=1.0):
        """
            Sigmoid kernel.

            .. math::
                K(x, y) = \\tanh(\\alpha \\cdot \\langle x, y \\rangle)

            Parameters:
            -----------
            - x (tf.Tensor): Input tensor.
            - y (tf.Tensor): Input tensor.
            - alpha (float): Parameter value for the sigmoid kernel. Default is 1.0.

            Returns:
            -----------
            tf.Tensor: Kernel values.
        """
        kernel_values = tf.tanh(alpha * tf.reduce_sum(tf.multiply(x, y), axis=-1))
        return kernel_values

    @staticmethod
    def exponential(x, y, alpha=1.0):
        """
            Exponential kernel.

            .. math::
                K(x, y) = \\exp\\left(-\\alpha \\sqrt{||x - y||^2}\\right)

            Parameters:
            -----------
            - x (tf.Tensor): Input tensor.
            - y (tf.Tensor): Input tensor.
            - alpha (float): Parameter value for the exponential kernel. Default is 1.0.

            Returns:
            -----------
            tf.Tensor: Kernel values.
        """
        pairwise_diff = tf.subtract(x, y)
        squared_diff = tf.reduce_sum(tf.square(pairwise_diff), axis=-1)
        kernel_values = tf.exp(-tf.sqrt(squared_diff) * alpha)
        return kernel_values

    @staticmethod
    def cosine(x, y, alpha=None):
        """
            Cosine similarity kernel.

            Computes the cosine similarity between two vectors.

            .. math::
                K(x, y) = \\frac{{\\langle x, y \\rangle}}{{||x|| \\cdot ||y||}}

            Parameters:
            -----------
            - x (tf.Tensor): Input tensor.
            - y (tf.Tensor): Input tensor.
            - alpha (None): This parameter is not used. It's included for compatibility with other kernel functions.

            Returns:
            -----------
            tf.Tensor: Kernel values representing cosine similarity between x and y.
        """
        x_norm = tf.norm(x, axis=-1)
        y_norm = tf.norm(y, axis=-1)
        dot_product = tf.reduce_sum(tf.multiply(x, y), axis=-1)
        kernel_values = dot_product / (x_norm * y_norm)
        return kernel_values

    @staticmethod
    def morlet_wavelet(x, y, alpha=1.0):
        """
            Morlet wavelet kernel.

            .. math::
                K(x, y) = \\cos\\left(2\\pi\\frac{||x - y||^2}{\\alpha}\\right)

            Parameters:
            -----------
            - x (tf.Tensor): Input tensor.
            - y (tf.Tensor): Input tensor.
            - alpha (float): Parameter value for the Morlet wavelet kernel. Default is 1.0.

            Returns:
            -----------
            tf.Tensor: Kernel values.
        """
        pairwise_diff = tf.subtract(x, y)
        squared_diff = tf.reduce_sum(tf.square(pairwise_diff), axis=-1)
        wavelet_values = tf.cos(2 * np.pi * squared_diff / alpha)
        return wavelet_values

    @staticmethod
    def mexican_hat_wavelet(x, y, alpha=1.0):
        """
            Mexican hat wavelet kernel.

            .. math::
                K(x, y) = \\left(1 - \\left(\\frac{||x - y||^2}{\\alpha}\\right)^2\\right)
                            \\exp\\left(-\\frac{||x - y||^2}{2\\alpha^2}\\right)

            Parameters:
            -----------
            - x (tf.Tensor): Input tensor.
            - y (tf.Tensor): Input tensor.
            - alpha (float): Parameter value for the Mexican hat wavelet kernel. Default is 1.0.

            Returns:
            -----------
            tf.Tensor: Kernel values.
        """
        pairwise_diff = tf.subtract(x, y)
        squared_diff = tf.reduce_sum(tf.square(pairwise_diff), axis=-1)
        wavelet_values = (1 - (squared_diff / alpha) ** 2) * tf.exp(-squared_diff / (2 * alpha ** 2))
        return wavelet_values

    @staticmethod
    def haar_wavelet(x, y, alpha=1.0):
        """
            Haar wavelet kernel.

            Parameters:
            -----------
            - x (tf.Tensor): Input tensor.
            - y (tf.Tensor): Input tensor.
            - alpha (float): Parameter value for the Haar wavelet kernel. Default is 1.0.

            Returns:
            -----------
            tf.Tensor: Kernel values.
        """
        pairwise_diff = tf.subtract(x, y)
        abs_diff = tf.abs(pairwise_diff)
        wavelet_values = tf.where(abs_diff < alpha / 2.0, 1.0, 0.0)
        wavelet_matrix = tf.reduce_sum(wavelet_values, axis=-1)  # Compute the sum along the last axis
        return wavelet_matrix

    @staticmethod
    def rational_quadratic(x, y, alpha=1.0):
        """
            Rational quadratic kernel.

            .. math::
                K(x, y) = 1 - \\frac{||x - y||^2}{||x - y||^2 + \\alpha}

            Parameters:
            -----------
            - x (tf.Tensor): Input tensor.
            - y (tf.Tensor): Input tensor.
            - alpha (float): Parameter value for the rational quadratic kernel. Default is 1.0.

            Returns:
            -----------
            tf.Tensor: Kernel values.
        """
        pairwise_diff = tf.subtract(x, y)
        squared_diff = tf.reduce_sum(tf.square(pairwise_diff), axis=-1)
        kernel_values = 1.0 - (squared_diff / (squared_diff + alpha))
        return kernel_values


class CombinedSumKernel(Kernel):
    """
        Initialize a combined sum kernel object.

        Parameters:
        -----------
        - kernels (list of Kernel objects): List of individual kernel objects to be combined.
    """
    def __init__(self, kernels):
        super().__init__()
        names = []
        params = []
        for k in kernels:
            names.append(k.kernel_name)
            params.append(k.kernel_param)
        self.kernel_name = names
        self.kernel_param = params
        self.ev = lambda x1, x2: self.__sum_kernel(x1, x2, kernels)

    @staticmethod
    def __sum_kernel(x, y, kernels):
        """
            Compute the sum kernel value for a given pair of input tensors.

            Parameters:
            -----------
            - x (tf.Tensor): Input tensor.
            - y (tf.Tensor): Input tensor.
            - kernels (list of Kernel objects): List of individual kernel objects.

            Returns:
            -----------
            tf.Tensor: Sum kernel values.
        """
        kernel_values = tf.reduce_sum([kernel.ev(x, y) for kernel in kernels], axis=0)
        return kernel_values


class CombinedProductKernel(Kernel):
    """
        Initialize a combined product kernel object.

        Parameters:
        -----------
        - kernels (list of Kernel objects): List of individual kernel objects to be combined.
    """
    def __init__(self, kernels):
        super().__init__()
        names = []
        params = []
        for k in kernels:
            names.append(k.kernel_name)
            params.append(k.kernel_param)
        self.kernel_name = names
        self.kernel_param = params
        self.ev = lambda x1, x2: self.__product_kernel(x1, x2, kernels)

    @staticmethod
    def __product_kernel(x, y, kernels):
        """
            Compute the product kernel value for a given pair of input tensors.

            Parameters:
            -----------
            - x (tf.Tensor): Input tensor.
            - y (tf.Tensor): Input tensor.
            - kernels (list of Kernel objects): List of individual kernel objects.

            Returns:
            -----------
            tf.Tensor: Product kernel values.
        """
        kernel_values = tf.reduce_prod([kernel.ev(x, y) for kernel in kernels], axis=0)
        return kernel_values
