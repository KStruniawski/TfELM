import numpy as np
import tensorflow as tf


class ActivationFunction:
    """
        A class containing various activation functions.

        Attributes:
        -----------
        - act_param (float): The parameter used by some activation functions. Defaults to 1.0.
        - act_param2 (float): The second parameter used by some activation functions. Defaults to 1.0.
        - knots (list): A list of knots used by the cubic spline function. Defaults to [1, 1, 1, 1, 1].

        Methods:
        -----------
        - identity(x): Identity function.
        - sigmoid(x): Sigmoid function.
        - tanh(x): Hyperbolic tangent function.
        - relu(x): Rectified Linear Unit (ReLU) function.
        - leaky_relu(x): Leaky ReLU function.
        - prelu(x): Parametric ReLU function.
        - elu(x): Exponential Linear Unit (ELU) function.
        - softplus(x): Softplus function.
        - bent_identity(x): Bent Identity function.
        - gaussian(x): Gaussian function.
        - sinusoidal(x): Sinusoidal function.
        - isru(x): Inverse Square Root Unit (ISRU) function.
        - isrlu(x): Inverse Square Root Linear Unit (ISRLU) function.
        - selu(x): Scaled Exponential Linear Unit (SELU) function.
        - softmax(x): Softmax function.
        - ssigmoid(x): Symmetric Sigmoid function.
        - silu(x): SiLU (Swish) function.
        - gelu(x): Gaussian Error Linear Units (GELU) function.
        - log(x): Logarithmic function.
        - cube(x): Cubic function.
        - inverse(x): Inverse function.
        - swish(x): Swish function.
        - mish(x): Mish function.
        - bis(x): Bent Identity Smoothed function.
        - gompertz(x): Gompertz function.
        - elliott(x): Elliott function.
        - isq(x): Inverse Square function.
        - hardshrink(x): Hard Shrink function.
        - softshrink(x): Soft Shrink function.
        - sqrelu(x): Squared Rectified Linear Unit (SQReLU) function.
        - sine(x): Sine function.
        - softexp(x): Soft Exponential function.
        - arctan(x): Arctan function.
        - sin_transfer(x): Sinusoidal Transfer function.
        - hsigmoid(x): Hard Sigmoid function.
        - tsigmoid(x): Tangent Sigmoid function.
        - arcsinh(x): ArcSinH function.
        - logit(x): Logit function.
        - tlu(x): Truncated Linear Unit (TLU) function.
        - aq(x): Asymmetric Quadratic function.
        - logsigmoid(x): Logarithmic Sigmoid function.
        - cosine(x): Cosine function.
        - relu_cos(x): Rectified Cosine function.
        - imq(x): Inverse Multiquadratic function.
        - cos_sigmoid(x): Cosine Sigmoid function.
        - triangular(x): Triangular function.
        - hardtanh(x): Hard Tanh function.
        - inverse_sine(x): Inverse Sine function.
        - bezier(x): Quadratic Bezier function.
        - bsigmoid(x): Bipolar Sigmoid function.
        - power(x, a=1.0): Power function.
        - gswish(x): Gaussian Swish function.
        - invgamma(x): Inverse Gamma function.
        - softclip(x): Soft Clip function.
        - inverse_cosine(x): Inverse Cosine function.
        - sinusoid(x): Sinusoid function.
        - inv_logit(x): Inverse Logit function.
        - soft_exponential(x): Soft Exponential function.
        - srelu(x): Smooth Rectified Linear Unit (SReLU) function.
        - inverse_tangent(x): Inverse Tangent function.
        - hswish(x): Hard Swish function.
        - aqrelu(x): Asymmetric Quadratic ReLU function.
        - gelu2(x): Gaussian Error Linear Units 2 (GELU2) function.
        - sinusoid2(x): Sinusoid 2 function.
        - inverse_tanh(x): Inverse Hyperbolic Tangent function.
        - leaky_softplus(x): Leaky Softplus function.
        - gaussian_tangent(x): Gaussian Tangent function.
        - exp_cosine(x): Exponential Cosine function.
        - gaussian_cdf(x): Gaussian Cumulative Distribution Function (CDF) function.
        - hmish(x): Hard-Mish function.
        - smooth_sigmoid(x): Smooth Sigmoid function.
        - log_exp(x): Logarithm of Exponential function.
        - cubic(x): Cubic function.
        - exp_sine(x): Exponential Sine function.
        - sym_sigmoid(x): Symmetric Sigmoid function.
        - square(x): Squared function.
        - soft_clipping(x): Soft Clipping function.
        - swish_gaussian(x): Swish-Gaussian function.
        - hard_shrink(x): Hard Shrink function.
        - smooth_hard_tanh(x): Smooth Hard Tanh function.
        - bipolar_sigmoid(x): Bipolar Sigmoid function.
        - log_sigmoid(x): Logarithmic Sigmoid function.
        - hard_sigmoid(x): Hard Sigmoid function.
        - invsqrt(x): Inverse Square Root function.
        - gauss_tanh(x): Gaussian Tangent Hyperbolic function.
        - egaulu(x): EGAULU function.
        - logarithm(x): Logarithm function.
        - inv_sine(x): Inverse Sine function.
        - hard_tanh(x): Hard Tanh function.
        - bent_identity_smoothed(x): Bent Identity Smoothed function.
        - pos_softplus(x): Positive Softplus function.
        - inv_multiquadratic(x): Inverse Multiquadratic function.
        - inv_cosine(x): Inverse Cosine function.
        - asymmetric_gaussian(x): Asymmetric Gaussian function.
        - inv_quadratic(x): Inverse Quadratic function.
        - gaussian_squared(x): Gaussian Squared function.
        - symmetric_sigmoid(x): Symmetric Sigmoid function.
        - inv_cubic(x): Inverse Cubic function.
        - cauchy(x): Cauchy function.
        - exponential_quadratic(x): Exponential Quadratic function.
        - rational_quadratic(x): Rational Quadratic function.
        - cubic_spline(x): Cubic Spline function.
        - symmetric_soft_clipping(x): Symmetric Soft Clipping function.
        - binary_step(x): Binary Step function.
        - imrbf(x): Inverse Multiquadratic Radial Basis Function (IMRBF) function.
        - cloglog(x): Complementary Log-Log (cLogLog) function.
        - nrelu(x): Noisy Rectified Linear Unit (NReLU) function.
    """
    def __init__(self, act_name="", act_param=1.0, act_param2=1.0, knots=None):
        """
            Initialize the ActivationFunction.

            Parameters:
            - act_name (str): Name of the activation function. Defaults to "".
            - act_param (float): The parameter used by some activation functions. Defaults to 1.0.
            - act_param2 (float): The second parameter used by some activation functions. Defaults to 1.0.
            - knots (list): A list of knots used by the cubic spline function. Defaults to [1, 1, 1, 1, 1].
        """
        if knots is None:
            knots = [1, 1, 1, 1, 1]
        self.act_param = act_param
        self.act_param2 = act_param2
        self.knots = knots

    @staticmethod
    def identity(x):
        """Identity function."""
        return x

    @staticmethod
    def sigmoid(x):
        """Sigmoid function."""
        return tf.sigmoid(x)

    @staticmethod
    def tanh(x):
        """Hyperbolic tangent function."""
        return tf.tanh(x)

    @staticmethod
    def relu(x):
        """Rectified Linear Unit (ReLU) function."""
        return tf.keras.activations.relu(x)

    # Leaky ReLU Function:
    def leaky_relu(self, x):
        """Leaky Rectified Linear Unit (ReLU) function."""
        return tf.keras.layers.LeakyReLU(self.act_param)(x)

    # Parametric ReLU Function:
    def prelu(self, x):
        """Parametric Rectified Linear Unit (ReLU) function."""
        return tf.keras.layers.PReLU(alpha_initializer=tf.constant_initializer(self.act_param))(x)

    # Exponential Linear Unit (ELU) Function:
    def elu(self, x):
        """Exponential Linear Unit (ELU) function."""
        return tf.keras.activations.elu(x, alpha=self.act_param)

    # SoftPlus Function:
    @staticmethod
    def softplus(x):
        """SoftPlus function."""
        return tf.keras.activations.softplus(x)

    # Bent Identity Function:
    @staticmethod
    def bent_identity(x):
        """Bent Identity function."""
        return tf.keras.activations.linear(tf.sqrt(x ** 2.0 + 1.0) - 1.0 + x)

    # Gaussian Function:
    @staticmethod
    def gaussian(x):
        """Gaussian function."""
        return tf.exp(-tf.square(x))

    # Sinusoidal Function:
    @staticmethod
    def sinusoidal(x):
        """Sinusoidal function."""
        return tf.sin(x)

    # Inverse Square Root Unit (ISRU) Function:
    def isru(self, x):
        """Inverse Square Root Unit (ISRU) function."""
        return tf.math.divide(x, tf.math.sqrt(1 + self.act_param * tf.square(x)))

    # Inverse Square Root Linear Unit (ISRLU) Function:
    def isrlu(self, x):
        """Inverse Square Root Linear Unit (ISRLU) function."""
        return tf.where(x < 0.0, x / tf.sqrt(1.0 + self.act_param * tf.square(x)), x)

    # Scaled Exponential Linear Unit (SELU) Function:
    def selu(self, x):
        """Scaled Exponential Linear Unit (SELU) function."""
        return self.act_param2 * tf.where(x > 0.0, x, self.act_param * (tf.exp(x) - 1.0))

    # Softmax Function
    @staticmethod
    def softmax(x):
        """Softmax function."""
        return tf.keras.activations.softmax(x)

    # Symmetric Sigmoid Function
    def ssigmoid(self, x):
        """Symmetric Sigmoid function."""
        return (2.0 / (1.0 + tf.exp(-self.act_param * x))) - 1.0

    # SiLU (Swish) Function:
    @staticmethod
    def silu(x):
        """SiLU (Swish) function."""
        return tf.keras.activations.swish(x)

    # Gaussian Error Linear Units (GELU) Function:
    @staticmethod
    def gelu(x):
        """Gaussian Error Linear Units (GELU) function."""
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

    # Logarithmic Function:
    @staticmethod
    def log(x):
        """Logarithmic function."""
        return tf.math.log(x + 1e-10)

    @staticmethod
    def cube(x):
        """Cube function."""
        return tf.math.pow(x, 3.0)

    @staticmethod
    def inverse(x):
        """Inverse function."""
        return tf.math.divide(1.0, x)

    # Swish Function:
    def swish(self, x):
        """Swish function."""
        return x * tf.keras.activations.sigmoid(self.act_param * x)

    # Mish Function:
    @staticmethod
    def mish(x):
        """Mish function."""
        return x * tf.keras.activations.tanh(tf.keras.activations.softplus(x))

    # Bent Identity Smoothed Function:
    @staticmethod
    def bis(x):
        """Bent Identity Smoothed function."""
        return tf.keras.activations.softplus(x) - 0.5

    # Gompertz Function:
    @staticmethod
    def gompertz(x):
        """Gompertz function."""
        return tf.exp(-tf.exp(-x))

    # Elliott Function:
    @staticmethod
    def elliott(x):
        """Elliott function."""
        return x / (1.0 + tf.abs(x))

    # Inverse Square Function:
    @staticmethod
    def isq(x):
        """Inverse Square function."""
        return tf.math.pow(x, -2.0)

    # Hard Shrink Function:
    def hardshrink(self, x):
        """Hard Shrink function."""
        return tf.where(tf.abs(x) < self.act_param, 0.0, x)

    # Soft Shrink Function:
    def softshrink(self, x):
        """Soft Shrink function."""
        return tf.where(x > self.act_param, x - self.act_param, tf.where(x < -self.act_param, x + self.act_param, 0.0))

    # Squared Rectified Linear Unit (SQReLU) Function:
    def sqrelu(self, x):
        """Squared Rectified Linear Unit (SQReLU) function."""
        return tf.where(x < 0.0, tf.math.pow(x, 2.0), tf.where(x < self.act_param, x, self.act_param + tf.math.pow(x - self.act_param, 2.0)))

    # Sine Function:
    @staticmethod
    def sine(x):
        """Sine function."""
        return tf.sinh(x)

    # Soft Exponential Function:
    def softexp(self, x):
        """Soft Exponential function."""
        return tf.where(x > 0.0, (tf.math.exp(self.act_param * x) - 1.0) / self.act_param, x)

    # ArcTan Function:
    @staticmethod
    def arctan(x):
        """ArcTan function."""
        return tf.atan(x)

    # Sinusoidal Transfer Function:
    @staticmethod
    def sin_transfer(x):
        """Sinusoidal Transfer function."""
        return tf.sinh(tf.abs(x))

    # Hard Sigmoid Function:
    @staticmethod
    def hsigmoid(x):
        """Hard Sigmoid function."""
        return tf.minimum(tf.maximum(0.0, x + 1.0), 1.0)

    # Tangent Sigmoid Function:
    @staticmethod
    def tsigmoid(x):
        """Tangent Sigmoid function."""
        return tf.tanh(tf.sigmoid(x))

    # ArcSinH Function:
    @staticmethod
    def arcsinh(x):
        """ArcSinH function."""
        return tf.math.asinh(x)

    # Logit Function:
    @staticmethod
    def logit(x):
        """Logit function."""
        return tf.math.log(x / (1.0 - x))

    # Truncated Linear Unit (TLU) Function:
    def tlu(self, x):
        """Truncated Linear Unit (TLU) function."""
        return tf.where(x > self.act_param, self.act_param, tf.where(x < 0.0, 0.0, x))

    # Asymmetric Quadratic Function:
    def aq(self, x):
        """Asymmetric Quadratic function."""
        return tf.where(x < 0.0, self.act_param * tf.math.pow(x, 2.0), self.act_param2 * tf.math.pow(x, 2.0))

    # Logarithmic Sigmoid Function:
    @staticmethod
    def logsigmoid(x):
        """Logarithmic Sigmoid function."""
        return -tf.keras.activations.softplus(-x)

    # Cosine Function
    @staticmethod
    def cosine(x):
        """Cosine function."""
        return tf.cos(x)

    # Rectified Cosine Function
    @staticmethod
    def relu_cos(x):
        """Rectified Cosine Function."""
        return tf.where(x < 0.0, 0.0, tf.where(x < np.pi / 2.0, x, 0.5 * (1.0 + tf.cos(x))))

    # Inverse Multiquadratic Function
    def imq(self, x):
        """Inverse Multiquadratic Function."""
        return 1.0 / tf.sqrt(tf.math.pow(x, 2.0) + tf.math.pow(self.act_param, 2.0))

    # Cosine Sigmoid Function
    @staticmethod
    def cos_sigmoid(x):
        """Cosine Sigmoid Function."""
        return tf.cos(tf.keras.activations.sigmoid(x))

    # Triangular Function
    @staticmethod
    def triangular(x):
        """Triangular Function."""
        return tf.maximum(0.0, 1.0 - tf.abs(x))

    # Hard-Tanh Function
    @staticmethod
    def hardtanh(x):
        """Hard-Tanh Function."""
        return tf.minimum(tf.maximum(x, -1.0), 1.0)

    # Inverse Sine Function
    @staticmethod
    def inverse_sine(x):
        """Inverse Sine Function."""
        return tf.asin(x)

    # Quadratic Bezier Function
    @staticmethod
    def bezier(x):
        """Quadratic Bezier Function."""
        return tf.where(x < 0.0, 0.0, tf.where(x > 1.0, 1.0, 1.0 - tf.square(1.0 - x)))

    # Bipolar Sigmoid Function
    @staticmethod
    def bsigmoid(x):
        """Bipolar Sigmoid Function."""
        return (tf.exp(x) - tf.exp(-x)) / (tf.exp(x) + tf.exp(-x))

    # Power Function
    @staticmethod
    def power(x, a=1.0):
        """Power Function"""
        return tf.math.pow(x, a)

    # Gaussian Swish Function
    def gswish(self, x):
        """Gaussian Swish Function."""
        return x * tf.keras.activations.sigmoid(self.act_param * x) * tf.keras.activations.softmax(x)

    # Inverse Gamma Function
    def invgamma(self, x):
        """Inverse Gamma Function."""
        return 1.0 / tf.math.pow(x, self.act_param)

    # Soft Clip Function
    def softclip(self, x):
        """Soft Clip Function."""
        return tf.where(x > self.act_param, self.act_param + (x - self.act_param) / (1.0 + tf.exp(-50.0 * (x - self.act_param))),
                        tf.where(x < -self.act_param, -self.act_param + (x + self.act_param) / (1.0 + tf.exp(50.0 * (x + self.act_param))), x))

    # Inverse Cosine Function
    @staticmethod
    def inverse_cosine(x):
        """Inverse Cosine Function."""
        return tf.acos(x)

    # Sinusoid Function
    @staticmethod
    def sinusoid(x):
        """Sinusoid Function."""
        return tf.sin(x)

    # Inverse Logit Function
    @staticmethod
    def inv_logit(x):
        """Inverse Logit Function."""
        return tf.math.log(x / (1.0 - x))

    # Soft Exponential Function
    def soft_exponential(self, x):
        """Soft Exponential Function."""
        return tf.where(x < 0.0, (tf.exp(x * self.act_param) - 1.0) / self.act_param, x)

    # Smooth Rectified Linear Unit (SReLU) Function
    def srelu(self, x):
        """Smooth Rectified Linear Unit (SReLU) Function."""
        return tf.where(x < 0.0, self.act_param * (tf.exp(x) - 1.0), tf.where(x > self.act_param2, x, x / self.act_param2 * (tf.exp(-x * self.act_param2) - 1.0) + self.act_param))

    # Inverse Tangent Function
    @staticmethod
    def inverse_tangent(x):
        """Inverse Tangent Function."""
        return tf.atan(x)

    # Hard Swish Function
    @staticmethod
    def hswish(x):
        """Hard Swish Function."""
        return x * tf.nn.relu6(x + 3.0) / 6.0

    # Asymmetric Quadratic Function
    def aqrelu(self, x):
        """Asymmetric Quadratic Function."""
        return tf.where(x < 0.0, self.act_param * (x + self.act_param2) ** 2.0, x)

    # Gaussian Error Linear Unit 2 (GELU2) Function
    @staticmethod
    def gelu2(x):
        """Gaussian Error Linear Unit 2 (GELU2) Function."""
        return x * tf.keras.backend.sigmoid(1.702 * x)

    # Sinusoidal Function 2
    @staticmethod
    def sinusoid2(x):
        """Sinusoidal Function 2."""
        return tf.sin(tf.math.sqrt(tf.abs(x)))

    # Inverse Hyperbolic Tangent Function
    @staticmethod
    def inverse_tanh(x):
        """Inverse Hyperbolic Tangent Function."""
        return tf.atanh(x)

    # Leaky Softplus Function
    def leaky_softplus(self, x):
        """Leaky Softplus Function."""
        return tf.where(x > 0.0, tf.math.log(1.0 + tf.exp(x)), self.act_param * x * tf.math.log(1.0 + tf.exp(x / self.act_param)))

    # Gaussian Tangent Function
    @staticmethod
    def gaussian_tangent(x):
        """Gaussian Tangent Function."""
        return tf.exp(-(tf.keras.backend.tanh(x) ** 2.0))

    # Exponential Cosine Function
    @staticmethod
    def exp_cosine(x):
        """Exponential Cosine Function."""
        return tf.exp(tf.cos(x)) - 2.0 * tf.cos(4.0 * x)

    # Gaussian Cumulative Distribution Function (CDF) Function
    @staticmethod
    def gaussian_cdf(x):
        """Gaussian Cumulative Distribution Function (CDF) Function."""
        return 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

    # Hard-Mish Function
    @staticmethod
    def hmish(x):
        """Hard-Mish Function."""
        return x * tf.nn.relu6(x + 3.0) / 6.0 * tf.nn.relu6(-x + 3.0) / 6.0

    # Smooth Sigmoid Function
    def smooth_sigmoid(self, x):
        """Smooth Sigmoid Function."""
        return (1.0 + tf.tanh(x * self.act_param / 2.0)) / 2.0

    # Logarithm of Exponential Function
    @staticmethod
    def log_exp(x):
        """Logarithm of Exponential Function."""
        return tf.math.log(tf.math.exp(x) + 1.0)

    # Cubic Function
    @staticmethod
    def cubic(x):
        """Cubic Function."""
        return tf.pow(x, 3.0)

    # Exponential Sine Function
    @staticmethod
    def exp_sine(x):
        """Exponential Sine Function."""
        return tf.exp(tf.sin(x)) - 1.0

    # Symmetric Sigmoid Function
    @staticmethod
    def sym_sigmoid(x):
        """Symmetric Sigmoid Function."""
        return 2.0 / (1.0 + tf.exp(-x)) - 1.0

    # Squared Function
    @staticmethod
    def square(x):
        """Squared Function."""
        return tf.pow(x, 2.0)

    # Soft Clipping Function
    def soft_clipping(self, x):
        """Soft Clipping Function."""
        return tf.where(x < -self.act_param, -1.0, tf.where(x > self.act_param, 1.0, x))

    # Swish-Gaussian Function
    @staticmethod
    def swish_gaussian(x):
        """Swish-Gaussian Function."""
        return x * tf.math.exp(-tf.pow(x, 2.0) / 2.0) / (1.0 + tf.math.exp(-x))

    # Hard Shrink Function
    def hard_shrink(self, x):
        """Hard Shrink Function."""
        return tf.where(tf.abs(x) < self.act_param, 0.0, x)

    # Smooth Hard Tanh Function
    def smooth_hard_tanh(self, x):
        """Smooth Hard Tanh Function."""
        return tf.where(x < -1.0 / self.act_param, -1.0, tf.where(x > 1.0 / self.act_param, 1.0, self.act_param * x))

    # Bipolar Sigmoid Function
    @staticmethod
    def bipolar_sigmoid(x):
        """Bipolar Sigmoid Function."""
        return (1.0 - tf.exp(-x)) / (1.0 + tf.exp(-x))

    # Logarithmic Sigmoid Function
    @staticmethod
    def log_sigmoid(x):
        """Logarithmic Sigmoid Function."""
        return tf.math.log(1.0 / (1.0 + tf.exp(-x)))

    # Hard Sigmoid Function
    @staticmethod
    def hard_sigmoid(x):
        """Hard Sigmoid Function."""
        return tf.clip_by_value((x + 1.0) / 2.0, 0.0, 1.0)

    # Inverse Square Root Function
    @staticmethod
    def invsqrt(x):
        """Inverse Square Root Function."""
        return x / tf.sqrt(1.0 + tf.square(x))

    # Gaussian Tangent Hyperbolic Function
    @staticmethod
    def gauss_tanh(x):
        """Gaussian Tangent Hyperbolic Function."""
        return tf.math.exp(-tf.square(tf.math.tanh(x)))

    # EGAULU Function
    def egaulu(self, x):
        """EGAULU Function."""
        return x * tf.where(x > 0.0, 1.0, self.act_param * tf.math.exp(x) - self.act_param)

    # Logarithm Function
    @staticmethod
    def logarithm(x):
        """Logarithm Function."""
        return tf.math.log(tf.abs(x) + 1.0)

    # Inverse Sine Function
    @staticmethod
    def inv_sine(x):
        """Inverse Sine Function."""
        return tf.asin(x)

    # Hard Tanh Function
    @staticmethod
    def hard_tanh(x):
        """Hard Tanh Function."""
        return tf.clip_by_value(x, -1.0, 1.0)

    # Bent Identity Smoothed Function
    def bent_identity_smoothed(self, x):
        """Bent Identity Smoothed Function."""
        return (tf.sqrt(tf.square(x) + 1.0) - 1.0) / 2.0 + self.act_param * (tf.exp(x) - tf.exp(-x)) / (tf.exp(x) + tf.exp(-x))

    # Positive Softplus Function
    @staticmethod
    def pos_softplus(x):
        """Positive Softplus Function."""
        return tf.math.log(1.0 + tf.exp(x))

    # Inverse Multiquadratic Function
    def inv_multiquadratic(self, x):
        """Inverse Multiquadratic Function."""
        return 1.0 / tf.sqrt(tf.square(x) + tf.square(self.act_param))

    # Inverse Cosine Function
    @staticmethod
    def inv_cosine(x):
        """Inverse Cosine Function."""
        return tf.acos(x)

    # Asymmetric Gaussian Function
    def asymmetric_gaussian(self, x):
        """Asymmetric Gaussian Function"""
        return tf.exp(-tf.pow(tf.abs(x - self.act_param2), 2.0) / (2.0 * tf.pow(self.act_param, 2.0)))

    # Inverse Quadratic Function
    def inv_quadratic(self, x):
        """Inverse Quadratic Function."""
        return 1.0 / (tf.square(x) + tf.square(self.act_param))

    # Gaussian Squared Function
    def gaussian_squared(self, x):
        """Gaussian Squared Function."""
        return tf.exp(-tf.square(x) / (2.0 * tf.square(self.act_param))) * tf.square(x)

    # Symmetric Sigmoid Function
    def symmetric_sigmoid(self, x):
        """Symmetric Sigmoid Function."""
        return tf.where(x >= 0.0, 1.0 / (1.0 + tf.exp(-self.act_param * x)), tf.exp(self.act_param * x) / (1.0 + tf.exp(self.act_param * x)))

    # Inverse Cubic Function
    def inv_cubic(self, x):
        """Inverse Cubic Function."""
        return 1.0 / (tf.pow(x, 3.0) + tf.pow(self.act_param, 3.0))

    # Cauchy Function
    def cauchy(self, x):
        """Cauchy Function."""
        return (self.act_param ** 2.0) / (tf.square(x) + self.act_param ** 2.0)

    # Exponential Quadratic Function
    def exponential_quadratic(self, x):
        """Exponential Quadratic Function."""
        return tf.math.log(tf.exp(x) + tf.exp(-self.act_param * x)) / self.act_param

    # Rational Quadratic Function
    def rational_quadratic(self, x):
        """Rational Quadratic Function."""
        return 1.0 - tf.pow(1.0 + tf.square(x) / (2.0 * self.act_param), -self.act_param)

    # Cubic Spline Function
    def cubic_spline(self, x):
        """Cubic Spline Function."""
        knots = self.knots
        knots = tf.convert_to_tensor(knots, dtype=tf.float32)
        n = knots.shape[0]

        def lagrange_interpolation(i, x):
            result = 1.0
            for j in range(n):
                if i != j:
                    result *= (x - knots[j]) / (knots[i] - knots[j])
            return result

        output = tf.zeros_like(x, dtype=tf.float32)
        for i in range(n):
            output += lagrange_interpolation(i, x)

        return output

    # Symmetric Soft Clipping Function
    def symmetric_soft_clipping(self, x):
        """Symmetric Soft Clipping Function."""
        return x / (1 + tf.exp(-self.act_param * (x - self.act_param2))) + tf.math.log(1 + tf.exp(self.act_param * self.act_param2)) / self.act_param

    # Binary Step Function
    def binary_step(self, x):
        """Binary Step Function."""
        return tf.where(x < self.act_param, tf.zeros_like(x), tf.ones_like(x))

    # Inverse Multiquadratic Radial Basis Function (IMRBF) Function
    def imrbf(self, x):
        """Inverse Multiquadratic Radial Basis Function (IMRBF) Function."""
        return tf.sqrt(tf.pow(x, 2.0) + tf.pow(self.act_param, 2.0))

    # Complementary Log-Log (cLogLog) Function
    @staticmethod
    def cloglog(x):
        """Complementary Log-Log (cLogLog) Function."""
        return tf.math.log(-tf.math.log(1.0 - x))

    # Noisy Rectified Linear Unit (NReLU) Function
    def nrelu(self, x):
        """Noisy Rectified Linear Unit (NReLU) Function."""
        noise = tf.random.normal(tf.shape(x), mean=1.0, stddev=self.act_param, dtype=x.dtype)
        return tf.where(x >= 0.0, x, noise * x)

