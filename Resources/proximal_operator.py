import tensorflow as tf


def proximal_operator(x, alpha):
    """Soft thresholding proximal operator."""
    return tf.math.sign(x) * tf.maximum(tf.abs(x) - alpha, 0.0)