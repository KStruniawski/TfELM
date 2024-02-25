import tensorflow as tf


def apply_denoising(x, denoising, denoising_param):
    """
        Apply denoising to the input tensor.

        Parameters:
        -----------
        - x (tf.Tensor): Input tensor.
        - denoising (str): Denoising method. Choices: 'gs' for Gaussian noise, 'mn' for masking noise, 'sp' for salt-and-pepper noise.
        - denoising_param (float): Denoising parameter.

        Returns:
        -----------
        tf.Tensor: Denoised tensor.

        Raises:
        -----------
        Exception: If an incorrect denoising method is provided.
    """
    if denoising == 'gs':
        gs = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=denoising_param, dtype=tf.float32)
        return x + gs
    elif denoising == 'mn':
        mask = tf.math.less(tf.random.uniform(tf.shape(x)), denoising_param)
        return tf.where(mask, tf.zeros_like(x), x)
    elif denoising == 'sp':
        min_mask = tf.math.less(tf.random.uniform(tf.shape(x)), denoising_param/2)
        max_mask = tf.math.less(tf.random.uniform(tf.shape(x)), denoising_param/2)
        noisy_matrix = tf.where(min_mask, tf.reduce_min(x), x)
        noisy_matrix = tf.where(max_mask, tf.reduce_max(x), noisy_matrix)
        return noisy_matrix
    else:
        raise Exception("Incorrect denoise method!")