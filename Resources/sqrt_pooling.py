import tensorflow as tf


def sqrt_pooling(input_tensor, pool_size):
    """
        Perform square root pooling on the input tensor.

        Square root pooling computes the average of values within a pooling window, squares the result, and then takes the
        square root of the pooled values.

        Parameters:
        -----------
            input_tensor (tf.Tensor): The input tensor.
            pool_size (int): The size of the pooling window.

        Returns:
        -----------
            tf.Tensor: The pooled tensor after square root pooling.
    """
    pooled = tf.nn.pool(input_tensor, window_shape=[pool_size, pool_size], pooling_type='AVG',
                        strides=[pool_size, pool_size], padding='SAME')
    pooled = (pooled * pool_size ** 2) ** 2
    pooled = tf.sqrt(pooled)
    return pooled