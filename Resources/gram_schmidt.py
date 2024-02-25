import tensorflow as tf


def gram_schmidt(vectors, num_vectors=None):
    """
        Perform Gram-Schmidt orthogonalization on a set of vectors.

        Gram-Schmidt orthogonalization is a method to orthogonalize a set of vectors
        in an inner product space, making them pairwise orthogonal.

        Parameters:
        -----------
        - vectors (tf.Tensor): Input vectors to be orthogonalized. Shape: (..., d, n), where d is the dimensionality of the vectors
          and n is the number of vectors.
        - num_vectors (int or None): Number of vectors to orthogonalize. If None, all vectors will be orthogonalized. Default is None.

        Returns:
        -----------
        tf.Tensor: Orthogonalized vectors.

        Notes:
        -----------
        - The input tensors are assumed to have their vectors as the last dimension.
    """
    with tf.name_scope('gram_schmidt'):
        n = tf.shape(vectors)[-1]
        if num_vectors is None:
            num_vectors = n
        cond = lambda vecs, i: i < num_vectors - 1

        def body_fn(vecs, i):
            # Slice out the vector w.r.t. which we're orthogonalizing the rest.
            u = tf.math.l2_normalize(vecs[..., i, tf.newaxis], axis=-2)
            # Find weights by dotting the d x 1 against the d x n.
            weights = tf.einsum('...dm,...dn->...n', u, vecs)
            # Project out vector `u` from the trailing vectors.
            masked_weights = tf.where(
                tf.range(n) > i, weights, 0.)[..., tf.newaxis, :]
            vecs = vecs - tf.math.multiply_no_nan(u, masked_weights)
            vecs = tf.reshape(vecs, vectors.shape)
            return vecs, i + 1

        vectors, _ = tf.while_loop(cond, body_fn, (vectors, tf.zeros([], tf.int32)))
        vec_norm = tf.linalg.norm(vectors, ord=2, axis=-2, keepdims=True)
        return tf.math.divide_no_nan(vectors, vec_norm)
