import tensorflow as tf


def fast_pinv(A, mode='gauss'):
    if mode == 'gauss_right':
        AAT = tf.matmul(A, tf.transpose(A))
        AA_inv = tf.linalg.inv(AAT)
        pinvA = tf.matmul(tf.transpose(A), tf.matmul(AA_inv, tf.eye(tf.shape(A)[0])))
    elif mode == 'gauss':
        At = tf.transpose(A)
        At_A = tf.matmul(At, A)
        At_A_inv = tf.linalg.inv(At_A)
        iAt = tf.matmul(tf.eye(tf.shape(A)[1]), At)
        pinvA = tf.matmul(At_A_inv, iAt)
    elif mode == 'regular':
        s, u, v = tf.linalg.svd(A, full_matrices=False)
        s_inv = tf.linalg.diag(1.0 / s)
        pinvA = tf.matmul(tf.matmul(v, s_inv), tf.transpose(u))
    elif mode == 'svd':
        s, u, v = tf.linalg.svd(A, full_matrices=False)
        eps = 1e-15
        s_inv = tf.where(tf.abs(s) > eps, 1.0 / s, tf.zeros_like(s))
        pinvA = tf.matmul(tf.matmul(v, tf.linalg.diag(s_inv)), tf.transpose(u))
    elif mode == 'qr':
        q, r = tf.linalg.qr(A, full_matrices=False, name="qr_with_pivoting")
        diag_r = tf.linalg.diag_part(r)
        eps = 1e-15
        diag_r_inv = tf.where(tf.abs(diag_r) > eps, 1.0 / diag_r, tf.zeros_like(diag_r))
        pinvA = tf.matmul(tf.transpose(r), tf.matmul(tf.linalg.diag(diag_r_inv), tf.transpose(q)))
    return pinvA
