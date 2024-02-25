import tensorflow as tf


def generate_contrainted_weights(x, y, number_neurons, norm_threshold=0.1, parallel_threshold=0.1):
    """
        Generates constrained weights for the hidden layer of the Extreme Learning Machine (ELM) based on class differences in the input data.

        Args:
        -----------
            x (tf.Tensor): The input data of shape (batch_size, input_size).
            y (tf.Tensor): The target output data of shape (batch_size, num_classes).
            number_neurons (int): The numbwe of neurons in hidden layer.
            norm_threshold (float, optional): The threshold for the norm of the difference vector between classes. Defaults to 0.1.
            parallel_threshold (float, optional): The threshold for the cosine of the angle between the difference vector and existing vectors. Defaults to 0.1.

        Returns:
        -----------
            None

        Generates normalized difference vectors between random samples of two different classes from the input data and stores them as hidden layer weights (alpha) along with biases (bias).

        The function first identifies unique class labels from the target output data (y) and then iteratively selects pairs of samples from different classes. For each pair, it calculates the difference vector and checks if it meets the specified criteria:

        1. The norm of the difference vector must be greater than the norm_threshold.
        2. The angle between the difference vector and existing vectors must be greater than the parallel_threshold.

        If a valid difference vector is found, it is normalized and stored in the alpha matrix, and the corresponding bias value is calculated and stored.

        The process continues until the desired number of hidden neurons is reached.
    """

    normalized_vectors = []
    normalized_bs = []
    chosen_difference_vectors = []

    unique_labels = tf.unique(tf.argmax(y, axis=1)).y

    def is_nearly_parallel(new_vector, existing_vectors, angle_threshold):
        for existing_vector in existing_vectors:
            cos_angle = tf.tensordot(existing_vector, new_vector, axes=(-1, -1)) / (
                    tf.norm(existing_vector) * tf.norm(new_vector)
            )
            if cos_angle > 1 - angle_threshold:
                return True
        return False

    while len(chosen_difference_vectors) < number_neurons:
        # Randomly draw two different classes
        class_labels = tf.random.shuffle(unique_labels)[:2]
        indices_c1 = tf.where(tf.equal(tf.argmax(y, axis=1), class_labels[0]))[:, 0]
        indices_c2 = tf.where(tf.equal(tf.argmax(y, axis=1), class_labels[1]))[:, 0]
        sample_index_c1 = tf.random.shuffle(indices_c1)[0]
        sample_index_c2 = tf.random.shuffle(indices_c2)[0]

        x_c1 = tf.gather(x, sample_index_c1)
        x_c2 = tf.gather(x, sample_index_c2)

        # Generate difference vector
        diff_vector = x_c2 - x_c1

        # Check conditions for keeping the difference vector
        if tf.norm(diff_vector) < norm_threshold or \
                is_nearly_parallel(diff_vector, chosen_difference_vectors, parallel_threshold):
            continue

        normalized_vector = diff_vector / tf.linalg.norm(diff_vector)
        normalized_b = tf.reduce_sum((x_c2 + x_c1) * (x_c2 - x_c1)) / tf.linalg.norm(diff_vector)

        # Store the chosen difference vector
        normalized_vectors.append(normalized_vector)
        normalized_bs.append(normalized_b)
        chosen_difference_vectors.append(diff_vector)

    normalized_vectors_matrix = tf.stack(normalized_vectors, axis=1)
    normalized_bs_vector = tf.stack(normalized_bs, axis=0)
    bias = normalized_bs_vector
    alpha = normalized_vectors_matrix
    return bias, alpha
