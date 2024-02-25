import tensorflow as tf

from Resources.ReceptiveFieldGenerator import ReceptiveFieldGenerator


class ReceptiveFieldGaussianGenerator(ReceptiveFieldGenerator):
    """
        Gaussian Receptive Field Generator.

        This class extends the ReceptiveFieldGenerator class to generate Gaussian receptive fields.

        Parameters:
        -----------
            input_size (tuple): The size of the input images (height, width, channels).
            exclusion_distance (int, optional): The minimum distance from the image border to exclude when generating
                receptive fields. Defaults to 3.
            q_threshold (int, optional): The minimum area of the generated receptive fields. Defaults to 10.
            num_classes (int, optional): Number of classes. Defaults to None.
            sigma (float, optional): Standard deviation of the Gaussian distribution. Defaults to None.

        Attributes:
        -----------
            sigma (float): Standard deviation of the Gaussian distribution.

        Methods:
        -----------
            to_dict(): Convert the generator to a dictionary of attributes.
            load(attributes): Load a generator instance from a dictionary of attributes.
            _apply_rectangle_mask(image, top_left, bottom_right): Apply a Gaussian mask to an image.

        Examples:
        -----------
        Initialization of Receptive Field Generator

        >>> rf = ReceptiveFieldGaussianGenerator(input_size=(28, 28, 1))

        Initialize an Extreme Learning Machine layer with receptive field (RF-ELM)

        >>> elm = ELMLayer(number_neurons=num_neurons, activation='mish', receptive_field_generator=rf)
        >>> model = ELMModel(elm)

        Define a cross-validation strategy

        >>> cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

        Perform cross-validation to evaluate the model performance

        >>> scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

        Print the mean accuracy score obtained from cross-validation

        >>> print(np.mean(scores))

        Fit the ELM model to the entire dataset

        >>> model.fit(X, y)
    """
    def __init__(self, input_size, exclusion_distance=3, q_threshold=10, num_classes=None, sigma=None):
        super().__init__(input_size, exclusion_distance, q_threshold, num_classes)
        self.sigma = sigma

    def to_dict(self):
        """
            Convert the generator to a dictionary of attributes.

            Returns:
            -----------
                dict: A dictionary containing the attributes of the generator.
        """
        attributes = {
            'rf_name': 'ReceptiveFieldGaussianGenerator',
            'input_size': self.input_size,
            'exclusion_distance': self.exclusion_distance,
            'q_threshold': self.q_threshold,
            'num_classes': self.num_classes,
            'sigma': self.sigma
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
            Load a generator instance from a dictionary of attributes.

            Parameters:
            -----------
                attributes (dict): A dictionary containing the attributes of the generator.

            Returns:
            -----------
                ReceptiveFieldGaussianGenerator: An instance of the ReceptiveFieldGaussianGenerator class
                    loaded from the attributes.
        """
        input_size = attributes.pop('input_size')
        exclusion_distance = attributes.pop('exclusion_distance')
        q_threshold = attributes.pop('q_threshold')
        num_classes = attributes.pop('num_classes')
        sigma = attributes.pop('sigma')
        return cls(input_size, exclusion_distance, q_threshold, num_classes, sigma)

    def _apply_rectangle_mask(self, image, top_left, bottom_right):
        """
            Apply a Gaussian mask to an image.

            Parameters:
            -----------
                image (Tensor): The input image.
                top_left (Tensor): Coordinates of the top-left corner of the rectangle.
                bottom_right (Tensor): Coordinates of the bottom-right corner of the rectangle.

            Returns:
            -----------
                Tensor: The masked image.
        """
        # Create grid of indices
        indices_x, indices_y = tf.meshgrid(tf.range(self.input_size[0]), tf.range(self.input_size[1]), indexing='ij')

        # Calculate center of the rectangle
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2

        # Calculate distance from each pixel to the center of the rectangle
        distance = tf.sqrt(tf.cast(tf.square(indices_x - center_x) + tf.square(indices_y - center_y), dtype=tf.float32))

        # Calculate values correlated to normal distribution
        if self.sigma is None:
            sigma = tf.reduce_max(
                tf.cast(tf.stack([bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]]), tf.float32)) / 3.0
        else:
            sigma = self.sigma
        mask_values = tf.exp(-tf.square(distance) / (2 * tf.square(sigma)))

        # Expand the mask to cover all features (last dimension of the image)
        mask_values = tf.expand_dims(mask_values, axis=-1)
        mask_values = tf.tile(mask_values, [1] * (len(self.input_size) - 1) + [image.shape[-1]])

        # Apply the mask to the image
        masked_image = image * mask_values

        return masked_image