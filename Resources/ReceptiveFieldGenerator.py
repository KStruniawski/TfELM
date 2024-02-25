import tensorflow as tf


class ReceptiveFieldGenerator:
    """
        Receptive Field Generator.

        This class implements a generator for receptive fields used in the Deep Extreme Learning Machine (Deep ELM) model.
        Receptive fields are rectangular regions applied to feature maps in convolutional layers.

        Parameters:
        -----------
            input_size (tuple): The size of the input images (height, width, channels).
            exclusion_distance (int, optional): The minimum distance from the image border to exclude when generating
                receptive fields. Defaults to 3.
            q_threshold (int, optional): The minimum area of the generated receptive fields. Defaults to 10.
            num_classes (int, optional): Number of classes. Defaults to None.

        Attributes:
        -----------
            L (int): Number of neurons.
            input_size (tuple): The size of the input images (height, width, channels).
            exclusion_distance (int): The minimum distance from the image border to exclude when generating
                receptive fields.
            q_threshold (int): The minimum area of the generated receptive fields.
            num_classes (int): Number of classes.

        Methods:
        -----------
            to_dict(): Convert the generator to a dictionary of attributes.
            load(attributes): Load a generator instance from a dictionary of attributes.
            generate_receptive_fields(W): Generate receptive fields from weight matrices.
            __generate_double_pairs(): Generate pairs of coordinates for receptive field corners.
            _apply_rectangle_mask(image, top_left, bottom_right): Apply a rectangle mask to an image.
            __reshape_image(image_flat): Reshape a flattened image to its original shape.
    """
    def __init__(self, input_size, exclusion_distance=3, q_threshold=10, num_classes=None):
        self.L = None
        self.input_size = input_size
        self.exclusion_distance = exclusion_distance
        self.q_threshold = q_threshold
        self.num_classes = num_classes

    def to_dict(self):
        """
            Convert the generator to a dictionary of attributes.

            Returns:
            -----------
                dict: A dictionary containing the attributes of the generator.
        """
        attributes = {
            'rf_name': 'ReceptiveFieldGenerator',
            'input_size': self.input_size,
            'exclusion_distance': self.exclusion_distance,
            'q_threshold': self.q_threshold,
            'num_classes': self.num_classes
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
                ReceptiveFieldGenerator: An instance of the ReceptiveFieldGenerator class loaded from the attributes.
        """
        input_size = attributes.pop('input_size')
        exclusion_distance = attributes.pop('exclusion_distance')
        q_threshold = attributes.pop('q_threshold')
        num_classes = attributes.pop('num_classes')
        return cls(input_size, exclusion_distance, q_threshold, num_classes)

    def generate_receptive_fields(self, W):
        """
            Generate receptive fields from weight matrices.

            Parameters:
            -----------
                W (Tensor): The weight matrix.

            Returns:
            -----------
                Tensor: The processed weight matrix with receptive fields.
        """
        self.L = tf.shape(W)[-1]
        top_left_coords, bottom_right_coords = self.__generate_double_pairs()
        W = tf.transpose(W)
        reshaped_images = tf.map_fn(self.__reshape_image, W, dtype=tf.float32)
        masked_images = [self._apply_rectangle_mask(reshaped_images[i], top_left_coords[i], bottom_right_coords[i])
                         for i in range(W.shape[0])]
        flattened_images = [tf.reshape(masked_image, [-1]) for masked_image in masked_images]
        W_processed = tf.stack(flattened_images, axis=0)
        W_processed = tf.nn.l2_normalize(W_processed, axis=1)

        if self.num_classes is not None:
            mask = tf.ones_like(W_processed[:, :self.num_classes])
            W_processed = tf.concat([mask, W_processed[:, self.num_classes:]], axis=1)
        return tf.transpose(W_processed)

    def __reshape_image(self, image_flat):
        """
            Reshape a flattened image to its original shape.

            Parameters:
            -----------
                image_flat (Tensor): The flattened image.

            Returns:
            -----------
                Tensor: The reshaped image.
        """
        return tf.reshape(image_flat, self.input_size)

    def _apply_rectangle_mask(self, image, top_left, bottom_right):
        """
           Apply a rectangle mask to an image.

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

        # Create binary mask
        mask = tf.where(
            tf.logical_and(
                tf.logical_and(tf.math.greater_equal(indices_x, top_left[0]),
                               tf.math.less(indices_x, bottom_right[0])),
                tf.logical_and(tf.math.greater_equal(indices_y, top_left[1]),
                               tf.math.less(indices_y, bottom_right[1]))
            ),
            1.0,
            0.0
        )

        # Expand the mask to cover all features (last dimension of the image)
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.tile(mask, [1] * (len(self.input_size) - 1) + [image.shape[-1]])

        # Apply the mask to the image
        masked_image = image * mask

        return masked_image

    def __generate_double_pairs(self):
        """
            Generate pairs of coordinates for receptive field corners.

            Returns:
            -----------
                Tuple[Tensor, Tensor]: Top-left and bottom-right corner coordinates.
        """
        x_coords = tf.random.uniform((self.L, 2), minval=self.exclusion_distance,
                                            maxval=self.input_size[0] - self.exclusion_distance, dtype=tf.int32)

        y_coords = tf.random.uniform((self.L, 2), minval=self.exclusion_distance,
                                                maxval=self.input_size[1] - self.exclusion_distance, dtype=tf.int32)

        # Ensure that the selected coordinates form a square with an area greater than q_threshold
        while tf.reduce_any(tf.less(tf.reduce_max(x_coords[:, 0] - y_coords[:, 0]), self.q_threshold)) or \
                tf.reduce_any(tf.less(tf.reduce_max(x_coords[:, 1] - y_coords[:, 1]), self.q_threshold)) or \
                tf.reduce_any(tf.less(tf.reduce_min(x_coords[:, 0]), self.exclusion_distance)) or \
                tf.reduce_any(tf.less(tf.reduce_min(x_coords[:, 1]), self.exclusion_distance)) or \
                tf.reduce_any(tf.less(tf.reduce_min(self.input_size[0]) - y_coords[:, 0], self.exclusion_distance)) or \
                tf.reduce_any(tf.less(tf.reduce_min(self.input_size[1]) - y_coords[:, 1], self.exclusion_distance)):
            x_coords = tf.random.uniform((self.L, 2), minval=self.exclusion_distance,
                                         maxval=self.input_size[0] - self.exclusion_distance, dtype=tf.int32)
            y_coords = tf.random.uniform((self.L, 2), minval=self.exclusion_distance,
                                         maxval=self.input_size[1] - self.exclusion_distance, dtype=tf.int32)
        top_left_coords = tf.concat([x_coords[:, :1], y_coords[:, :1]], axis=1)
        bottom_right_coords = tf.concat([x_coords[:, 1:], y_coords[:, 1:]], axis=1)
        return top_left_coords, bottom_right_coords
