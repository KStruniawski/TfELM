import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from Layers.ELMLayer import ELMLayer
from Layers.WELMLayer import WELMLayer
from Layers.KELMLayer import KELMLayer
from Models.ELMModel import ELMModel
from Resources.generate_random_filters import generate_random_filters
from Resources.sqrt_pooling import sqrt_pooling


class LRFELMModel(BaseEstimator, ClassifierMixin):
    """
        A Local Receptive Field Extreme Learning Machine (LRF ELM) model.

        This model combines random CNN filters with an Extreme Learning Machine (ELM) for classification tasks.

        Parameters:
        -----------
        - elm_model (ELMModel): The ELM model to be used for training and prediction.
        - num_feature_maps (int): Number of feature maps in the random CNN filters.
        - filter_size (int): Size of the filters used in the random CNN.
        - num_input_channels (int): Number of input channels (e.g., 1 for grayscale images, 3 for RGB images).
        - pool_size (int): Size of the pooling window for max pooling.
        - classification (bool): Whether the task is classification (default is True).
        - random_weights (bool): Whether to use random weights for the CNN filters (default is True).
        - **args: Additional keyword arguments.

        Attributes:
        -----------
        - classes_ (array): The unique classes present in the training data.
        - kernels (tensor): The random CNN filters.

        Examples:
        -----------
        Split the dataset into training and testing sets

        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        Convert input data to the appropriate shape for CNN (assuming MNIST dataset)

        >>> X_train = np.reshape(X_train, (48000, 28, 28, 1))
        >>> X_test = np.reshape(X_test, (12000, 28, 28, 1))

        Initialize the ELMLayer with specified number of neurons and number of classes

        >>> layer = ELMLayer(number_neurons=5000, C=10)

        Initialize the ELMModel with the ELMLayer

        >>> elm_model = ELMModel(layer)

        Initialize the LRFELMModel with the ELMModel

        >>> model = LRFELMModel(elm_model=elm_model)

        Fit the LRFELMModel to the training data

        >>> model.fit(X_train, y_train)

        Predict the labels for the testing data

        >>> pred = model.predict(X_test)

        Print the accuracy score of the model

        >>> print(accuracy_score(pred, y_test))

        Fit the ELM model to the entire dataset

        >>> X = np.reshape(X, (60000, 28, 28, 1))
        >>> model.fit(X, y)

        Save the trained model to a file

        >>> model.save("Saved Models/LRFELM_Model.h5")

        Load the saved model from the file

        >>> model = model.load("Saved Models/LRFELM_Model.h5")

        Evaluate the accuracy of the model on the training data

        >>> acc = accuracy_score(model.predict(X), y)
    """
    def __init__(self, elm_model, num_feature_maps=48, filter_size=4, num_input_channels=1, pool_size=3,
                 classification=True, random_weights=True, **args):
        self.classes_ = None
        self.classification = classification
        self.num_feature_maps = num_feature_maps
        self.filter_size = filter_size
        self.num_input_channels = num_input_channels
        self.pool_size = pool_size
        self.elm_model = elm_model
        self.random_weights = random_weights
        if "kernels" in args:
            self.kernels = args["kernels"]
        else:
            self.kernels = None

    def fit(self, X, y):
        """
            Fit the LRF ELM model to the training data.

            Parameters:
            -----------
            - X (array-like): The input data.
            - y (array-like): The target labels.

            Returns:
            -----------
            None

            Examples:
            -----------
            Split the dataset into training and testing sets

            >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            Convert input data to the appropriate shape for CNN (assuming MNIST dataset)

            >>> X_train = np.reshape(X_train, (48000, 28, 28, 1))
            >>> X_test = np.reshape(X_test, (12000, 28, 28, 1))

            Initialize the ELMLayer with specified number of neurons and number of classes

            >>> layer = ELMLayer(number_neurons=5000, C=10)

            Initialize the ELMModel with the ELMLayer

            >>> elm_model = ELMModel(layer)

            Initialize the LRFELMModel with the ELMModel

            >>> model = LRFELMModel(elm_model=elm_model)

            Fit the LRFELMModel to the training data

            >>> model.fit(X_train, y_train)
        """
        if self.classification:
            self.classes_ = unique_labels(y)
        else:
            self.classes_ = [0]

        if len(np.shape(y)) == 1:
            y = to_categorical(y)

        N = X.shape[0]
        X = tf.cast(X, dtype=tf.float32)
        self.kernels = generate_random_filters(self.filter_size, self.num_feature_maps, self.num_input_channels)
        conv_output = tf.nn.conv2d(X, self.kernels, strides=[1, 1, 1, 1], padding='VALID')
        pooled_output = sqrt_pooling(conv_output, self.pool_size)
        flattened_output = tf.reshape(pooled_output, [N, -1])
        self.elm_model.fit(flattened_output, y)

    def predict(self, X):
        """
            Predict the labels for the input data.

            Parameters:
            -----------
            - X (array-like): The input data.

            Returns:
            -----------
            array: Predicted labels.

            Examples:
            -----------
            Split the dataset into training and testing sets

            >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            Convert input data to the appropriate shape for CNN (assuming MNIST dataset)

            >>> X_train = np.reshape(X_train, (48000, 28, 28, 1))
            >>> X_test = np.reshape(X_test, (12000, 28, 28, 1))

            Initialize the ELMLayer with specified number of neurons and number of classes

            >>> layer = ELMLayer(number_neurons=5000, C=10)

            Initialize the ELMModel with the ELMLayer

            >>> elm_model = ELMModel(layer)

            Initialize the LRFELMModel with the ELMModel

            >>> model = LRFELMModel(elm_model=elm_model)

            Fit the LRFELMModel to the training data

            >>> model.fit(X_train, y_train)

            Predict the labels for the testing data

            >>> pred = model.predict(X_test)
        """
        N = X.shape[0]
        X = tf.cast(X, dtype=tf.float32)
        conv_output = tf.nn.conv2d(X, self.kernels, strides=[1, 1, 1, 1], padding='VALID')
        pooled_output = sqrt_pooling(conv_output, self.pool_size)
        flattened_output = tf.reshape(pooled_output, [N, -1])
        return self.elm_model.predict(flattened_output)

    def predict_proba(self, X):
        """
            Predict class probabilities for the input data.

            Parameters:
            -----------
            - X (array-like): The input data.

            Returns:
            -----------
            array: Predicted class probabilities.

            Examples:
            -----------
            Split the dataset into training and testing sets

            >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            Convert input data to the appropriate shape for CNN (assuming MNIST dataset)

            >>> X_train = np.reshape(X_train, (48000, 28, 28, 1))
            >>> X_test = np.reshape(X_test, (12000, 28, 28, 1))

            Initialize the ELMLayer with specified number of neurons and number of classes

            >>> layer = ELMLayer(number_neurons=5000, C=10)

            Initialize the ELMModel with the ELMLayer

            >>> elm_model = ELMModel(layer)

            Initialize the LRFELMModel with the ELMModel

            >>> model = LRFELMModel(elm_model=elm_model)

            Fit the LRFELMModel to the training data

            >>> model.fit(X_train, y_train)

            Predict the prediction probability for the testing data

            >>> pred_proba = model.predict_proba(X_test)
        """
        N = X.shape[0]
        X = tf.cast(X, dtype=tf.float32)
        conv_output = tf.nn.conv2d(X, self.kernels, strides=[1, 1, 1, 1], padding='VALID')
        pooled_output = sqrt_pooling(conv_output, self.pool_size)
        flattened_output = tf.reshape(pooled_output, [N, -1])
        return self.elm_model.predict_proba(flattened_output)

    def save(self, file_path):
        """
        Serialize the current instance and save it to a HDF5 file.

        Parameters:
        -----------
        - path (str): The file path where the serialized instance will be saved.

        Returns:
        -----------
        None

        Examples:
        -----------
        Save the trained model to a file

        >>> model.save("Saved Models/LRFELM_Model.h5")
        """
        try:
            with h5py.File(file_path, 'w') as h5file:
                for key, value in self.to_dict().items():
                    if value is None:
                        value = 'None'
                    h5file.create_dataset(key, data=value)
                h5file.close()
        except Exception as e:
            print(f"Error saving to HDF5: {e}")

    @classmethod
    def load(cls, file_path: str):
        """
        Deserialize an instance from a file.

        Parameters:
        -----------
        - file_path (str): The file path from which to load the serialized instance.

        Returns:
        -----------
        LRFELMModel: An instance of the LRFELMModel class loaded from the file.

        Examples:
        -----------
        Load the saved model from the file

        >>> model = model.load("Saved Models/LRFELM_Model.h5")
        """
        try:
            with h5py.File(file_path, 'r') as h5file:
                # Extract attributes from the HDF5 file
                attributes = {key: h5file[key][()] for key in h5file.keys()}

                for key, value in attributes.items():
                    if type(value) is bytes:
                        v = value.decode('utf-8')
                        attributes[key] = v

                if "classification" in attributes:
                    classification = attributes.pop("classification")
                if "model_name" in attributes:
                    m_name = attributes.pop("model_name")
                if "random_weights" in attributes:
                    random_weights = attributes.pop("random_weights")
                if "kernels" in attributes:
                    kernels = attributes.pop("kernels")
                if "num_feature_maps" in attributes:
                    num_feature_maps = attributes.pop("num_feature_maps")
                if "filter_size" in attributes:
                    filter_size = attributes.pop("filter_size")
                if "num_input_channels" in attributes:
                    num_input_channels = attributes.pop("num_input_channels")
                if "pool_size" in attributes:
                    pool_size = attributes.pop("pool_size")
                if "classes_" in attributes:
                    c = attributes.pop("classes_")

                model = eval(f"{m_name}.load(file_path)")
                m = cls(elm_model=model, num_feature_maps=num_feature_maps, filter_size=filter_size,
                        num_input_channels=num_input_channels, pool_size=pool_size, classification=classification,
                        random_weights=random_weights, kernels=kernels)
                m.classes_ = c
                return m
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None  # Return None or raise an exception based on your error-handling strategy

    def to_dict(self):
        """
            Serialize the current instance into a dictionary.

            Returns:
            -----------
            dict: A dictionary containing serialized attributes of the instance.
        """
        attributes = self.elm_model.to_dict()
        attributes["model_name"] = self.elm_model.__class__.__name__
        attributes["classification"] = self.classification
        attributes["random_weights"] = self.random_weights
        attributes["kernels"] = self.kernels
        attributes["num_feature_maps"] = self.num_feature_maps
        attributes["filter_size"] = self.filter_size
        attributes["num_input_channels"] = self.num_input_channels
        attributes["pool_size"] = self.pool_size
        attributes["classes_"] = self.classes_

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes