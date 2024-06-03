import h5py
import numpy as np
from sklearn.utils.multiclass import unique_labels

from Layers.USKELMLayer import USKELMLayer


class USKELMModel:
    """
        Implementation of the Unsupervised Kernel Extreme Learning Machine (USKELM) model.

        This class represents an Unsupervised Kernel Extreme Learning Machine (USKELM) model. It encapsulates the USKELMLayer
        and provides methods for fitting the model to data, making predictions, saving and loading the model, and
        converting the model to a dictionary.

        Parameters:
        -----------
        - layer (USKELMLayer): The USKELMLayer instance representing the underlying USKELM model.
        - random_weights (bool): Whether to initialize the model with random weights. Defaults to True.

        Attributes:
        -----------
        - classes_: The classes array, initialized to None.
        - layer (USKELMLayer): The USKELMLayer instance representing the model's layer.
        - activation: The activation function used in the layer.
        - act_params: Additional parameters for the activation function.
        - C: Regularization parameter C.
        - is_orthogonalized: Whether the layer is orthogonalized.
        - layer: The USELMLayer instance representing the underlying USKELM model.
        - random_weights: Whether the model is initialized with random weights.

        Methods:
        -----------
        - fit(x): Fit the USKELM model to the input data.
        - predict(X, clustering=False, k=None): Make predictions using the trained model.
        - save(file_path): Save the model to an HDF5 file.
        - load(file_path): Load the model from an HDF5 file.
        - to_dict(): Convert the model to a dictionary representation.

        Examples:
        -----------
        Embedding into 3 dimensional space
        ----
        >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
        >>> layer = USKELMLayer(kernel=kernel, embedding_size=3, lam=0.001)
        >>> model = USKELMModel(layer=layer)
        >>> model.fit(X)

        >>> model.save("Saved Models/USKELM_Model_1.h5")
        >>> model = model.load("Saved Models/USKELM_Model_1.h5")

        >>> pred = model.predict(X)

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection='3d')

        >>> ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=y.flatten(), cmap=plt.cm.Paired, marker='o', edgecolors='k')

        Add labels and title

        >>> ax.set_xlabel('Feature 1')
        >>> ax.set_ylabel('Feature 2')
        >>> ax.set_zlabel('Feature 3')
        >>> ax.set_title('3D Scatter Plot with Colored Labels')

        Show the plot

        >>> plt.show()

        Clustering
        ----

        >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
        >>> layer = USKELMLayer(kernel=kernel, embedding_size=3, lam=0.001)
        >>> model = USKELMModel(layer)
        >>> model.fit(X)
        >>> pred, cluster_labels = model.predict(X, clustering=True, k=10)

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection='3d')

        >>> ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=cluster_labels, cmap=plt.cm.Paired, marker='o', edgecolors='k')

        Add labels and title

        >>> ax.set_xlabel('Feature 1')
        >>> ax.set_ylabel('Feature 2')
        >>> ax.set_zlabel('Feature 3')
        >>> ax.set_title('3D Scatter Plot with Colored Labels')

        Show the plot

        >>> plt.show()
    """
    def __init__(self, layer: USKELMLayer, classification=True, random_weights=True):
        self.classes_ = None
        self.activation = layer.activation
        self.act_params = layer.act_params
        self.C = layer.C
        self.classification = classification
        self.layer = layer
        self.random_weights = random_weights

    def fit(self, x):
        """
            Fit the model to the input data.

            Parameters:
            -----------
                x (array-like): Input data.

            Returns:
            -----------
                None

            Examples:
            -----------
            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
            >>> layer = USKELMLayer(kernel=kernel, embedding_size=3, lam=0.001)
            >>> model = USKELMModel(layer=layer)
            >>> model.fit(X)
        """
        if self.random_weights:
            self.layer.build(np.shape(x))
        self.classes_ = unique_labels(np.zeros(np.shape(x)[0]))
        self.layer.fit(x)

    def predict(self, X, clustering=False, k=None):
        """
            Predict labels for input data.

            Parameters:
            -----------
                X (array-like): Input data.
                clustering (bool): Whether to perform clustering.
                k (int): Number of clusters if clustering is True.

            Returns:
            -----------
                array: Predicted labels.

            Examples:
            -----------
            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
            >>> layer = USKELMLayer(kernel=kernel, embedding_size=3, lam=0.001)
            >>> model = USKELMModel(layer=layer)
            >>> model.fit(X)
            >>> pred = model.predict(X)
        """
        pred = self.layer.predict(X, clustering, k)
        return pred.numpy()

    def save(self, file_path):
        """
            Serialize the model and save it to an HDF5 file.

            Parameters:
            -----------
                file_path (str): The file path where the serialized model will be saved.

            Returns:
            -----------
                None

            Examples:
            -----------
            >>> model.save("Saved Models/USKELM_Model_1.h5")
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
            Load a serialized model from an HDF5 file.

            Parameters:
            -----------
                file_path (str): The file path from which the model will be loaded.

            Returns:
            -----------
                USKELMModel: The loaded model instance.

            Examples:
            -----------
            >>> model = model.load("Saved Models/USKELM_Model_1.h5")
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
                    c = attributes.pop("classification")
                if "name" in attributes:
                    l_type = attributes.pop("name")

                layer = eval(f"{l_type}.load(attributes)")
                model = cls(layer, c)
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None  # Return None or raise an exception based on your error-handling strategy

    def to_dict(self):
        """
            Convert the model to a dictionary.

            Returns:
            -----------
                dict: Dictionary containing the model attributes.
        """
        attributes = self.layer.to_dict()
        attributes["classification"] = self.classification
        attributes["random_weights"] = self.random_weights

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

