import h5py
import numpy as np

from Layers.USELMLayer import USELMLayer


class USELMModel:
    """
        Unsupervised ELM (USELM) Model.

        This class represents an Unsupervised Extreme Learning Machine (USELM) model. It encapsulates the USELMLayer
        and provides methods for fitting the model to data, making predictions, saving and loading the model, and
        converting the model to a dictionary.

        Parameters:
        -----------
        - layer (USELMLayer): The USELMLayer instance representing the underlying USELM model.
        - random_weights (bool): Whether to initialize the model with random weights. Defaults to True.

        Attributes:
        -----------
        - classes_: The classes array, initialized to None.
        - number_neurons: The number of neurons in the layer.
        - activation: The activation function used in the layer.
        - act_params: Additional parameters for the activation function.
        - C: Regularization parameter C.
        - is_orthogonalized: Whether the layer is orthogonalized.
        - layer: The USELMLayer instance representing the underlying USELM model.
        - random_weights: Whether the model is initialized with random weights.

        Methods:
        -----------
        - fit(x): Fit the USELM model to the input data.
        - predict(X, clustering=False, k=None): Make predictions using the trained model.
        - save(file_path): Save the model to an HDF5 file.
        - load(file_path): Load the model from an HDF5 file.
        - to_dict(): Convert the model to a dictionary representation.

        Examples:
        -----------
        Embedding into 3 dimensional space
        ----
        >>> layer = USELMLayer(number_neurons=5000, embedding_size=3, lam=0.001)
        >>> model = USELMModel(layer)
        >>> model.fit(X)

        >>> model.save("Saved Models/USELM_Model_1.h5")
        >>> model = model.load("Saved Models/USELM_Model_1.h5")

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
        >>> layer = USELMLayer(number_neurons=5000, embedding_size=3, lam=0.001)
        >>> model = USELMModel(layer)
        >>> model.fit(X)

        >>> model.save("Saved Models/USELM_Model_1.h5")
        >>> model = model.load("Saved Models/USELM_Model_1.h5")

        >>> pred = model.predict(X, clustering=True, k=10)

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection='3d')

        >>> ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=y.flatten(), cmap=plt.cm.Paired, marker='o', edgecolors='k')

        >>> ax.set_xlabel('Feature 1')
        >>> ax.set_ylabel('Feature 2')
        >>> ax.set_zlabel('Feature 3')
        >>> ax.set_title('3D Scatter Plot with Colored Labels')

        Show the plot
        >>> plt.show()
    """
    def __init__(self, layer: USELMLayer, random_weights=True):
        self.classes_ = None
        self.number_neurons = layer.number_neurons
        self.activation = layer.activation
        self.act_params = layer.act_params
        self.C = layer.C
        self.is_orthogonalized = layer.is_orthogonalized
        self.layer = layer
        self.random_weights = random_weights

    def fit(self, x):
        """
           Fit the USELM model to the input data.

           Parameters:
           -----------
           - x: Input data.

           Examples:
           -----------
           >>> layer = USELMLayer(number_neurons=5000, embedding_size=3, lam=0.001)
           >>> model = USELMModel(layer)
           >>> model.fit(X)
       """
        if self.random_weights:
            self.layer.build(np.shape(x))
        self.classes_ = np.zeros(np.shape(x)[0])
        self.layer.fit(x)

    def predict(self, X, clustering=False, k=None):
        """
            Make predictions using the trained model.

            Parameters:
            -----------
            - X: Input data for prediction.
            - clustering (bool): Whether to perform clustering. Defaults to False.
            - k: Number of clusters if clustering is True.

            Returns:
            -----------
            - Predicted values.

            Examples:
            -----------
            >>> layer = USELMLayer(number_neurons=5000, embedding_size=3, lam=0.001)
            >>> model = USELMModel(layer)
            >>> model.fit(X)
            >>> pred = model.predict(X)
        """
        pred = self.layer.predict(X, clustering, k)
        return pred.numpy()

    def save(self, file_path):
        """
            Save the model to an HDF5 file.

            Parameters:
            -----------
            - file_path (str): File path to save the model.

            Examples:
            -----------
            >>> model.save("Saved Models/USELM_Model_1.h5")
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
            Load the model from an HDF5 file.

            Parameters:
            -----------
            - file_path (str): File path to load the model from.

            Returns:
            -----------
            - Loaded USELMModel instance.

            Examples:
            -----------
            >>> model = model.load("Saved Models/USELM_Model_1.h5")
        """
        try:
            with h5py.File(file_path, 'r') as h5file:
                # Extract attributes from the HDF5 file
                attributes = {key: h5file[key][()] for key in h5file.keys()}

                for key, value in attributes.items():
                    if type(value) is bytes:
                        v = value.decode('utf-8')
                        attributes[key] = v

                if "name" in attributes:
                    l_type = attributes.pop("name")

                layer = eval(f"{l_type}(**attributes)")
                model = cls(layer)
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None  # Return None or raise an exception based on your error-handling strategy

    def to_dict(self):
        """
            Convert the model to a dictionary representation.

            Returns:
            -----------
            - Dictionary containing model attributes.
        """
        attributes = self.layer.to_dict()
        attributes["random_weights"] = self.random_weights

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes