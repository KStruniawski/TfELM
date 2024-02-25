import h5py
import numpy as np
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf
from Layers.KELMLayer import KELMLayer
from Resources.Kernel import Kernel, CombinedSumKernel, CombinedProductKernel


class KELMModel(BaseEstimator, ClassifierMixin):
    """
        Kernel Extreme Learning Machine (KELM) Model.

        This class implements a Kernel Extreme Learning Machine model for classification or regression tasks.

        Parameters:
        -----------
            kelm (KELMLayer): Instance of a Kernel ELM model.
            classification (bool, optional): Whether the task is classification or regression.
                Defaults to True.

        Attributes:
        -----------
            classes_ (array-like): Unique class labels.
            classification (bool): Indicates whether the task is classification or regression.
            kelm (KELMLayer): Instance of a Kernel ELM model.

        Methods:
        -----------
            fit(X, y): Fit the KELM model to training data.
            predict(X): Predict class labels or regression values for input data.
            to_dict(): Convert the model to a dictionary of attributes.
            load(file_path): Deserialize an instance from a file.
            save(file_path): Serialize the current instance and save it to a HDF5 file.
            predict_proba(X): Predict class labels or regression values for input data.

        Examples:
        -----------
        Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)

        >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

        Initialize a Kernel Extreme Learning Machine (KELM) layer

        >>> layer = KELMLayer(kernel, 'mish')

        Initialize a Kernel Extreme Learning Machine (KELM) model

        >>> model = KELMModel(layer)

        Define a cross-validation strategy

        >>> cv = RepeatedKFold(n_splits=10, n_repeats=50)

        Perform cross-validation to evaluate the model performance

        >>> scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

        Print the mean accuracy score obtained from cross-validation

        >>> print(np.mean(scores))

        Fit the ELM model to the entire dataset

        >>> model.fit(X, y)

        Save the trained model to a file

        >>> model.save("Saved Models/KELM_Model.h5")

        Load the saved model from the file

        >>> model = model.load("Saved Models/KELM_Model.h5")

        Evaluate the accuracy of the model on the training data

        >>> acc = accuracy_score(model.predict(X), y)
    """
    def __init__(self, kelm: KELMLayer, classification=True):
        self.classes_ = None
        self.classification = classification
        self.kelm = kelm

    def fit(self, X, y):
        """
            Fit the KELM model to training data.

            Args:
            -----------
                X (array-like): Training input samples.
                y (array-like): Target values.

            Examples:
            -----------
            Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)

            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

            Initialize a Kernel Extreme Learning Machine (KELM) layer

            >>> layer = KELMLayer(kernel, 'mish')

            Initialize a Kernel Extreme Learning Machine (KELM) model

            >>> model = KELMModel(layer)

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)
        """
        self.kelm.build(X.shape)

        if self.classification:
            self.classes_ = unique_labels(y)
        else:
            self.classes_ = [0]

        if len(np.shape(y)) == 1:
            y = to_categorical(y)

        self.kelm.fit(X, y)

    def predict(self, X):
        """
            Predict class labels or regression values for input data.

            Args:
            -----------
                X (array-like): Input samples.

            Returns:
            -----------
                array-like: Predicted class labels or regression values.

            Examples:
            -----------
            Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)

            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

            Initialize a Kernel Extreme Learning Machine (KELM) layer

            >>> layer = KELMLayer(kernel, 'mish')

            Initialize a Kernel Extreme Learning Machine (KELM) model

            >>> model = KELMModel(layer)

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the accuracy of the model on the training data

            >>> acc = accuracy_score(model.predict(X), y)
        """
        pred = self.kelm.predict(X)
        if self.classification:
            return tf.math.argmax(pred, axis=1)
        else:
            return pred

    def to_dict(self):
        """
            Convert the model to a dictionary of attributes.

            Returns:
            --------
            attributes : dict
                A dictionary containing the attributes of the model.
        """
        attributes = self.kelm.to_dict()
        attributes["classification"] = self.classification

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, file_path: str):
        """
        Deserialize an instance from a file.

        Parameters:
        - file_path (str): The file path from which to load the serialized instance.

        Returns:
        ELMLayer: An instance of the ELMLayer class loaded from the file.
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

    def save(self, file_path):
        """
        Serialize the current instance and save it to a HDF5 file.

        Parameters:
        - path (str): The file path where the serialized instance will be saved.

        Returns:
        None
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

    def predict_proba(self, X):
        """
            Predict class labels or regression values for input data.

            Args:
            -----------
                X (array-like): Input samples.

            Returns:
            -----------
                array-like: Predicted class labels or regression values.

            Examples:
            -----------
            Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)

            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

            Initialize a Kernel Extreme Learning Machine (KELM) layer

            >>> layer = KELMLayer(kernel, 'mish')

            Initialize a Kernel Extreme Learning Machine (KELM) model

            >>> model = KELMModel(layer)

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the prediction proba of the model on the training data

            >>> pred_proba = model.predict_proba(X)
        """
        pred = self.kelm.predict(X)
        return tf.keras.activations.softmax(pred)
