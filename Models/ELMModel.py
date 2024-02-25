import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from Layers.ELMLayer import ELMLayer
from Layers.WELMLayer import WELMLayer
from Layers.KELMLayer import KELMLayer
from Layers.SubELMLayer import SubELMLayer


class ELMModel(BaseEstimator, ClassifierMixin):
    """
    Extreme Learning Machine model.

    This class implements an Extreme Learning Machine (ELM) model, which is a single-hidden-layer feedforward neural
    network. The model can be used for both classification and regression tasks.

    Parameters:
    -----------
    layer : ELMLayer
        The hidden layer of the ELM model.
    classification : bool, default=True
        Indicates whether the model is used for classification (True) or regression (False) tasks.
    random_weights : bool, default=True
        Indicates whether to randomly initialize the weights of the hidden layer.

    Attributes:
    -----------
    classes_ : array-like, shape (n_classes,)
        The unique class labels in the training data.

    Examples:
    -----------
    Initialize an Extreme Learning Machine (ELM) layer with 1000 neurons

    >>> elm = ELMLayer(number_neurons=1000, activation='mish')

    Create an ELM model using the trained ELM layer

    >>> model = ELMModel(elm)

    Define a cross-validation strategy

    >>> cv = RepeatedKFold(n_splits=10, n_repeats=50)

    Perform cross-validation to evaluate the model performance

    >>> scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

    Print the mean accuracy score obtained from cross-validation

    >>> print(np.mean(scores))

    Fit the ELM model to the entire dataset

    >>> model.fit(X, y)

    Save the trained model to a file

    >>> model.save("Saved Models/ELM_Model.h5")

    Load the saved model from the file

    >>> model = model.load("Saved Models/ELM_Model.h5")

    Evaluate the accuracy of the model on the training data

    >>> acc = accuracy_score(model.predict(X), y)
    >>> print(acc)

    """
    def __init__(self, layer, classification=True, random_weights=True):
        self.classes_ = None
        self.classification = classification
        self.layer = layer
        self.random_weights = random_weights

    def fit(self, X, y):
        """
            Fit the ELM model to the training data.

            Parameters:
            -----------
            X : array-like, shape (n_samples, n_features)
                The input training data.
            y : array-like, shape (n_samples,) or (n_samples, n_classes)
                The target values for classification or regression tasks.

            Returns:
            --------
            self : object
                Returns the instance itself.

            Example:
            -----------
            Initialize an Extreme Learning Machine (ELM) layer with 1000 neurons

            >>> elm = ELMLayer(number_neurons=1000, activation='mish')

            Create an ELM model using the trained ELM layer

            >>> model = ELMModel(elm)

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)
        """
        if self.random_weights:
            self.layer.build(X.shape)
        if self.classification:
            self.classes_ = unique_labels(y)
        else:
            self.classes_ = [0]

        if len(np.shape(y)) == 1:
            y = to_categorical(y)

        self.layer.fit(X, y)

    def predict(self, X):
        """
            Predict class labels or regression values for the input data.

            Parameters:
            -----------
            X : array-like, shape (n_samples, n_features)
                The input data.

            Returns:
            --------
            y_pred : array-like, shape (n_samples,)
                The predicted class labels or regression values.

            Example:
            -----------
            Initialize an Extreme Learning Machine (ELM) layer with 1000 neurons

            >>> elm = ELMLayer(number_neurons=1000, activation='mish')

            Create an ELM model using the trained ELM layer

            >>> model = ELMModel(elm)

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the accuracy of the model on the training data

            >>> acc = accuracy_score(model.predict(X), y)
        """
        pred = self.layer.predict(X)
        if self.classification:
            return tf.math.argmax(pred, axis=1).numpy()
        else:
            return pred.numpy()

    def predict_proba(self, x):
        """
            Predict class probabilities for the input data.

            Parameters:
            -----------
            X : array-like, shape (n_samples, n_features)
                The input data.

            Returns:
            --------
            y_proba : array-like, shape (n_samples, n_classes)
                The predicted class probabilities.

            Example:
            -----------
            Initialize an Extreme Learning Machine (ELM) layer with 1000 neurons

            >>> elm = ELMLayer(number_neurons=1000, activation='mish')

            Create an ELM model using the trained ELM layer

            >>> model = ELMModel(elm)

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the accuracy of the model on the training data

            >>> pred_proba = model.predict_proba(X)
        """
        return self.layer.predict_proba(x)

    def save(self, file_path):
        """
            Save the model to an HDF5 file.

            Parameters:
            -----------
            file_path : str
                The path to the HDF5 file where the model will be saved.

            Example:
            -----------
            Save the trained model to a file

            >>> model.save("Saved Models/ELM_Model.h5")
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
            Load a model from an HDF5 file.

            Parameters:
            -----------
            file_path : str
                The path to the HDF5 file containing the model.

            Returns:
            --------
            model : ELMModel
                The loaded ELMModel instance.

            Example:
            -----------
            Load the saved model from the file

            >>> model = ELMModel.load("Saved Models/ELM_Model.h5")
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
                if "random_weights" in attributes:
                    r = attributes.pop("random_weights")

                layer = eval(f"{l_type}(**attributes)")
                model = cls(layer, c, r)
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None  # Return None or raise an exception based on your error-handling strategy

    def to_dict(self):
        """
            Convert the model to a dictionary of attributes.

            Returns:
            --------
            attributes : dict
                A dictionary containing the attributes of the model.
        """
        attributes = self.layer.to_dict()
        attributes["classification"] = self.classification
        attributes["random_weights"] = self.random_weights

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes


