import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

from Layers.SSELMLayer import SSELMLayer


class SSELMModel(BaseEstimator, ClassifierMixin):
    """
        Semi-Supervised Extreme Learning Machine (SSELM) model.

        This class represents a semi-supervised learning model based on the Extreme Learning Machine (ELM) architecture.
        It combines labeled and unlabeled data for training, allowing for enhanced performance in scenarios where labeled
        data is limited. The model consists of an underlying SSELMLayer, which is responsible for the core computations
        and learning processes.

        Parameters:
        -----------
        - layer (SSELMLayer): Instance of the SSELMLayer serving as the core computational unit of the model.
        - classification (bool): Indicates whether the model is used for classification. Defaults to True.
        - random_weights (bool): Specifies whether to initialize weights randomly. Defaults to True.

        Attributes:
        -----------
        - classes_ (numpy.ndarray): Array of unique class labels present in the labeled training data.
        - number_neurons (int): Number of neurons in the SSELMLayer.
        - activation (function): Activation function used in the SSELMLayer.
        - act_params (dict): Parameters of the activation function.
        - C (float): Regularization parameter.
        - is_orthogonalized (bool): Indicates whether the SSELMLayer is orthogonalized.
        - beta_optimizer (ELMOptimizer): Optimizer used for updating beta weights.
        - classification (bool): Indicates whether the model is used for classification.
        - layer (SSELMLayer): Instance of the SSELMLayer serving as the core computational unit of the model.
        - random_weights (bool): Specifies whether weights are randomly initialized.

        Methods:
        -----------
        - fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled): Fits the model to labeled and unlabeled data.
        - predict(X): Predicts outputs for the given input data.
        - save(file_path): Serializes and saves the model to a HDF5 file.
        - load(file_path): Loads a previously serialized model from a HDF5 file.
        - to_dict(): Converts model attributes to a dictionary.
        - predict_proba(X): Predicts output probabilities for the given input data.

        Examples:
        -----------
        Splitting the dataset into labeled, validation, test, and unlabeled sets using semi-supervised split

        >>> X_labeled, X_val, X_test, X_unlabeled, y_labeled, y_val, y_test, y_unlabeled = ss_split_dataset(X, y, 50, 50, 136)

        Initializing the Semi-Supervised Extreme Learning Machine (SS-ELM) layer
        Number of neurons: 1000, Regularization parameter: 0.001

        >>> layer = SSELMLayer(number_neurons=1000, lam=0.001)

        Initializing the SS-ELM model with the defined layer

        >>> model = SSELMModel(layer)

        Performing semi-supervised cross-validation using repeated k-fold
        Number of labeled, validation, test, and unlabeled samples are provided as parameters

        >>> cv = SSRepeatedKFold(n_splits=(50, 314, 50, 136), n_repeats=50)

        Scoring the model on validation and test sets using ROC AUC metric

        >>> scores_val, scores_test = ss_cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

        Printing mean scores for validation and test sets

        >>> print("Valid: " + str(np.mean(scores_val)))
        >>> print("Test: " + str(np.mean(scores_test)))

        Fitting the SS-ELM model to the labeled and unlabeled data

        >>> model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)

        Saving the trained model to a file

        >>> model.save("Saved Models/SS-ELM_Model_1.h5")

        Loading the saved model from the file

        >>> model = model.load("Saved Models/SS-ELM_Model_1.h5")

        Making predictions on the validation and test sets

        >>> pred_test = model.predict(X_test)
        >>> pred_val = model.predict(X_val)
    """
    def __init__(self, layer: SSELMLayer, classification=True, random_weights=True):
        self.classes_ = None
        self.number_neurons = layer.number_neurons
        self.activation = layer.activation
        self.act_params = layer.act_params
        self.C = layer.C
        self.is_orthogonalized = layer.is_orthogonalized
        self.beta_optimizer = layer.beta_optimizer
        self.classification = classification
        self.layer = layer
        self.random_weights = random_weights

    def fit(self, X_labeled, X_unlabeled, y_labeled, y_unlabeled):
        """
            Fits the model to labeled and unlabeled data.

            Parameters:
            -----------
            - X_labeled (numpy.ndarray): Labeled input features.
            - X_unlabeled (numpy.ndarray): Unlabeled input features.
            - y_labeled (numpy.ndarray): Labeled target labels.
            - y_unlabeled (numpy.ndarray): Unlabeled target labels.

            Returns:
            -----------
            None

            Examples:
            -----------
            Splitting the dataset into labeled, validation, test, and unlabeled sets using semi-supervised split

            >>> X_labeled, X_val, X_test, X_unlabeled, y_labeled, y_val, y_test, y_unlabeled = ss_split_dataset(X, y, 50, 50, 136)

            Initializing the Semi-Supervised Extreme Learning Machine (SS-ELM) layer
            Number of neurons: 1000, Regularization parameter: 0.001

            >>> layer = SSELMLayer(number_neurons=1000, lam=0.001)

            Initializing the SS-ELM model with the defined layer

            >>> model = SSELMModel(layer)

            Fitting the SS-ELM model to the labeled and unlabeled data

            >>> model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
        """
        if self.random_weights:
            self.layer.build(np.shape(X_labeled))
        self.classes_ = unique_labels(y_labeled)
        y_labeled_cat = to_categorical(y_labeled)
        y_unlabeled_cat = to_categorical(y_unlabeled)
        self.layer.fit(X_labeled, X_unlabeled, y_labeled_cat, y_unlabeled_cat)

    def predict(self, X):
        """
            Predicts the output for the given input data.

            Parameters:
            -----------
            - X (numpy.ndarray): Input data.

            Returns:
            -----------
            numpy.ndarray: Predicted output.

            Examples:
            -----------
            Splitting the dataset into labeled, validation, test, and unlabeled sets using semi-supervised split

            >>> X_labeled, X_val, X_test, X_unlabeled, y_labeled, y_val, y_test, y_unlabeled = ss_split_dataset(X, y, 50, 50, 136)

            Initializing the Semi-Supervised Extreme Learning Machine (SS-ELM) layer
            Number of neurons: 1000, Regularization parameter: 0.001

            >>> layer = SSELMLayer(number_neurons=1000, lam=0.001)

            Initializing the SS-ELM model with the defined layer

            >>> model = SSELMModel(layer)

            Fitting the SS-ELM model to the labeled and unlabeled data

            >>> model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)

            Making predictions on the validation and test sets

            >>> pred_test = model.predict(X_test)
            >>> pred_val = model.predict(X_val)
        """
        pred = self.layer.predict(X)
        if self.classification:
            return tf.math.argmax(pred, axis=1).numpy()
        else:
            return pred.numpy()

    def save(self, file_path):
        """
            Serialize the model and save it to a HDF5 file.

            Parameters:
            -----------
            - file_path (str): The file path where the serialized model will be saved.

            Returns:
            -----------
            None

            Examples:
            -----------
            Saving the trained model to a file

            >>> model.save("Saved Models/SS-ELM_Model_1.h5")
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
            Load a previously serialized model from a HDF5 file.

            Parameters:
            -----------
            - file_path (str): The file path from which to load the serialized model.

            Returns:
            -----------
            SSELMModel: Loaded model instance.

            Examples:
            -----------
            Loading the saved model from the file

            >>> model = model.load("Saved Models/SS-ELM_Model_1.h5")
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

                layer = eval(f"{l_type}(**attributes)")
                model = cls(layer, c)
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None  # Return None or raise an exception based on your error-handling strategy

    def to_dict(self):
        """
            Converts the model's attributes to a dictionary.

            Returns:
            -----------
            dict: Dictionary containing the model's attributes.
        """
        attributes = self.layer.to_dict()
        attributes["classification"] = self.classification
        attributes["random_weights"] = self.random_weights

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    def predict_proba(self, X):
        """
            Predicts the probabilities output for the given input data.

            Parameters:
            -----------
            - X (numpy.ndarray): Input data.

            Returns:
            -----------
            numpy.ndarray: Predicted output probabilities.

            Examples:
            -----------
            Splitting the dataset into labeled, validation, test, and unlabeled sets using semi-supervised split

            >>> X_labeled, X_val, X_test, X_unlabeled, y_labeled, y_val, y_test, y_unlabeled = ss_split_dataset(X, y, 50, 50, 136)

            Initializing the Semi-Supervised Extreme Learning Machine (SS-ELM) layer
            Number of neurons: 1000, Regularization parameter: 0.001

            >>> layer = SSELMLayer(number_neurons=1000, lam=0.001)

            Initializing the SS-ELM model with the defined layer

            >>> model = SSELMModel(layer)

            Fitting the SS-ELM model to the labeled and unlabeled data

            >>> model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)

            Making prediction probabilities on the validation and test sets

            >>> pred_test = model.predict_proba(X_test)
            >>> pred_val = model.predict_proba(X_val)
        """
        return self.layer.predict_proba(X)
