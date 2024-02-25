import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.utils.multiclass import unique_labels

from Layers.SSKELMLayer import SSKELMLayer


class SSKELMModel:
    """
        Semi-Supervised Kernel Extreme Learning Machine (SSKELM) model.

        This model utilizes a semi-supervised version of the Kernel Extreme Learning Machine (KELM) algorithm, implemented
        using the SSKELMLayer, for both labeled and unlabeled data classification or regression tasks.

        Parameters:
        -----------
        - layer (SSKELMLayer): The underlying SSKELM layer.
        - classification (bool): Whether the task is classification (True) or regression (False). Default is True.
        - random_weights (bool): Whether to initialize random weights. Default is True.

        Attributes:
        -----------
        - classes_ (None or array-like): Unique class labels for classification tasks.
        - activation (callable): Activation function.
        - act_params (dict): Parameters for the activation function.
        - C (float): Regularization parameter.
        - classification (bool): Flag indicating whether the task is classification.
        - layer (SSKELMLayer): The underlying SSKELM layer.
        - random_weights (bool): Flag indicating whether random weights are used.

        Methods:
        -----------
        - fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled): Fit the model to labeled and unlabeled data.
        - predict(X): Predict labels or values for the given input data.
        - save(file_path): Save the model to an HDF5 file.
        - load(file_path): Load a model instance from an HDF5 file.
        - to_dict(): Convert the model's attributes to a dictionary.
        - predict_proba(X): Predict class probabilities for the given input data.

        Notes:
        -----------
        - This model supports both classification and regression tasks.
        - It utilizes the SSKELMLayer for semi-supervised learning with kernel-based feature mapping.

        Examples:
        -----------
        Splitting the dataset into labeled, validation, test, and unlabeled sets using semi-supervised split

        >>> X_labeled, X_val, X_test, X_unlabeled, y_labeled, y_val, y_test, y_unlabeled = ss_split_dataset(X, y, 50, 50, 136)

        Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)

        >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

        Initializing the Semi-Supervised Kernel Extreme Learning Machine (SS-KELM) layer
        Regularization parameter: 0.001

        >>> layer = SSKELMLayer(lam=0.01, kernel=kernel)

        Initializing the SS-KELM model with the defined layer

        >>> model = SSKELMModel(layer)

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

        >>> model.save("Saved Models/SS-KELM_Model_1.h5")

        Loading the saved model from the file

        >>> model = model.load("Saved Models/SS-KELM_Model_1.h5")

        Making predictions on the validation and test sets

        >>> pred_test = model.predict(X_test)
        >>> pred_val = model.predict(X_val)
    """
    def __init__(self, layer: SSKELMLayer, classification=True, random_weights=True):
        self.classes_ = None
        self.activation = layer.activation
        self.act_params = layer.act_params
        self.C = layer.C
        self.classification = classification
        self.layer = layer
        self.random_weights = random_weights

    def fit(self, X_labeled, X_unlabeled, y_labeled, y_unlabeled):
        """
            Fit the model to labeled and unlabeled data.

            Parameters:
            -----------
            - X_labeled (np.ndarray or tf.Tensor): Labeled input data.
            - X_unlabeled (np.ndarray or tf.Tensor): Unlabeled input data.
            - y_labeled (np.ndarray or tf.Tensor): Labeled target data.
            - y_unlabeled (np.ndarray or tf.Tensor): Unlabeled target data.

            Returns:
            -----------
            None

            Examples:
            -----------
            Splitting the dataset into labeled, validation, test, and unlabeled sets using semi-supervised split

            >>> X_labeled, X_val, X_test, X_unlabeled, y_labeled, y_val, y_test, y_unlabeled = ss_split_dataset(X, y, 50, 50, 136)

            Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)

            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

            Initializing the Semi-Supervised Kernel Extreme Learning Machine (SS-KELM) layer
            Regularization parameter: 0.001

            >>> layer = SSKELMLayer(lam=0.01, kernel=kernel)

            Initializing the SS-KELM model with the defined layer

            >>> model = SSKELMModel(layer)

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
            Predict labels or values for the given input data.

            Parameters:
            -----------
            - X (np.ndarray or tf.Tensor): Input data.

            Returns:
            -----------
            np.ndarray or tf.Tensor: Predicted labels or values.

            Examples:
            -----------
            Splitting the dataset into labeled, validation, test, and unlabeled sets using semi-supervised split

            >>> X_labeled, X_val, X_test, X_unlabeled, y_labeled, y_val, y_test, y_unlabeled = ss_split_dataset(X, y, 50, 50, 136)

            Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)

            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

            Initializing the Semi-Supervised Kernel Extreme Learning Machine (SS-KELM) layer
            Regularization parameter: 0.001

            >>> layer = SSKELMLayer(lam=0.01, kernel=kernel)

            Initializing the SS-KELM model with the defined layer

            >>> model = SSKELMModel(layer)

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
            Save the model to an HDF5 file.

            Parameters:
            -----------
            - file_path (str): Path to the HDF5 file.

            Returns:
            -----------
            None

            Examples:
            -----------
            Saving the trained model to a file

            >>> model.save("Saved Models/SS-KELM_Model_1.h5")
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
            Load a model instance from an HDF5 file.

            Parameters:
            -----------
            - file_path (str): Path to the HDF5 file.

            Returns:
            -----------
            SSKELMModel: Loaded model instance.

            Examples:
            -----------
            Loading the saved model from the file

            >>> model = model.load("Saved Models/SS-KELM_Model_1.h5")
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
            Convert the model's attributes to a dictionary.

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
            Predict class probabilities for the given input data.

            Parameters:
            -----------
            - X (np.ndarray or tf.Tensor): Input data.

            Returns:
            -----------
            tf.Tensor: Predicted class probabilities' tensor.

            Examples:
            -----------
            Splitting the dataset into labeled, validation, test, and unlabeled sets using semi-supervised split

            >>> X_labeled, X_val, X_test, X_unlabeled, y_labeled, y_val, y_test, y_unlabeled = ss_split_dataset(X, y, 50, 50, 136)

            Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)

            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

            Initializing the Semi-Supervised Kernel Extreme Learning Machine (SS-KELM) layer
            Regularization parameter: 0.001

            >>> layer = SSKELMLayer(lam=0.01, kernel=kernel)

            Initializing the SS-KELM model with the defined layer

            >>> model = SSKELMModel(layer)

            Fitting the SS-ELM model to the labeled and unlabeled data

            >>> model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)

            Making predictions on the validation and test sets

            >>> pred_test = model.predict_proba(X_test)
            >>> pred_val = model.predict_proba(X_val)
        """
        return self.layer.predict_proba(X)
