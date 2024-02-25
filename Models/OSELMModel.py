import h5py
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm

from Layers.OSELMLayer import OSELMLayer
from Optimizers.ELMOptimizer import ELMOptimizer


class OSELMModel(BaseEstimator, ClassifierMixin):
    """
        Online Sequential Extreme Learning Machine (OS-ELM) model.

        This model provides an implementation of the Online Sequential Extreme Learning Machine (OS-ELM),
        a variant of Extreme Learning Machine (ELM) suitable for online learning scenarios.

        Parameters:
        -----------
        - oselm (OSELMLayer): An instance of the OSELMLayer representing the ELM layer.
        - prefetch_size (int): Size of the initial data to be used for ELM initialization.
        - batch_size (int): Size of the mini-batches for sequential learning.
        - classification (bool): If True, the model is used for classification, and output labels are one-hot encoded.
        - verbose (int): Verbosity mode (0 or 1).

        Attributes:
        -----------
        - classes_ (numpy.ndarray): Unique classes observed during fitting.

        Methods:
        -----------
        - fit(x, y): Fit the OS-ELM model to the given input-output pairs.
        - predict(X): Predict the output for the given input data.
        - save(file_path): Serialize the current instance and save it to a HDF5 file.
        - load(file_path): Deserialize an instance from a file.
        - to_dict(): Convert the current instance to a dictionary.
        - predict_proba(X): Predict the probabilities output for the given input data.

        Examples:
        -----------
        Initialize OSELMLayer with specified parameters

        >>> layer = OSELMLayer(1000, 'tanh')

        Initialize OSELMModel with the OSELMLayer and other parameters

        >>> model = OSELMModel(layer, prefetch_size=1200, batch_size=64, verbose=0)

        Perform cross-validation

        >>> cv = RepeatedKFold(n_splits=10, n_repeats=50)
        >>> scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

        Print mean accuracy score

        >>> print(np.mean(scores))

        Fit the ML-ELM model to the entire dataset

        >>> model.fit(X, y)

        Save the trained model to a file

        >>> model.save('Saved Models/OSELM_Model.h5')

        Load the saved model from the file

        >>> model = model.load('Saved Models/OSELM_Model.h5')

        Evaluate the accuracy of the model on the training data

        >>> acc = accuracy_score(model.predict(X), y)
    """
    def __init__(self,
                 oselm,
                 prefetch_size,
                 batch_size,
                 classification=True,
                 verbose=0):
        self.classes_ = None
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.verbose = verbose
        self.classification = classification
        self.oselm = oselm

    def fit(self, x, y):
        """
        Fits the ELM model to the given input-output pairs.

        Parameters:
        -----------
        - X (numpy.ndarray): Input data tensor.
        - y (numpy.ndarray): Output labels.

        Returns:
        -----------
        None

        Examples:
        -----------
        Initialize OSELMLayer with specified parameters

        >>> layer = OSELMLayer(1000, 'tanh')

        Initialize OSELMModel with the OSELMLayer and other parameters

        >>> model = OSELMModel(layer, prefetch_size=1200, batch_size=64, verbose=0)

        Fit the ML-ELM model to the entire dataset

        >>> model.fit(X, y)
        """
        y = to_categorical(y)
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        train_x_init = x[:self.prefetch_size]
        train_x_seq = x[self.prefetch_size:]
        train_y_init = y[:self.prefetch_size]
        train_y_seq = y[self.prefetch_size:]

        if self.verbose == 1:
            pbar = tqdm(total=len(train_x_seq)//self.batch_size, desc='OS-ELM : Initialize step')
        self.oselm.build(x.shape)
        self.oselm.fit_initialize(train_x_init, train_y_init)

        if self.verbose == 1:
            pbar.set_description('OS-ELM : Sequential step')
        j = 1
        for i in range(0, len(train_x_seq), self.batch_size):
            train_x_batch = train_x_seq[i:i + self.batch_size]
            train_y_batch = train_y_seq[i:i + self.batch_size]
            self.oselm.fit_seq(train_x_batch, train_y_batch)
            if self.verbose == 1:
                pbar.update(n=j)
                j += 1
        if self.verbose == 1:
            pbar.update(n=j + 1)
            pbar.close()
        self.classes_ = unique_labels(y)

    def predict(self, X):
        """
        Predicts the output for the given input data.

        Parameters:
        -----------
        - X (tf.Tensor): Input data tensor.

        Returns:
        -----------
        tf.Tensor: Predicted output tensor.

        Examples:
        -----------
        Initialize OSELMLayer with specified parameters

        >>> layer = OSELMLayer(1000, 'tanh')

        Initialize OSELMModel with the OSELMLayer and other parameters

        >>> model = OSELMModel(layer, prefetch_size=1200, batch_size=64, verbose=0)

        Fit the ML-ELM model to the entire dataset

        >>> model.fit(X, y)

        Evaluate the accuracy of the model on the training data

        >>> acc = accuracy_score(model.predict(X), y)
        """
        pred = self.oselm.predict(X)
        if self.classification:
            return tf.math.argmax(pred, axis=1).numpy()
        else:
            return pred.numpy()

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

        >>> model.save('Saved Models/OSELM_Model.h5')
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
        ELMLayer: An instance of the ELMLayer class loaded from the file.

        Examples:
        -----------
        Load the saved model from the file

        >>> model = model.load('Saved Models/OSELM_Model.h5')
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
                if "prefetch_size" in attributes:
                    prefetch_size = attributes.pop("prefetch_size")
                if "batch_size" in attributes:
                    batch_size = attributes.pop("batch_size")
                if "verbose" in attributes:
                    verbose = attributes.pop("verbose")

                layer = eval(f"{l_type}(**attributes)")
                model = cls(layer, classification=c, prefetch_size=prefetch_size, batch_size=batch_size, verbose=verbose)
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None  # Return None or raise an exception based on your error-handling strategy

    def to_dict(self):
        """
            Convert the current instance to a dictionary.

            Returns:
            -----------
            dict: Dictionary containing instance attributes.
        """
        attributes = self.oselm.to_dict()
        attributes["classification"] = self.classification,
        attributes["prefetch_size"] = self.prefetch_size,
        attributes["batch_size"] = self.batch_size,
        attributes["verbose"] = self.verbose,
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    def predict_proba(self, X):
        """
            Predict the probabilities output for the given input data.

            Parameters:
            -----------
            - X (tf.Tensor): Input data tensor.

            Returns:
            -----------
            tf.Tensor: Predicted probabilities tensor.

            Examples:
            -----------
            Initialize OSELMLayer with specified parameters

            >>> layer = OSELMLayer(1000, 'tanh')

            Initialize OSELMModel with the OSELMLayer and other parameters

            >>> model = OSELMModel(layer, prefetch_size=1200, batch_size=64, verbose=0)

            Perform cross-validation

            >>> cv = RepeatedKFold(n_splits=10, n_repeats=50)
            >>> scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

            Print mean accuracy score

            >>> print(np.mean(scores))

            Fit the ML-ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the prediction probability of the model on the training data

            >>> acc = accuracy_score(model.predict(X), y)
        """
        return self.oselm.predict_proba(X)
