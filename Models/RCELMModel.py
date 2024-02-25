import h5py
import tensorflow as tf
import tqdm
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.multiclass import unique_labels

from Layers import USELMLayer
from Layers.GELM_AE_Layer import GELM_AE_Layer
from Layers.KELMLayer import KELMLayer
from Layers.ELMLayer import ELMLayer
from Resources.apply_denoising import apply_denoising
from Resources.get_layers import get_layers


class RCELMModel(BaseEstimator, RegressorMixin):
    """
        Residual Compensation Extreme Learning Machine (RC-ELM) model.

        This model consists of a series of layers, where the first layer acts as a baseline model
        (e.g., an Extreme Learning Machine), and subsequent layers are used to compensate for
        the residuals of the previous layers.

        Parameters:
        -----------
        - layers (list): List of layers to be added to the model. Default is None.
        - verbose (int): Verbosity mode. 0 = silent, 1 = verbose. Default is 0.

        Attributes:
        -----------
        - lambdas (tf.Tensor): Lambdas calculated for weighting the predictions of compensation layers.
        - errors (tf.Tensor): Residual errors obtained from compensation layers.
        - layers (list): List of layers in the model.
        - verbose (int): Verbosity mode.

        Methods:
        -----------
        - add(layer): Adds a layer to the model.
        - fit(x, y): Fits the RC-ELM model to the given input-output pairs.
        - predict(x): Predicts the output for the given input data.
        - summary(): Prints a summary of the model architecture.
        - to_dict(): Converts the model into a dictionary representation.
        - save(file_path): Serializes the model and saves it to an HDF5 file.
        - load(file_path): Deserializes a model from an HDF5 file.

        Examples:
        -----------
        Initialize a Residual Compensation Multilayer Extreme Learning Machine model

        >>> model = RCELMModel()

        Add ELM layers to the Multilayer Extreme Learning Machine

        >>> model.add(ELMLayer(number_neurons=1000, activation='sigmoid', C=10))
        >>> model.add(ELMLayer(number_neurons=2000, activation='sigmoid', C=10))
        >>> model.add(ELMLayer(number_neurons=1000, activation='sigmoid', C=10))

        Fit the RC-MLELM model to the entire dataset

        >>> model.fit(X, y)
        >>> y_pred = model.predict(X)
        >>> print(calculate_rmse(y, y_pred))

        Save the trained model to a file

        >>> model.save('Saved Models/RC_ML_ELM_Model.h5')

        Load the saved model from the file

        >>> model = model.load('Saved Models/RC_ML_ELM_Model.h5')

        Evaluate the accuracy of the model on the training data

        >>> y_pred = model.predict(X_test)
        >>> print(calculate_rmse(15, y_pred))
    """
    def __init__(self, layers=None, verbose=0):
        self.lambdas = None
        self.errors = None
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        self.verbose = verbose

    def add(self, layer):
        """
            Add a layer to the model.

            Parameters:
            -----------
            - layer: Layer to be added to the model.

            Returns:
            -----------
            None

            Examples:
            -----------
            Initialize a Residual Compensation Multilayer Extreme Learning Machine model

            >>> model = RCELMModel()

            Add ELM layers to the Multilayer Extreme Learning Machine

            >>> model.add(ELMLayer(number_neurons=1000, activation='sigmoid', C=10))
            >>> model.add(ELMLayer(number_neurons=2000, activation='sigmoid', C=10))
            >>> model.add(ELMLayer(number_neurons=1000, activation='sigmoid', C=10))

            Fit the RC-MLELM model to the entire dataset

            >>> model.fit(X, y)
            >>> y_pred = model.predict(X)
            >>> print(calculate_rmse(y, y_pred))

            Save the trained model to a file

            >>> model.save('Saved Models/RC_ML_ELM_Model.h5')

            Load the saved model from the file

            >>> model = model.load('Saved Models/RC_ML_ELM_Model.h5')

            Evaluate the accuracy of the model on the training data

            >>> y_pred = model.predict(X_test)
            >>> print(calculate_rmse(15, y_pred))
        """
        self.layers.append(layer)

    def fit(self, x, y):
        """
            Fit the RC-ELM model to the given input-output pairs.

            Parameters:
            -----------
            - x (numpy.ndarray): Input data tensor.
            - y (numpy.ndarray): Output labels.

            Returns:
            -----------
            None

            Examples:
            -----------
            Initialize a Residual Compensation Multilayer Extreme Learning Machine model

            >>> model = RCELMModel()

            Add ELM layers to the Multilayer Extreme Learning Machine

            >>> model.add(ELMLayer(number_neurons=1000, activation='sigmoid', C=10))
            >>> model.add(ELMLayer(number_neurons=2000, activation='sigmoid', C=10))
            >>> model.add(ELMLayer(number_neurons=1000, activation='sigmoid', C=10))

            Fit the RC-MLELM model to the entire dataset

            >>> model.fit(X, y)
        """
        if self.verbose == 1:
            pbar = tqdm.tqdm(total=len(self.layers), desc=' RC-ELM : Baseline layer step ')

        for layer in self.layers:
            layer.build(x.shape)

        elm_baseline = self.layers[0]
        elm_baseline.fit(x, y)
        y_hat = elm_baseline.predict(x)
        e_k = y - y_hat

        if self.verbose == 1:
            pbar.set_description(' RC-ELM : Compensation layer step ')

        self.errors = []
        self.lambdas = []

        i = 1
        for layer in self.layers[1:]:
            elm_k = layer
            if layer.denoising is None:
                elm_k.fit(x, e_k)
            else:
                x_noised = apply_denoising(x, layer.denoising, layer.denoising_param)
                elm_k.fit(x_noised, e_k)
            e_k1_hat = elm_k.predict(x)
            e_k = e_k - e_k1_hat
            self.errors.append(e_k)
            self.lambdas.append(1 / e_k ** 2)

            if self.verbose == 1:
                pbar.update(n=i)
                i = i + 1

        self.lambdas = self.lambdas / sum(self.lambdas)
        self.errors = tf.stack(self.errors, axis=0)

        if self.verbose == 1:
            pbar.update(n=i+1)
            pbar.close()

    def predict(self, x):
        """
            Predict the output for the given input data.

            Parameters:
            -----------
            - x (numpy.ndarray): Input data tensor.

            Returns:
            -----------
            numpy.ndarray: Predicted output tensor.

            Examples:
            -----------
            Initialize a Residual Compensation Multilayer Extreme Learning Machine model

            >>> model = RCELMModel()

            Add ELM layers to the Multilayer Extreme Learning Machine

            >>> model.add(ELMLayer(number_neurons=1000, activation='sigmoid', C=10))
            >>> model.add(ELMLayer(number_neurons=2000, activation='sigmoid', C=10))
            >>> model.add(ELMLayer(number_neurons=1000, activation='sigmoid', C=10))

            Fit the RC-MLELM model to the entire dataset

            >>> model.fit(X, y)
            >>> y_pred = model.predict(X)
            >>> print(calculate_rmse(y, y_pred))
        """
        elm_baseline = self.layers[0]
        y_hat = elm_baseline.predict(x)

        preds = []
        for layer in self.layers[1:]:
            y_pred = layer.predict(x)
            preds.append(y_pred)

        preds = tf.stack(preds, axis=0)
        product = self.lambdas * preds
        stacked_errors = tf.reduce_sum(product, axis=0)

        y_pred = y_hat + stacked_errors
        return y_pred.numpy()

    def summary(self):
        """
            Print a summary of the model architecture.

            Returns:
            -----------
            None
        """
        total = 0
        trainable = 0
        non_trainable = 0
        i = 0
        prev = None
        print("_________________________________________________________________")
        print(" Layer (type)                Output Shape              Param #   ")
        print("=================================================================")
        for layer in self.layers:
            if layer.__class__ is not prev.__class__:
                i = 0
            if layer.output is None:
                sh = "Unknown"
            else:
                sh = layer.output.shape
            print(f"{layer}_{i} ({layer.__class__.__name__})         {sh}                  {layer.count_params()['all']}      ")
            total = total + layer.count_params()['all']
            trainable = trainable + layer.count_params()['trainable']
            non_trainable = non_trainable + layer.count_params()['non_trainable']
            i = i + 1
            prev = layer
        print("=================================================================")
        print(f"Total params: {total}")
        print(f"Trainable params: {trainable}")
        print(f"Non-trainable params: {non_trainable}")
        print("_________________________________________________________________")

    def to_dict(self):
        """
            Convert the model into a dictionary representation.

            Returns:
            -----------
            dict: Dictionary representation of the model.
        """
        attributes = {
            'verbose': self.verbose,
            'lambdas': self.lambdas,
            'errors': self.errors
        }
        for i, layer in enumerate(self.layers):
            key_prefix = f'layer.{i}.'
            for key, value in layer.to_dict().items():
                k = key_prefix + key
                attributes.update({k: value})
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

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

        >>> model.save('Saved Models/RC_ML_ELM_Model.h5')
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
            RCELMModel: An instance of the RCELMModel class loaded from the file.

            Examples:
            -----------
            Load the saved model from the file

            >>> model = model.load('Saved Models/RC_ML_ELM_Model.h5')
        """
        try:
            with h5py.File(file_path, 'r') as h5file:
                # Extract attributes from the HDF5 file
                attributes = {key: h5file[key][()] for key in h5file.keys()}

                for key, value in attributes.items():
                    if type(value) is bytes:
                        v = value.decode('utf-8')
                        attributes[key] = v

                v = attributes.pop('verbose')
                l = attributes.pop('lambdas')
                e = attributes.pop('errors')

                model = cls(verbose=v)
                model.lambdas = l
                model.errors = e

                layers = get_layers(attributes)
                model.layers = layers
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None  # Return None or raise an exception based on your error-handling strategy