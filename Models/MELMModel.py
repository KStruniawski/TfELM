import h5py
import tensorflow as tf
import tqdm
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin

from Layers import USELMLayer
from Layers.GELM_AE_Layer import GELM_AE_Layer
from Layers.KELMLayer import KELMLayer
from Layers.ELMLayer import ELMLayer
from Resources.apply_denoising import apply_denoising
from Resources.get_layers import get_layers


class MELMModel(BaseEstimator, ClassifierMixin):
    """
        Multi-layer Extreme Learning Machine (MELM) model.

        MELMModel is a multi-layer variant of Extreme Learning Machine (ELM) model,
        consisting of multiple layers, each with its own computational units.

        Parameters:
        -----------
        layers (list, optional): List of layers. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.
        verbose (int): Verbosity level.
            Controls the amount of information printed during model fitting and prediction.

        Attributes:
        -----------
        classes_ (array-like): The classes labels.
            Initialized to None and populated during fitting.

        Examples:
        -----------
        Create a Multilayer ELM model

        >>> model = MELMModel()

        Add 3 layers of neurons

        >>> model.add(ELMLayer(number_neurons=1000, activation='mish', C=10))
        >>> model.add(ELMLayer(number_neurons=2000, activation='mish', C=10))
        >>> model.add(ELMLayer(number_neurons=1000, activation='mish', C=10))

        Define a cross-validation strategy

        >>> cv = RepeatedKFold(n_splits=10, n_repeats=50)

        Perform cross-validation to evaluate the model performance

        >>> scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

        Print the mean accuracy score obtained from cross-validation

        >>> print(np.mean(scores))

        Fit the ELM model to the entire dataset

        >>> model.fit(X, y)

        Save the trained model to a file

        >>> model.save("Saved Models/MELM_Model.h5")

        Load the saved model from the file

        >>> model = model.load("Saved Models/MELM_Model.h5")

        Evaluate the accuracy of the model on the training data

        >>> acc = accuracy_score(model.predict(X), y)
    """

    def __init__(self, layers=None, verbose=0):
        self.classes_ = None
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        self.verbose = verbose

    def add(self, layer):
        """
            Add a layer to the model.

            Args:
            -----------
                layer: Layer to be added.

            Examples:
            -----------
            Create a Multilayer ELM model

            >>> model = MELMModel()

            Add 3 layers of neurons

            >>> model.add(ELMLayer(number_neurons=1000, activation='mish', C=10))
            >>> model.add(ELMLayer(number_neurons=2000, activation='mish', C=10))
            >>> model.add(ELMLayer(number_neurons=1000, activation='mish', C=10))
        """
        self.layers.append(layer)

    def fit(self, x, y):
        """
            Fit the MELM model to input-output pairs.

            This method trains the MELM model by iteratively fitting each layer in the model.
            During the training process, input data is propagated through each layer.

            Args:
            -----------
                x (array-like): Input data.
                y (array-like): Output data.

            Examples:
            -----------
            Create a Multilayer ELM model

            >>> model = MELMModel()

            Add 3 layers of neurons

            >>> model.add(ELMLayer(number_neurons=1000, activation='mish', C=10))
            >>> model.add(ELMLayer(number_neurons=2000, activation='mish', C=10))
            >>> model.add(ELMLayer(number_neurons=1000, activation='mish', C=10))

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)
        """
        if len(y.shape) == 1:
            y = to_categorical(y)

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

            if self.verbose == 1:
                pbar.update(n=i)
                i = i + 1

        if self.verbose == 1:
            pbar.update(n=i+1)
            pbar.close()

    def predict(self, x):
        """
            Predict the output for input data.

            This method predicts the output for the given input data by propagating
            it through the layers of the MELM model.

            Args:
            -----------
                x (array-like): Input data.

            Returns:
            -----------
                array-like: Predicted output.

            Examples:
            -----------
            Create a Multilayer ELM model

            >>> model = MELMModel()

            Add 3 layers of neurons

            >>> model.add(ELMLayer(number_neurons=1000, activation='mish', C=10))
            >>> model.add(ELMLayer(number_neurons=2000, activation='mish', C=10))
            >>> model.add(ELMLayer(number_neurons=1000, activation='mish', C=10))

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the accuracy of the model on the training data

            >>> acc = accuracy_score(model.predict(X), y)
        """
        elm_baseline = self.layers[0]
        y_hat = elm_baseline.predict(x)

        preds = []
        for layer in self.layers[1:]:
            y_pred = layer.predict(x)
            preds.append(y_pred)

        preds = tf.stack(preds, axis=0)
        preds = tf.reduce_sum(preds, axis=0)
        y_pred = y_hat + preds
        y_pred = tf.math.argmax(y_pred, axis=1)
        return y_pred.numpy()

    def predict_proba(self, x):
        """
            Predict class probabilities for input data.

            This method predicts class probabilities for the given input data
            by propagating it through the layers of the MELM model and applying softmax.

            Args:
            -----------
                x (array-like): Input data.

            Returns:
            -----------
                array-like: Predicted class probabilities.

            Examples:
            -----------
            Create a Multilayer ELM model

            >>> model = MELMModel()

            Add 3 layers of neurons

            >>> model.add(ELMLayer(number_neurons=1000, activation='mish', C=10))
            >>> model.add(ELMLayer(number_neurons=2000, activation='mish', C=10))
            >>> model.add(ELMLayer(number_neurons=1000, activation='mish', C=10))

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the prediction probability of the model on the training data

            >>> pred_proba = model.predict_proba(X)
        """
        elm_baseline = self.layers[0]
        y_hat = elm_baseline.predict_proba(x)

        preds = []
        for layer in self.layers[1:]:
            y_pred = layer.predict_proba(x)
            preds.append(y_pred)

        preds = tf.stack(preds, axis=0)

        y_pred = y_hat + preds
        y_pred = tf.math.argmax(y_pred, axis=1)
        return tf.keras.activations.softmax(y_pred).numpy()

    def summary(self):
        """
            Print a summary of the model architecture.

            This method prints a summary of the MELM model architecture,
            including information about each layer and the total number of parameters.
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
            Convert the model configuration to a dictionary.

            This method converts the configuration of the MELM model to a dictionary
            for saving or further processing.

            Returns:
            -----------
                dict: Dictionary containing the model configuration.
        """
        attributes = {
            'verbose': self.verbose
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
            Save the model to a file.

            This method saves the MELM model to a HDF5 file format.

            Args:
            -----------
                file_path (str): File path to save the model.

            Examples:
            -----------
            Save the trained model to a file

            >>> model.save("Saved Models/MELM_Model.h5")
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
            Load a model from a file.

            This method loads a MELM model from a HDF5 file.

            Args:
            -----------
                file_path (str): File path to load the model from.

            Returns:
            -----------
                MELMModel: Loaded MELM model.

            Examples:
            -----------
            Load the saved model from the file

            >>> model = model.load("Saved Models/MELM_Model.h5")
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

                model = cls(verbose=v)

                layers = get_layers(attributes)
                model.layers = layers
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None  # Return None or raise an exception based on your error-handling strategy


