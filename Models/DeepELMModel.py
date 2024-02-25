import h5py
import numpy as np
import tensorflow as tf
import tqdm
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

from Layers.GELM_AE_Layer import GELM_AE_Layer
from Layers.ELMLayer import ELMLayer
from Layers.WELMLayer import WELMLayer
from Layers.KELMLayer import KELMLayer
from Resources.apply_denoising import apply_denoising
from Resources.get_layers import get_layers
from Resources.labelize_images import labelize_images, unlabelize_images


class DeepELMModel(BaseEstimator, ClassifierMixin):
    """
        Deep Extreme Learning Machine (Deep ELM) Model.

        This class implements a Deep ELM model, which is a variant of the Extreme Learning Machine (ELM) that
        incorporates multiple layers. Each layer of the Deep ELM model consists of an ELM layer.

        Parameters:
        -----------
            classification (bool, optional): Whether the task is classification or regression.
                Defaults to True.
            layers (list, optional): List of ELMLayer objects representing the layers of the model.
                Defaults to None.
            verbose (int, optional): Verbosity level (0 for silent, 1 for progress bar).
                Defaults to 0.

        Attributes:
        -----------
            classes_ (array-like): Unique class labels.
            classification (bool): Indicates whether the task is classification or regression.
            layers (list): List of ELMLayer objects representing the layers of the model.
            verbose (int): Verbosity level.

        Methods:
        -----------
            add(layer): Add an ELMLayer to the model.
            fit(X, y): Fit the Deep ELM model to training data.
            predict(X): Predict class labels or regression values for input data.
            predict_proba(X): Predict class probabilities for input data.
            summary(): Print a summary of the model architecture.
            to_dict(): Convert the model to a dictionary of attributes.
            save(file_path): Serialize the model and save it to an HDF5 file.
            load(file_path): Deserialize a model instance from an HDF5 file.

        Example:
        -----------
        Initialize a ReceptiveFieldGenerator with input size (28, 28, 1) and 10 output classes

        >>> rf = ReceptiveFieldGenerator(input_size=(28, 28, 1), num_classes=10)

        Initialize a DeepELMModel

        >>> model = DeepELMModel()

        Add ELMLayers to the model with different numbers of neurons and the same receptive field generator
        The receptive field generator ensures that each layer has the same receptive field configuration

        >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))
        >>> model.add(ELMLayer(number_neurons=2000, receptive_field_generator=rf))
        >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))

        Define a cross-validation strategy

        >>> cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

        Perform cross-validation to evaluate the model performance

        >>> scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

        Print the mean accuracy score obtained from cross-validation

        >>> print(np.mean(scores))

        Fit the ELM model to the entire dataset

        >>> model.fit(X, y)

        Save the trained model to a file

        >>> model.save("Saved Models/DeepELM_Model.h5")

        Load the saved model from the file

        >>> model = model.load("Saved Models/DeepELM_Model.h5")

        Evaluate the accuracy of the model on the training data

        >>> acc = accuracy_score(model.predict(X), y)
    """
    def __init__(self, classification=True, layers=None, verbose=0):
        self.classes_ = None
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        self.classification = classification
        self.verbose = verbose

    def add(self, layer):
        """
            Add an ELMLayer to the model.

            Parameters:
            -----------
                layer (ELMLayer): An ELMLayer object representing the layer to be added.

            Example:
            -----------
            Add ELMLayers to the model with different numbers of neurons and the same receptive field generator

            >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))
        """
        self.layers.append(layer)

    def fit(self, x, y):
        """
            Fit the Deep ELM model to training data.

            Parameters:
            -----------
                x (array-like): Training input samples.
                y (array-like): Target values.

            Example:
            -----------
            Initialize a ReceptiveFieldGenerator with input size (28, 28, 1) and 10 output classes

            >>> rf = ReceptiveFieldGenerator(input_size=(28, 28, 1), num_classes=10)

            Initialize a DeepELMModel

            >>> model = DeepELMModel()

            Add ELMLayers to the model with different numbers of neurons and the same receptive field generator
            The receptive field generator ensures that each layer has the same receptive field configuration

            >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))
            >>> model.add(ELMLayer(number_neurons=2000, receptive_field_generator=rf))
            >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)
        """
        if self.classification:
            self.classes_ = unique_labels(y)
        else:
            self.classes_ = [0]

        if self.classification:
            self.classes_ = unique_labels(y)

        if self.verbose == 1:
            pbar = tqdm.tqdm(total=len(self.layers), desc='AE-ELM : AE steps')

        x = labelize_images(x, y)
        x = tf.cast(x, dtype=tf.float32)

        for layer in self.layers:
            layer.build(x.shape)

        feature_map = x
        for i, layer in enumerate(self.layers):
            if layer.denoising is None:
                layer.fit(feature_map, feature_map)
            else:
                feature_map_noised = apply_denoising(feature_map, layer.denoising, layer.denoising_param)
                layer.fit(feature_map_noised, feature_map)

            feature_map = layer.output
            if self.verbose == 1:
                pbar.update(n=i)
                i = i + 1

        if self.verbose == 1:
            pbar.update(n=i+1)
            pbar.close()

    def predict(self, x):
        """
            Predict class labels or regression values for input data.

            Parameters:
            -----------
                x (array-like): Input samples.

            Returns:
            -----------
                array-like: Predicted class labels or regression values.

            Example:
            -----------
            Initialize a ReceptiveFieldGenerator with input size (28, 28, 1) and 10 output classes

            >>> rf = ReceptiveFieldGenerator(input_size=(28, 28, 1), num_classes=10)

            Initialize a DeepELMModel

            >>> model = DeepELMModel()

            Add ELMLayers to the model with different numbers of neurons and the same receptive field generator
            The receptive field generator ensures that each layer has the same receptive field configuration

            >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))
            >>> model.add(ELMLayer(number_neurons=2000, receptive_field_generator=rf))
            >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the accuracy of the model on the training data

            >>> acc = accuracy_score(model.predict(X), y)
        """
        x = tf.cast(x, dtype=tf.float32)

        feature_map = x
        for layer in self.layers:
            feature_map = layer.predict(feature_map)

        if self.classification:
            return unlabelize_images(feature_map.numpy(), num_classes=len(self.classes_))
        else:
            return feature_map.numpy()
        pass

    def predict_proba(self, x):
        """
            Predict class probabilities for input data.

            Parameters:
            -----------
                x (array-like): Input samples.

            Returns:
            -----------
                array-like: Predicted class probabilities.

            Example:
            -----------
            Initialize a ReceptiveFieldGenerator with input size (28, 28, 1) and 10 output classes

            >>> rf = ReceptiveFieldGenerator(input_size=(28, 28, 1), num_classes=10)

            Initialize a DeepELMModel

            >>> model = DeepELMModel()

            Add ELMLayers to the model with different numbers of neurons and the same receptive field generator
            The receptive field generator ensures that each layer has the same receptive field configuration

            >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))
            >>> model.add(ELMLayer(number_neurons=2000, receptive_field_generator=rf))
            >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the accuracy of the model on the training data

            >>> pred_proba = model.predict_proba(X)
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        pred = to_categorical(pred)
        return tf.keras.activations.softmax(pred)

    def summary(self):
        """
            Print a summary of the model architecture.
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
            Convert the model to a dictionary of attributes.

            Returns:
                dict: A dictionary containing the attributes of the model.
        """
        attributes = {
            'classification': self.classification,
            'verbose': self.verbose,
            'classes': self.classes_
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

        Example:
        -----------
            Initialize a ReceptiveFieldGenerator with input size (28, 28, 1) and 10 output classes

            >>> rf = ReceptiveFieldGenerator(input_size=(28, 28, 1), num_classes=10)

            Initialize a DeepELMModel

            >>> model = DeepELMModel()

            Add ELMLayers to the model with different numbers of neurons and the same receptive field generator
            The receptive field generator ensures that each layer has the same receptive field configuration

            >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))
            >>> model.add(ELMLayer(number_neurons=2000, receptive_field_generator=rf))
            >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))

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
            Deserialize a model instance from an HDF5 file.

            Parameters:
            -----------
                file_path (str): The file path from which to load the serialized instance.

            Returns:
            -----------
                DeepELMModel: An instance of the DeepELMModel class loaded from the file.

            Example:
            -----------
            Initialize a ReceptiveFieldGenerator with input size (28, 28, 1) and 10 output classes

            >>> rf = ReceptiveFieldGenerator(input_size=(28, 28, 1), num_classes=10)

            Initialize a DeepELMModel

            >>> model = DeepELMModel()

            Add ELMLayers to the model with different numbers of neurons and the same receptive field generator
            The receptive field generator ensures that each layer has the same receptive field configuration

            >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))
            >>> model.add(ELMLayer(number_neurons=2000, receptive_field_generator=rf))
            >>> model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))

            Load the saved model from the file

            >>> model = DeepELMModel.load("Saved Models/ELM_Model.h5")
        """
        try:
            with h5py.File(file_path, 'r') as h5file:
                # Extract attributes from the HDF5 file
                attributes = {key: h5file[key][()] for key in h5file.keys()}

                for key, value in attributes.items():
                    if type(value) is bytes:
                        v = value.decode('utf-8')
                        attributes[key] = v

                c = attributes.pop('classification')
                v = attributes.pop('verbose')
                cl = attributes.pop('classes')

                model = cls(classification=c, verbose=v)

                layers = get_layers(attributes)
                model.layers = layers
                model.classes_ = cl
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None  # Return None or raise an exception based on your error-handling strategy