import h5py
import tensorflow as tf
import tqdm
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

from Layers import USELMLayer
from Layers.GELM_AE_Layer import GELM_AE_Layer
from Layers.KELMLayer import KELMLayer
from Layers.ELMLayer import ELMLayer
from Resources.apply_denoising import apply_denoising
from Resources.get_layers import get_layers


class ML_ELMModel(BaseEstimator, ClassifierMixin):
    """
        Multilayer Extreme Learning Machine (ELM) model.

        This model consists of multiple layers of ELM units for feature extraction followed by a final ELM layer for classification.
        Can accept various types of ELM layer like KELM, GELM, SubELM

        Parameters:
        -----------
        - classification (bool): Whether the model is for classification (default True).
        - layers (list): List of ELM layers.
        - verbose (int): Verbosity level (default 0).

        Attributes:
        -----------
        - classes_ (array): Unique classes in the target data.
        - layers (list): List of ELM layers in the model.
        - classification (bool): Whether the model is for classification.
        - verbose (int): Verbosity level.

        Examples:
        -----------
        Initialize a Multilayer Extreme Learning Machine model

        >>> model = ML_ELMModel(verbose=1)

        Add ELM layers to the Multilayer Extreme Learning Machine

        >>> model.add(ELMLayer(number_neurons=50))
        >>> model.add(ELMLayer(number_neurons=60))
        >>> model.add(ELMLayer(number_neurons=50))
        >>> model.add(ELMLayer(number_neurons=1000))

        (optional) can also accept other types of layers
        Add KELM layers to the Multilayer Extreme Learning Machine
        Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)
        >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

        >>> model.add(KELMLayer(kernel=kernel))
        >>> model.add(KELMLayer(kernel=kernel))
        >>> model.add(KELMLayer(kernel=kernel))
        >>> model.add(ELMLayer(number_neurons=1000))

        Add D-ELM layers to the Multilayer Extreme Learning Machine with denoising mechanism

        >>> model.add(ELMLayer(number_neurons=50, denoising='sp', denoising_param=0.08))
        >>> model.add(ELMLayer(number_neurons=60, denoising='sp', denoising_param=0.08))
        >>> model.add(ELMLayer(number_neurons=50, denoising='sp', denoising_param=0.08))
        >>> model.add(ELMLayer(number_neurons=1000, denoising='sp', denoising_param=0.08))

        Add ELM autoencoder layers for unsupervised learning and feature extraction

        >>> model.add(GELM_AE_Layer(number_neurons=100))
        >>> model.add(GELM_AE_Layer(number_neurons=1000))
        >>> model.add(GELM_AE_Layer(number_neurons=1000))
        >>> model.add(GELM_AE_Layer(number_neurons=100))


        Define a cross-validation strategy

        >>> n_splits = 10
        >>> n_repeats = 10

        >>> cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

        Perform cross-validation to evaluate the model performance

        >>> scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

        Print the mean accuracy score obtained from cross-validation

        >>> print(np.mean(scores))

        Fit the ML-ELM model to the entire dataset

        >>> model.fit(X, y)

        Save the trained model to a file

        >>> model.save('Saved Models/ML_ELM_Model.h5')

        Load the saved model from the file

        >>> model = model.load('Saved Models/ML_ELM_Model.h5')

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
            Add an ELM layer to the model.

            Parameters:
            -----------
            - layer: The ELM layer to be added.

            Examples:
            -----------
            Initialize a Multilayer Extreme Learning Machine model

            >>> model = ML_ELMModel(verbose=1)

            Add ELM layers to the Multilayer Extreme Learning Machine

            >>> model.add(ELMLayer(number_neurons=50))
            >>> model.add(ELMLayer(number_neurons=60))
            >>> model.add(ELMLayer(number_neurons=50))
            >>> model.add(ELMLayer(number_neurons=1000))

            (optional) can also accept other types of layers
            Add KELM layers to the Multilayer Extreme Learning Machine
            Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)
            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

            >>> model.add(KELMLayer(kernel=kernel))
            >>> model.add(KELMLayer(kernel=kernel))
            >>> model.add(KELMLayer(kernel=kernel))
            >>> model.add(ELMLayer(number_neurons=1000))

            Add D-ELM layers to the Multilayer Extreme Learning Machine with denoising mechanism

            >>> model.add(ELMLayer(number_neurons=50, denoising='sp', denoising_param=0.08))
            >>> model.add(ELMLayer(number_neurons=60, denoising='sp', denoising_param=0.08))
            >>> model.add(ELMLayer(number_neurons=50, denoising='sp', denoising_param=0.08))
            >>> model.add(ELMLayer(number_neurons=1000, denoising='sp', denoising_param=0.08))

            Add ELM autoencoder layers for unsupervised learning and feature extraction

            >>> model.add(GELM_AE_Layer(number_neurons=100))
            >>> model.add(GELM_AE_Layer(number_neurons=1000))
            >>> model.add(GELM_AE_Layer(number_neurons=1000))
            >>> model.add(GELM_AE_Layer(number_neurons=100))
        """
        self.layers.append(layer)

    def fit(self, x, y=None):
        """
            Fit the Multilayer ELM model to the given data.

            Parameters:
            -----------
            - x (array): Input data.
            - y (array): Target labels (for classification) or None (for feature extraction).

            Examples:
            -----------
            Initialize a Multilayer Extreme Learning Machine model

            >>> model = ML_ELMModel(verbose=1)

            Add ELM layers to the Multilayer Extreme Learning Machine

            >>> model.add(ELMLayer(number_neurons=50))
            >>> model.add(ELMLayer(number_neurons=60))
            >>> model.add(ELMLayer(number_neurons=50))
            >>> model.add(ELMLayer(number_neurons=1000))

            (optional) can also accept other types of layers
            Add KELM layers to the Multilayer Extreme Learning Machine
            Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)
            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

            >>> model.add(KELMLayer(kernel=kernel))
            >>> model.add(KELMLayer(kernel=kernel))
            >>> model.add(KELMLayer(kernel=kernel))
            >>> model.add(ELMLayer(number_neurons=1000))

            Add D-ELM layers to the Multilayer Extreme Learning Machine with denoising mechanism

            >>> model.add(ELMLayer(number_neurons=50, denoising='sp', denoising_param=0.08))
            >>> model.add(ELMLayer(number_neurons=60, denoising='sp', denoising_param=0.08))
            >>> model.add(ELMLayer(number_neurons=50, denoising='sp', denoising_param=0.08))
            >>> model.add(ELMLayer(number_neurons=1000, denoising='sp', denoising_param=0.08))

            Add ELM autoencoder layers for unsupervised learning and feature extraction

            >>> model.add(GELM_AE_Layer(number_neurons=100))
            >>> model.add(GELM_AE_Layer(number_neurons=1000))
            >>> model.add(GELM_AE_Layer(number_neurons=1000))
            >>> model.add(GELM_AE_Layer(number_neurons=100))

            Fit the ML-ELM model to the entire dataset

            >>> model.fit(X, y)

            Save the trained model to a file

            >>> model.save('Saved Models/ML_ELM_Model.h5')

            Load the saved model from the file

            >>> model = model.load('Saved Models/ML_ELM_Model.h5')

            Evaluate the accuracy of the model on the training data

            >>> acc = accuracy_score(model.predict(X), y)
        """
        observations, features = x.shape

        if self.classification:
            self.classes_ = unique_labels(y)
            y = to_categorical(y)
        else:
            y = x

        if self.verbose == 1:
            pbar = tqdm.tqdm(total=len(self.layers), desc='AE-ELM : AE steps')

        self.layers[0].build(input_shape=x.shape)
        prev = self.layers[0]
        for layer in self.layers[1:]:
            if type(prev) is KELMLayer:
                layer.build(input_shape=(observations, features))
            else:
                layer.build(input_shape=(observations, prev.number_neurons))
            prev = layer

        feature_map = x
        for i, layer in enumerate(self.layers[:-1]):
            if layer.denoising is None:
                if isinstance(layer, GELM_AE_Layer):
                    layer.fit(feature_map)
                else:
                    layer.fit(feature_map, feature_map)
            else:
                feature_map_noised = apply_denoising(feature_map, layer.denoising, layer.denoising_param)
                if isinstance(layer, GELM_AE_Layer):
                    layer.fit(feature_map_noised)
                else:
                    layer.fit(feature_map_noised, feature_map)
            feature_map = layer.calc_output(feature_map)
            if self.verbose == 1:
                pbar.update(n=i)
                i = i + 1
        if self.verbose == 1:
            pbar.set_description('AE-ELM : ELM step')

        if isinstance(self.layers[-1], GELM_AE_Layer):
            self.layers[-1].fit(feature_map)
        else:
            self.layers[-1].fit(feature_map, y)

        if self.verbose == 1:
            pbar.update(n=i+1)
            pbar.close()

    def predict(self, x):
        """
            Predict the labels for the input data.

            Parameters:
            -----------
            - x (array): Input data.

            Returns:
            -----------
            array: Predicted labels.

                        Examples:
            -----------
            Initialize a Multilayer Extreme Learning Machine model

            >>> model = ML_ELMModel(verbose=1)

            Add ELM layers to the Multilayer Extreme Learning Machine

            >>> model.add(ELMLayer(number_neurons=50))
            >>> model.add(ELMLayer(number_neurons=60))
            >>> model.add(ELMLayer(number_neurons=50))
            >>> model.add(ELMLayer(number_neurons=1000))

            (optional) can also accept other types of layers
            Add KELM layers to the Multilayer Extreme Learning Machine
            Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)
            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

            >>> model.add(KELMLayer(kernel=kernel))
            >>> model.add(KELMLayer(kernel=kernel))
            >>> model.add(KELMLayer(kernel=kernel))
            >>> model.add(ELMLayer(number_neurons=1000))

            Add D-ELM layers to the Multilayer Extreme Learning Machine with denoising mechanism

            >>> model.add(ELMLayer(number_neurons=50, denoising='sp', denoising_param=0.08))
            >>> model.add(ELMLayer(number_neurons=60, denoising='sp', denoising_param=0.08))
            >>> model.add(ELMLayer(number_neurons=50, denoising='sp', denoising_param=0.08))
            >>> model.add(ELMLayer(number_neurons=1000, denoising='sp', denoising_param=0.08))

            Add ELM autoencoder layers for unsupervised learning and feature extraction

            >>> model.add(GELM_AE_Layer(number_neurons=100))
            >>> model.add(GELM_AE_Layer(number_neurons=1000))
            >>> model.add(GELM_AE_Layer(number_neurons=1000))
            >>> model.add(GELM_AE_Layer(number_neurons=100))

            Fit the ML-ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the accuracy of the model on the training data

            >>> acc = accuracy_score(model.predict(X), y)
        """
        x = tf.cast(x, dtype=tf.float32)

        feature_map = x
        for layer in self.layers[:-1]:
            feature_map = layer.calc_output(feature_map)
        feature_map = self.layers[-1].predict(feature_map)

        if self.classification:
            return tf.math.argmax(feature_map, axis=1).numpy()
        else:
            return feature_map.numpy()
        pass

    def predict_proba(self, x):
        """
            Predict the probabilities for each class for the input data.

            Parameters:
            -----------
            - x (array): Input data.

            Returns:
            -----------
            array: Predicted probabilities.

            Examples:
            -----------
            Initialize a Multilayer Extreme Learning Machine model

            >>> model = ML_ELMModel(verbose=1)

            Add ELM layers to the Multilayer Extreme Learning Machine

            >>> model.add(ELMLayer(number_neurons=50))
            >>> model.add(ELMLayer(number_neurons=60))
            >>> model.add(ELMLayer(number_neurons=50))
            >>> model.add(ELMLayer(number_neurons=1000))

            (optional) can also accept other types of layers
            Add KELM layers to the Multilayer Extreme Learning Machine
            Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)
            >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

            >>> model.add(KELMLayer(kernel=kernel))
            >>> model.add(KELMLayer(kernel=kernel))
            >>> model.add(KELMLayer(kernel=kernel))
            >>> model.add(ELMLayer(number_neurons=1000))

            Add D-ELM layers to the Multilayer Extreme Learning Machine with denoising mechanism

            >>> model.add(ELMLayer(number_neurons=50, denoising='sp', denoising_param=0.08))
            >>> model.add(ELMLayer(number_neurons=60, denoising='sp', denoising_param=0.08))
            >>> model.add(ELMLayer(number_neurons=50, denoising='sp', denoising_param=0.08))
            >>> model.add(ELMLayer(number_neurons=1000, denoising='sp', denoising_param=0.08))

            Add ELM autoencoder layers for unsupervised learning and feature extraction

            >>> model.add(GELM_AE_Layer(number_neurons=100))
            >>> model.add(GELM_AE_Layer(number_neurons=1000))
            >>> model.add(GELM_AE_Layer(number_neurons=1000))
            >>> model.add(GELM_AE_Layer(number_neurons=100))

            Fit the ML-ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the prediction probability of the model on the training data

            >>> pred_proba = model.predict_proba(X)
        """
        x = tf.cast(x, dtype=tf.float32)
        feature_map = x
        for layer in self.layers[:-1]:
            feature_map = layer.calc_output(feature_map)
        pred_prob = self.layers[-1].predict_proba(feature_map)
        return pred_prob

    def summary(self):
        """Print a summary of the model architecture and parameters."""
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
            Convert the model and its layers to a dictionary.

            Returns:
            -----------
            dict: A dictionary representation of the model and its layers.
        """
        attributes = {
            'classification': self.classification,
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

            >>> model.save('Saved Models/ML_ELM_Model.h5')
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
            ML_ELMModel: An instance of the ML_ELMModel class loaded from the file.

            Examples:
            -----------
            Load the saved model from the file

            >>> model = model.load('Saved Models/ML_ELM_Model.h5')
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

                model = cls(classification=c, verbose=v)

                layers = get_layers(attributes)
                model.layers = layers
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None  # Return None or raise an exception based on your error-handling strategy