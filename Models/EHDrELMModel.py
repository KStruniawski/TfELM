import h5py
import tensorflow as tf
import tqdm
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from Models.DeepELMModel import apply_denoising
from Resources.get_layers import get_layers


class EHDrELMModel(BaseEstimator, ClassifierMixin):
    """
        Enhanced Deep Representation Extreme Learning Machine (EHDrELM) Model.

        This class implements an enhanced version of the Deep Representation Extreme Learning Machine model,
        incorporating weighted input connections from the output of each hidden layer to the input of the subsequent layer.

        Parameters:
        -----------
            alpha (float): Weight factor for the weighted input connections.
            layers (list): List of layers in the EHDrELM model.
            w2_weights (list): List of weights for the input connections.
            verbose (int): Verbosity mode (0 or 1).

        Attributes:
        -----------
            classes_ (array-like): Unique class labels.
            alpha (float): Weight factor for the weighted input connections.
            layers (list): List of layers in the EHDrELM model.
            w2_weights (list): List of weights for the input connections.
            verbose (int): Verbosity mode (0 or 1).

        Methods:
        -----------
            add(layer): Add a layer to the EHDrELM model.
            fit(x, y): Fit the EHDrELM model to training data.
            predict(x): Predict class labels for input data.
            predict_proba(x): Predict class probabilities for input data.
            summary(): Print a summary of the EHDrELM model.
            to_dict(): Convert the model to a dictionary of attributes.
            save(file_path): Serialize the model and save it to an HDF5 file.
            load(file_path): Deserialize an instance from an HDF5 file.

        Examples:
        -----------
        Create an instance of the Enhanced Deep Representation ELM model (EHDrELMModel)

        >>> model = EHDrELMModel()
        >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
        >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
        >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
        >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))

        Define a cross-validation strategy

        >>> cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

        Perform cross-validation to evaluate the model performance

        >>> scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

        Print the mean accuracy score obtained from cross-validation

        >>> print(np.mean(scores))

        Fit the ELM model to the entire dataset

        >>> model.fit(X, y)

        Save the trained model to a file

        >>> model.save("Saved Models/EHDrELM_Model.h5")

        Load the saved model from the file

        >>> model = model.load("Saved Models/EHDrELM_Model.h5")

        Evaluate the accuracy of the model on the training data

        >>> acc = accuracy_score(model.predict(X), y)
    """
    def __init__(self, alpha=10.0, layers=None, w2_weights=None, verbose=0):
        self.classes_ = None
        self.alpha = alpha
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        if w2_weights is None:
            self.w2_weights = []
        else:
            self.w2_weights = w2_weights

        self.verbose = verbose

    def add(self, layer):
        """
            Add a layer to the EHDrELM model.

            Parameters:
            -----------
                layer: Layer to be added to the model.

            Examples:
            -----------
            Create an instance of the Enhanced Deep Representation ELM model (EHDrELMModel)

            >>> model = EHDrELMModel()
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
        """
        self.layers.append(layer)

    def fit(self, x, y):
        """
            Fit the EHDrELM model to training data.

            Parameters:
            -----------
                x (array-like): Training input samples.
                y (array-like): Target values.

            Examples:
            -----------
            Create an instance of the Enhanced Deep Representation ELM model (EHDrELMModel)

            >>> model = EHDrELMModel()
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)
        """
        if len(y.shape) == 1:
            y = to_categorical(y)

        if self.verbose == 1:
            pbar = tqdm.tqdm(total=len(self.layers), desc='DrELM')

        for layer in self.layers:
            layer.build(x.shape)

        if self.verbose == 1:
            pbar.set_description(' DrELM : Modules training step ')

        i = 1
        X_i = x
        m, d = x.shape
        _, c = y.shape
        for layer in self.layers[:-1]:
            if layer.denoising is None:
                layer.fit(X_i, y)
            else:
                x_noised = apply_denoising(X_i, layer.denoising, layer.denoising_param)
                layer.fit(x_noised, y)
            O_i = layer.output

            W_initializer = tf.random_uniform_initializer(-1, 1)
            W2 = tf.Variable(
                W_initializer(shape=(c, d)),
                dtype=tf.float32,
                trainable=False
            )
            self.w2_weights.append(W2)
            out = x + self.alpha * tf.matmul(O_i, W2)
            X_i = X_i + layer.activation(out)
            X_i = layer.activation(X_i)

            if self.verbose == 1:
                pbar.update(n=i)
                i = i + 1

        self.layers[-1].fit(X_i, y)

        if self.verbose == 1:
            pbar.update(n=i + 1)
            pbar.close()

    def predict(self, x):
        """
            Predict class labels for input data.

            Parameters:
            -----------
                x (array-like): Input samples.

            Returns:
            -----------
                array-like: Predicted class labels.

            Examples:
            -----------
            Create an instance of the Enhanced Deep Representation ELM model (EHDrELMModel)

            >>> model = EHDrELMModel()
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the accuracy of the model on the training data

            >>> acc = accuracy_score(model.predict(X), y)
        """
        X_i = x
        for i in range(0, len(self.layers) - 1):
            O_i = self.layers[i].predict(X_i)
            out = x + self.alpha * tf.matmul(O_i, self.w2_weights[i])
            X_i = X_i + self.layers[i].activation(out)
            X_i = self.layers[i].activation(X_i)
        O_i = self.layers[-1].predict(X_i)
        return tf.argmax(O_i, axis=1).numpy()

    def predict_proba(self, x):
        """
            Predict class probabilities for input data.

            Parameters:
            -----------
                x (array-like): Input samples.

            Returns:
            -----------
                array-like: Predicted class probabilities.

            Examples:
            -----------
            Create an instance of the Enhanced Deep Representation ELM model (EHDrELMModel)

            >>> model = EHDrELMModel()
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))
            >>> model.add(ELMLayer(number_neurons=num_neurons, activation='mish', C=1.8))

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the prediction probability of the model on the training data

            >>> pred_probba = model.predict_proba(X)
        """
        X_i = x
        for i in range(0, len(self.layers) - 1):
            O_i = self.layers[i].predict_proba(X_i)
            out = x + self.alpha * tf.matmul(O_i, self.w2_weights[i])
            X_i = out * tf.math.tanh(tf.math.softplus(out))
        O_i = self.layers[-1].predict_proba(X_i)
        return tf.keras.activations.softmax(O_i).numpy()

    def summary(self):
        """
            Print a summary of the EHDrELM model.
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
            print(
                f"{layer}_{i} ({layer.__class__.__name__})         {sh}                  {layer.count_params()['all']}      ")
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
            'verbose': self.verbose,
            'w2_weights': self.w2_weights,
            'alpha': self.alpha
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
            Serialize the current instance and save it to an HDF5 file.

            Parameters:
            -----------
                file_path (str): The file path where the serialized instance will be saved.

            Examples:
            -----------
            Save the trained model to a file

            >>> model.save("Saved Models/EHDrELM_Model.h5")
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
            Deserialize an instance from an HDF5 file.

            Parameters:
            -----------
                file_path (str): The file path from which to load the serialized instance.

            Returns:
            -----------
                EHDrELMModel: An instance of the EHDrELMModel class loaded from the file.

            Examples:
            -----------
            Load the saved model from the file
            >>> model = model.load("Saved Models/EHDrELM_Model.h5")
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
                w2 = attributes.pop('w2_weights')
                a = attributes.pop('alpha')
                model = cls(verbose=v, w2_weights=w2, alpha=a)

                layers = get_layers(attributes)
                model.layers = layers
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None  # Return None or raise an exception based on your error-handling strategy