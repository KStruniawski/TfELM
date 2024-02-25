from Optimizers.ELMOptimizer import ELMOptimizer
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf
from Resources.gram_schmidt import gram_schmidt


class SubELMLayer:
    """
        Subnetwork Extreme Learning Machine (SubELM) layer.

        This layer implements the SubELM algorithm, which divides the input space into subspaces and learns
        separate subnetworks for each subspace. It utilizes random feature mapping and activation functions to
        map the input data into higher-dimensional spaces, followed by the construction of subnetworks.

        Parameters:
        -----------
        - number_neurons (int): Number of neurons in the hidden layer.
        - number_subnets (int): Number of subnetworks to create.
        - neurons_subnets (int): Number of neurons in each subnetwork.
        - activation (str): Activation function to use. Default is 'tanh'.
        - act_params (dict): Parameters for the activation function. Default is None.

        Attributes:
        -----------
        - error_history (None): History of training errors.
        - feature_map (None): Feature map generated during training.
        - name (str): Name of the layer.
        - beta (None or tf.Tensor): Output weights.
        - bias (None or tf.Tensor): Bias values.
        - alpha (None or tf.Tensor): Input weights.
        - input (None or tf.Tensor): Input data.
        - output (None or tf.Tensor): Output data.
        - act_params (dict): Parameters for the activation function.
        - activation_name (str): Name of the activation function.
        - number_neurons (int): Number of neurons in the hidden layer.
        - number_subnets (int): Number of subnetworks.
        - neurons_subnets (int): Number of neurons in each subnetwork.

        Methods:
        -----------
        - build(input_shape): Build the layer.
        - fit(x, y): Fit the layer to input-output pairs.
        - predict(x): Predict output for the given input.
        - predict_proba(x): Predict class probabilities for the given input.
        - calc_output(x): Calculate the output for the given input.
        - apply_activation(x): Apply the activation function to the given input.
        - count_params(): Count the number of trainable and non-trainable parameters.
        - to_dict(): Convert the layer's attributes to a dictionary.
        - load(attributes): Load a layer instance from attributes.

        Notes:
        -----------
        - This layer divides the input space into subspaces and learns separate subnetworks for each subspace.
        - It utilizes random feature mapping and activation functions for mapping input data into higher-dimensional spaces.

        Example:
        -----------
        Initialize a Subnetwork Extreme Learning Machine (SubELM) layer

        >>> layer = SubELMLayer(1000, 200, 70, 'mish')

        Create an SubELM model using the trained ELM layer

        >>> model = ELMModel(layer)
    """
    def __init__(self,
                 number_neurons,
                 number_subnets,
                 neurons_subnets,
                 activation='tanh',
                 act_params=None,
                 **params):
        self.error_history = None
        self.feature_map = None
        self.name = "elm"
        self.beta = None
        self.bias = None
        self.alpha = None
        self.input = None
        self.output = None
        self.act_params = act_params
        self.denoising = None
        if act_params is None:
            act = ActivationFunction(1.0)
        elif "act_param" in act_params and "act_param2" in act_params:
            act = ActivationFunction(act_param=act_params["act_param"], act_param2=act_params["act_param2"])
        elif "act_param" in act_params:
            act = ActivationFunction(act_param=act_params["act_param"])
        elif "knots" in act_params:
            act = ActivationFunction(knots=act_params["knots"])
        else:
            raise Exception("TypeError: Wrong specified activation function parameters")
        self.activation_name = activation
        self.activation = eval("act." + activation)
        self.number_neurons = number_neurons
        self.number_subnets = number_subnets
        self.neurons_subnets = neurons_subnets

        if "beta" in params:
            self.beta = params.pop("beta")
        if "alpha" in params:
            self.alpha = params.pop("alpha")
        if "bias" in params:
            self.bias = params.pop("bias")
        if "A" in params:
            self.A = params.pop("A")
        if "sub_inputweights" in params:
            self.sub_inputweights = params.pop("sub_inputweights")
        if "sub_outputweights" in params:
            self.sub_outputweights = params.pop("sub_outputweights")

    def build(self, input_shape):
        """
           Build the layer.

           Parameters:
           -----------
           - input_shape (tuple): Shape of the input data.

           Returns:
           -----------
           None

           Example:
           -----------
            >>> layer = SubELMLayer(1000, 200, 70, 'mish')
            >>> layer.build(X.shape)
       """
        pass

    def fit(self, x, y):
        """
            Fit the layer to input-output pairs.

            Parameters:
            -----------
            - x (tf.Tensor): Input data.
            - y (tf.Tensor): Output data.

            Returns:
            -----------
            None

            Example:
            -----------
            >>> layer = SubELMLayer(1000, 200, 70, 'mish')
            >>> layer.build(X.shape)
            >>> layer.fit(train_data, train_targets)
        """
        alpha = x.shape[1]
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        self.input = x

        # Generate random input weights
        self.alpha = tf.random.uniform((alpha, self.number_neurons), minval=-1.0, maxval=1.0)

        # Calculate H matrix
        H = tf.matmul(x, self.alpha)
        H = self.activation(H)

        # Subnets mapping
        A = tf.random.uniform((self.number_subnets,), minval=0, maxval=self.number_neurons, dtype=tf.int32)
        self.sub_inputweights = tf.random.uniform((self.neurons_subnets, self.number_subnets), minval=-1.0, maxval=1.0)
        self.sub_outputweights = tf.random.uniform((self.number_subnets, self.neurons_subnets), minval=-1.0, maxval=1.0)

        # Initialize input_subnet
        input_subnet = tf.TensorArray(tf.float32, size=self.number_subnets)

        # Load the A-th hidden neurons
        for i in range(self.number_subnets):
            input_subnet = input_subnet.write(i, tf.gather(H, A[i], axis=0))
        input_subnet = input_subnet.stack()

        # Temporal hidden layer for the subnet
        H_subnet = tf.linalg.matmul(self.sub_inputweights, input_subnet)
        H_subnet = self.activation(H_subnet)

        # Calculate the output of the subnet
        O_subnet = tf.linalg.matmul(H_subnet, self.sub_outputweights, transpose_a=True, transpose_b=True)
        O_subnet = self.activation(O_subnet)

        # Update H
        indices = tf.expand_dims(A, axis=1)
        updates = tf.transpose(O_subnet)
        H = tf.tensor_scatter_nd_update(H, indices, updates)

        # Calculate output weights Beta
        self.beta = tf.linalg.matmul(tf.linalg.pinv(H), y)

        self.feature_map = H
        self.A = A
        self.output = tf.matmul(H, self.beta)

    def predict(self, x):
        """
           Predict output for the given input.

           Parameters:
           -----------
           - x (tf.Tensor): Input data.

           Returns:
           -----------
           tf.Tensor: Predicted output.

           Example:
           -----------
            >>> layer = SubELMLayer(1000, 200, 70, 'mish')
            >>> layer.build(X.shape)
            >>> layer.fit(train_data, train_targets)
            >>> pred = layer.predict(test_data)
       """
        x = tf.cast(x, dtype=tf.float32)
        H_test = tf.matmul(x, self.alpha)
        H_test = self.activation(H_test)
        ts_input_subnet = tf.TensorArray(tf.float32, size=self.number_subnets)
        for i in range(self.number_subnets):
            ts_input_subnet = ts_input_subnet.write(i, tf.gather(H_test, self.A[i], axis=0))
        ts_input_subnet = ts_input_subnet.stack()

        ts_H_subnet = tf.linalg.matmul(self.sub_inputweights, ts_input_subnet)
        ts_H_subnet = self.activation(ts_H_subnet)

        ts_O_subnet = tf.linalg.matmul(tf.transpose(ts_H_subnet), tf.transpose(self.sub_outputweights))
        ts_O_subnet = tf.transpose(ts_O_subnet)
        ts_O_subnet = self.activation(ts_O_subnet)

        for i in range(self.number_subnets):
            H_test = tf.tensor_scatter_nd_update(H_test, [[self.A[i]]], tf.expand_dims(ts_O_subnet[i, :], axis=0))
        return tf.matmul(H_test, self.beta)

    def predict_proba(self, x):
        """
            Predict class probabilities for the given input.

            Parameters:
            -----------
            - x (tf.Tensor): Input data.

            Returns:
            -----------
            tf.Tensor: Predicted class probabilities.

            Example:
            -----------
            >>> layer = SubELMLayer(1000, 200, 70, 'mish')
            >>> layer.build(X.shape)
            >>> layer.fit(train_data, train_targets)
            >>> pred = layer.predict_proba(test_data)
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        return tf.keras.activations.softmax(pred)

    def calc_output(self, x):
        """
            Calculate the output for the given input.

            Parameters:
            - x (tf.Tensor): Input data.

            Returns:
            tf.Tensor: Output.
        """
        x = tf.cast(x, dtype=tf.float32)
        out = self.activation(tf.matmul(x, self.beta, transpose_b=True))
        self.output = out
        return out

    def apply_activation(self, x):
        """
            Apply the activation function to the given input.

            Parameters:
            - x (tf.Tensor): Input data.

            Returns:
            tf.Tensor: Output after applying activation function.
        """
        return self.activation(x)

    def __str__(self):
        """
        Returns a string representation of the ELM layer.

        Returns:
        str: String representation.
        """
        return f"{self.name}, neurons: {self.number_neurons}"

    def count_params(self):
        """
        Counts the number of trainable and non-trainable parameters in the SubELM layer.

        Returns:
        dict: Dictionary containing counts for trainable, non-trainable, and total parameters.
        """
        if self.beta is None:
            trainable = 0
        else:
            trainable = self.beta.shape[0] * self.beta.shape[1]
        if self.alpha is None or self.bias is None:
            non_trainable = 0
        else:
            non_trainable = self.alpha.shape[0] * self.alpha.shape[1] + self.bias.shape[0]
        return {'trainable': trainable, 'non_trainable': non_trainable, 'all': trainable + non_trainable}

    def to_dict(self):
        """
            Convert the layer's attributes to a dictionary.

            Returns:
            dict: Dictionary containing the layer's attributes.
        """
        attributes = {
            'name': 'SubELMLayer',
            'number_neurons': self.number_neurons,
            'number_subnets': self.number_subnets,
            'neurons_subnets': self.neurons_subnets,
            'activation': self.activation_name,
            'act_params': self.act_params,
            "beta": self.beta,
            "alpha": self.alpha,
            "bias": self.bias,
            "A": self.A,
            "sub_outputweights": self.sub_outputweights,
            "sub_inputweights": self.sub_inputweights
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
            Load a layer instance from attributes.

            Parameters:
            - attributes (dict): Attributes to initialize the layer.

            Returns:
            SubELMLayer: Loaded layer instance.
        """
        return cls(**attributes)
