from copy import deepcopy
import mealpy
import numpy as np
from mealpy import swarm_based
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf


class ELMMAOptimizer:
    """
        Metaheuristics optimizer for optimizing Extreme Learning Machine (ELM) models.

        This optimizer tunes the parameters of an ELM model using a specified fitness function.

        Parameters:
        -----------
        - model: The ELM model to be optimized.
        - fitness_function (str or callable): The fitness function to be used for optimization.
            If 'acc', it calculates the accuracy score (1 - accuracy).
            If 'mae', it calculates the mean absolute error.
            If 'rmse', it calculates the root mean squared error.
            If a custom function is provided, it should take two arrays (predictions and true labels) and return a scalar.

        Attributes:
        -----------
        - y_test (array): Test labels.
        - x_test (array): Test data.
        - y_train (array): Training labels.
        - x_train (array): Training data.
        - model (object): The ELM model instance.
        - __model_optimized (object): The optimized ELM model instance.
        - fitness_function (callable): The fitness function used for optimization.

        Examples:
        -----------
        Initialize an Extreme Learning Machine (ELM) layer

        >>> layer = ELMLayer(number_neurons=1000, activation='tanh')
        >>> model = ELMModel(layer)

        Initialize an Extreme Learning Machine (ELM) Optimizer using Metaheuristic Algorithms from mealpy package

        >>> ma = ELMMAOptimizer(model)

        Run an Extreme Learning Machine (ELM) Optimizer using Metaheuristic Algorithms from mealpy package

        >>> model2, performance = ma.optimize(X, y, 'bio_based.SMA.BaseSMA', 5, 10, verbose=0)
        >>> print(f"Fitness function best value: {performance}")
    """
    def __init__(self, model, fitness_function=None):
        self.__d = None
        self.y_test = None
        self.x_test = None
        self.y_train = None
        self.x_train = None
        self.model = model
        self.__model_optimized = None

        if fitness_function == 'acc':
            self.fitness_function = lambda x, y: 1 - accuracy_score(x, y)
        elif fitness_function == 'mae':
            self.fitness_function = lambda x, y: np.mean(np.abs(x - y))
        elif fitness_function == 'rmse':
            self.fitness_function = lambda x, y: np.sqrt(np.mean(np.square(x - y)))
        elif callable(fitness_function):
            self.fitness_function = self.fitness_function
        else:
            self.fitness_function = lambda x, y: np.mean(np.square(x - y))

    def optimize(self, x, y, method_name, epochs, pop_size, n_splits=10, max_iter=200, lb=0, ub=1, verbose=0,
                 log_file=None, n_workers=8):
        """
            Optimize the ELM model.

            Parameters:
            -----------
            - x (array): Input data.
            - y (array): Target labels.
            - method_name (str): Name of the optimization method.
            - epochs (int): Number of epochs.
            - pop_size (int): Population size.
            - n_splits (int): Number of splits for cross-validation.
            - max_iter (int): Maximum number of iterations.
            - lb (float): Lower bound for parameter values.
            - ub (float): Upper bound for parameter values.
            - verbose (int): Verbosity level.
            - log_file (str): File path for logging.
            - n_workers (int): Number of workers for parallel processing.

            Returns:
            -----------
            tuple: Optimized ELM model and best fitness value.

            Examples:
            -----------
            >>> model2, performance = ma.optimize(X, y, 'bio_based.SMA.BaseSMA', 5, 10, verbose=0)
            >>> print(f"Fitness function best value: {performance}")
        """
        model = deepcopy(self.model)
        model.random_weights = False
        self.__model_optimized = model
        model.classification = False
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/n_splits)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        d = x.shape[-1]
        self.__d = d
        dim = (d + 1) * model.layer.number_neurons
        model.layer.build(np.shape(x_train))

        if verbose == 0 and log_file is not None:
            log_to = log_file
        elif verbose == 0:
            log_to = 'None'
        else:
            log_to = 'console'

        problem = {
            "fit_func": self.__fitness_normal,
            "lb": [lb, ] * dim,
            "ub": [ub, ] * dim,
            "minmax": "min",
            'verbose': True,
            "log_to": log_to,
        }
        term = {
            "max_epoch": max_iter
        }
        opt_model = eval(f"mealpy.{method_name}(epoch={epochs}, pop_size={pop_size})")
        best_position, best_fitness_value = opt_model.solve(problem, mode="thread", n_workers=n_workers,
                                                            termination=term)

        new_alpha = best_position[:d * model.layer.number_neurons]
        new_bias = best_position[d * model.layer.number_neurons:]
        model.alpha = tf.convert_to_tensor(new_alpha)
        model.bias = tf.convert_to_tensor(new_bias)
        model.classification = True
        return model, best_fitness_value

    def __fitness_normal(self, solution=None):
        """
            Calculate fitness value for optimization.

            Parameters:
            -----------
            - solution (array): Optimization solution.

            Returns:
            -----------
            float: Fitness value.
        """
        new_alpha = solution[:self.__d * self.__model_optimized.layer.number_neurons]
        new_bias = solution[self.__d * self.__model_optimized.layer.number_neurons:]
        self.__model_optimized.alpha = tf.convert_to_tensor(new_alpha)
        self.__model_optimized.bias = tf.convert_to_tensor(new_bias)
        self.__model_optimized.fit(self.x_train, self.y_train)
        pred = self.__model_optimized.predict(self.x_test)
        pred = np.max(pred, axis=1)
        return self.fitness_function(pred, self.y_test)

