import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from Layers.ELMLayer import ELMLayer
from Models.RCELMModel import RCELMModel
from Resources.rmse import calculate_rmse

# Hyperparameters:
num_neurons = 100
n_splits = 10
n_repeats = 10
layers = 5

# Loading sample dataset from Data folder
path = "../Data/abalone.txt"
df = pd.read_csv(path, delimiter='\t').fillna(0)
X = df.values[:, :-1]
y = df.values[:, -1]

# Preprocess data
X = preprocessing.normalize(X)
y = y.reshape(-1, 1)
X_test = np.array([0, 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15])
X_test = X_test.reshape(1, -1)

# Initialize a Residual Compensation Multilayer Extreme Learning Machine model
model = RCELMModel()

# Add ELM layers to the Multilayer Extreme Learning Machine
model.add(ELMLayer(number_neurons=num_neurons, activation='sigmoid', C=10))
model.add(ELMLayer(number_neurons=num_neurons, activation='sigmoid', C=10))
model.add(ELMLayer(number_neurons=num_neurons, activation='sigmoid', C=10))

# Fit the RC-MLELM model to the entire dataset
model.fit(X, y)
y_pred = model.predict(X)
print(calculate_rmse(y, y_pred))

# Save the trained model to a file
model.save('Saved Models/RC_ML_ELM_Model.h5')

# Load the saved model from the file
model = model.load('Saved Models/RC_ML_ELM_Model.h5')

# Evaluate the accuracy of the model on the training data
y_pred = model.predict(X_test)
print(calculate_rmse(15, y_pred))
