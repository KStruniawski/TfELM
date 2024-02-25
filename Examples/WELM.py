import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from Layers.WELMLayer import WELMLayer
from Models.ELMModel import ELMModel
from Optimizers.LBFGSELMOptimizer import LBFGSELMOptimizer
from Optimizers.PGDELMOptimizer import PGDELMOptimizer

# Hyperparameters:
num_neurons = 1000
n_splits = 10
n_repeats = 10

# Loading sample dataset from Data folder
path = "../Data/banana.txt"
df = pd.read_csv(path, delimiter='\t').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode class labels to numerical values
X = preprocessing.normalize(X)  # Normalize feature vectors

# Initialize a Weighted Extreme Learning Machine (ELM) layer
layer = WELMLayer(number_neurons=num_neurons, activation='tanh', weight_method='wei-1')

# Create an ELM model using the trained ELM layer
model = ELMModel(layer)

# Define a cross-validation strategy
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

# Perform cross-validation to evaluate the model performance
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

# Print the mean accuracy score obtained from cross-validation
print(np.mean(scores))

# Fit the ELM model to the entire dataset
model.fit(X, y)

# Save the trained model to a file
model.save("Saved Models/WELM_Model.h5")

# Load the saved model from the file
model = model.load("Saved Models/WELM_Model.h5")

# Evaluate the accuracy of the model on the training data
acc = accuracy_score(model.predict(X), y)
print(acc)