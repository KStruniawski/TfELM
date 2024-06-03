import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from Layers.ELMLayer import ELMLayer
from Models.ELMModel import ELMModel


# Hyperparameters:
num_neurons = 1000
n_splits = 10
n_repeats = 50

# Loading sample dataset from Data folder
path = "../Data/ionosphere.txt"
df = pd.read_csv(path, delimiter='\t').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode class labels to numerical values
X = preprocessing.normalize(X)  # Normalize feature vectors


# Fuzzify the imput
def fuzzify_input(X):
    # Define triangular membership functions for fuzzification
    # Here, we use a simple approach where we assume that the features are normalized between 0 and 1
    fuzzified_X = np.zeros_like(X)
    for i in range(X.shape[1]):
        fuzzified_X[:, i] = np.maximum(0, 1 - np.abs(X[:, i] - 0.5) / 0.5)
    return fuzzified_X


X = fuzzify_input(X)

# Initialize an Extreme Learning Machine (ELM) layer
elm = ELMLayer(number_neurons=num_neurons, activation='mish')

# Create an ELM model using the trained ELM layer
model = ELMModel(elm)

# Define a cross-validation strategy
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

# Perform cross-validation to evaluate the model performance
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

# Print the mean accuracy score obtained from cross-validation
print(np.mean(scores))

# Fit the ELM model to the entire dataset
model.fit(X, y)

# Save the trained model to a file
model.save("Saved Models/FELM_Model.h5")

# Load the saved model from the file
model = model.load("Saved Models/FELM_Model.h5")

# Evaluate the accuracy of the model on the training data
acc = accuracy_score(model.predict(X), y)
print(acc)
