import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from Layers.KELMLayer import KELMLayer
from Models.KELMModel import KELMModel
from Resources.Kernel import Kernel, CombinedSumKernel, CombinedProductKernel

# Hyperparameters:
num_neurons = 100
n_splits = 10
n_repeats = 10

# Loading sample dataset from Data folder
path = "../Data/ionosphere.txt"
df = pd.read_csv(path, delimiter='\t').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode class labels to numerical values
X = preprocessing.normalize(X)  # Normalize feature vectors

# Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)
kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

# Initialize a Kernel Extreme Learning Machine (KELM) layer
layer = KELMLayer(kernel, 'mish')

# Initialize a Kernel Extreme Learning Machine (KELM) model
model = KELMModel(layer)

# Define a cross-validation strategy
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

# Perform cross-validation to evaluate the model performance
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

# Print the mean accuracy score obtained from cross-validation
print(np.mean(scores))

# Fit the ELM model to the entire dataset
model.fit(X, y)

# Save the trained model to a file
model.save("Saved Models/KELM_Model.h5")

# Load the saved model from the file
model = model.load("Saved Models/KELM_Model.h5")

# Evaluate the accuracy of the model on the training data
acc = accuracy_score(model.predict(X), y)
print(acc)