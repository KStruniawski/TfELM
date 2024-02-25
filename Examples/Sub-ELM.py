import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from Layers.SubELMLayer import SubELMLayer
from Models.ELMModel import ELMModel

path = "../Data/ionosphere.txt"
df = pd.read_csv(path, delimiter='\t').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X = preprocessing.normalize(X)

# Initialize a Subnetwork Extreme Learning Machine (SubELM) layer
layer = SubELMLayer(1000, 200, 70, 'mish')

# Create an SubELM model using the trained ELM layer
model = ELMModel(layer)

# Hyperparameters:
n_splits = 10
n_repeats = 10

# Define a cross-validation strategy
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

# Perform cross-validation to evaluate the model performance
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

# Print the mean accuracy score obtained from cross-validation
print(np.mean(scores))

# Fit the ELM model to the entire dataset
model.fit(X, y)

# Save the trained model to a file
model.save("Saved Models/SubELM_Model.h5")

# Load the saved model from the file
model = model.load("Saved Models/SubELM_Model.h5")

# Evaluate the accuracy of the model on the training data
acc = accuracy_score(model.predict(X), y)
print(acc)