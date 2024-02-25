import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from Layers.ELMLayer import ELMLayer
from Models.DeepELMModel import DeepELMModel
from Resources.ReceptiveFieldGenerator import ReceptiveFieldGenerator


n_splits = 10
n_repeats = 10

# Loading sample dataset from Data folder
path = "../Data/mnist_train.txt"
df = pd.read_csv(path, delimiter=',').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode class labels to numerical values
X = preprocessing.normalize(X)  # Normalize feature vectors

# Initialize a ReceptiveFieldGenerator with input size (28, 28, 1) and 10 output classes
rf = ReceptiveFieldGenerator(input_size=(28, 28, 1), num_classes=10)

# Initialize a DeepELMModel
model = DeepELMModel()

# Add ELMLayers to the model with different numbers of neurons and the same receptive field generator
# The receptive field generator ensures that each layer has the same receptive field configuration
model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))
model.add(ELMLayer(number_neurons=2000, receptive_field_generator=rf))
model.add(ELMLayer(number_neurons=1000, receptive_field_generator=rf))

# Define a cross-validation strategy
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

# Perform cross-validation to evaluate the model performance
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

# Print the mean accuracy score obtained from cross-validation
print(np.mean(scores))

# Fit the ELM model to the entire dataset
model.fit(X, y)

# Save the trained model to a file
model.save("Saved Models/DeepELM_Model.h5")

# Load the saved model from the file
model = model.load("Saved Models/DeepELM_Model.h5")

# Evaluate the accuracy of the model on the training data
acc = accuracy_score(model.predict(X), y)
print(acc)