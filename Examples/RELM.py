import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from Layers.ELMLayer import ELMLayer
from Models.ELMModel import ELMModel
from Optimizers.ISTAELMOptimizer import ISTAELMOptimizer
from Optimizers.LBFGSELMOptimizer import LBFGSELMOptimizer
from Optimizers.PGDELMOptimizer import PGDELMOptimizer

# Hyperparameters:
num_neurons = 100
n_splits = 10
n_repeats = 10

# Loading sample dataset from Data folder
path = "../Data/Australian.txt"
df = pd.read_csv(path, delimiter='\t').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode class labels to numerical values
X = preprocessing.normalize(X)  # Normalize feature vectors

# Initialize optimizer (l1 norm)
optimizer = ISTAELMOptimizer(optimizer_loss='l1', optimizer_loss_reg=[0.01])
# Initialize a Regularized Extreme Learning Machine (ELM) layer with optimizer
elm = ELMLayer(number_neurons=num_neurons, activation='mish', beta_optimizer=optimizer)
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
model.save("Saved Models/RELM_Model.h5")

# Load the saved model from the file
model = model.load("Saved Models/RELM_Model.h5")

# Evaluate the accuracy of the model on the training data
acc = accuracy_score(model.predict(X), y)
print(acc)
