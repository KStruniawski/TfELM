import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from Layers.OSELMLayer import OSELMLayer
from Models.OSELMModel import OSELMModel
import os

# (optional) To disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path = "../Data/ionosphere.txt"
df = pd.read_csv(path, delimiter='\t').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode class labels to numerical values
X = preprocessing.normalize(X)  # Normalize feature vectors

# Hyperparameters:
batch_size = 1000
num_neurons = 100
prefetch_size = 2000
n_splits = 10
n_repeats = 10

# Initialize OSELMLayer with specified parameters
layer = OSELMLayer(num_neurons, 'tanh')

# Initialize OSELMModel with the OSELMLayer and other parameters
model = OSELMModel(layer, prefetch_size=prefetch_size, batch_size=batch_size, verbose=0)

# Perform cross-validation
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

# Print mean accuracy score
print(np.mean(scores))

# Fit the ML-ELM model to the entire dataset
model.fit(X, y)

# Save the trained model to a file
model.save('Saved Models/OSELM_Model.h5')

# Load the saved model from the file
model = model.load('Saved Models/OSELM_Model.h5')

# Evaluate the accuracy of the model on the training data
acc = accuracy_score(model.predict(X), y)
print(acc)