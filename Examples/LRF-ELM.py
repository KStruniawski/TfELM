import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from Layers.ELMLayer import ELMLayer
from Models.ELMModel import ELMModel
from Models.LRFELMModel import LRFELMModel
from Resources.generate_random_filters import generate_random_filters

# Set CUDA visible devices to -1 to disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define parameters for the LRF ELM model
num_feature_maps = 48
filter_size = 4
num_input_channels = 1
pool_size = 3

# Load the MNIST dataset
path = "../Data/mnist_train.txt"
df = pd.read_csv(path, delimiter=',').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X = preprocessing.normalize(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert input data to the appropriate shape for CNN (assuming MNIST dataset)
X_train = np.reshape(X_train, (48000, 28, 28, 1))
X_test = np.reshape(X_test, (12000, 28, 28, 1))

# Initialize the ELMLayer with specified number of neurons and number of classes
layer = ELMLayer(number_neurons=5000, C=10)
# Initialize the ELMModel with the ELMLayer
elm_model = ELMModel(layer)

# Initialize the LRFELMModel with the ELMModel
model = LRFELMModel(elm_model=elm_model)

# Fit the LRFELMModel to the training data
model.fit(X_train, y_train)

# Predict the labels for the testing data
pred = model.predict(X_test)

# Print the accuracy score of the model
print(accuracy_score(pred, y_test))

# Fit the ELM model to the entire dataset
X = np.reshape(X, (60000, 28, 28, 1))
model.fit(X, y)

# Save the trained model to a file
model.save("Saved Models/LRFELM_Model.h5")

# Load the saved model from the file
model = model.load("Saved Models/LRFELM_Model.h5")

# Evaluate the accuracy of the model on the training data
acc = accuracy_score(model.predict(X), y)
print(acc)