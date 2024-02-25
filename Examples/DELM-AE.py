import numpy as np
import pandas as pd
from keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from tensorflow.python.keras import models

from Layers.ELMLayer import ELMLayer
from Models.ML_ELMModel import ML_ELMModel

# Loading sample dataset from Data folder
file_path = "../Data/ionosphere.txt"
df = pd.read_csv(file_path, delimiter='\t').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode class labels to numerical values
X = preprocessing.normalize(X)  # Normalize feature vectors

# Create an instance of the Multilayer ELM model as an autoencoder for feature mapping
model = ML_ELMModel(verbose=0, classification=False)

# Add layers to the autoencoder
model.add(ELMLayer(number_neurons=50))
model.add(ELMLayer(number_neurons=60))
model.add(ELMLayer(number_neurons=50))
model.add(ELMLayer(number_neurons=1000))

# Fit the autoencoder to the data
model.fit(X, y)

# Predict the encoded features
X_hat = model.predict(X)

# Create a Multilayer Perceptron (MLP) classifier
model2 = models.Sequential()
model2.add(layers.Dense(128, activation='relu', input_shape=np.shape(X_hat)))
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))

# Compile the MLP classifier
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the MLP classifier using the encoded features
model2.fit(X_hat, y, epochs=1000, validation_split=0.2, verbose=0)

# Evaluate the trained MLP classifier
test_loss, test_accuracy = model2.evaluate(X_hat, y)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

