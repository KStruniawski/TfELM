import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from Layers.GELM_AE_Layer import GELM_AE_Layer
from Layers.USELMLayer import USELMLayer
from Models.ML_ELMModel import ML_ELMModel
from Models.USELMModel import USELMModel

# Load dataset
path = "../Data/COIL20.txt"
df = pd.read_csv(path, delimiter='\t').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X = preprocessing.normalize(X)
observations, features = X.shape

# Initialize Multilayer ELM model
model = ML_ELMModel(verbose=0, classification=False)

# Add ELM autoencoder layers for unsupervised learning and feature extraction
model.add(GELM_AE_Layer(number_neurons=100))
model.add(GELM_AE_Layer(number_neurons=1000))
model.add(GELM_AE_Layer(number_neurons=1000))
model.add(GELM_AE_Layer(number_neurons=100))

# Fit the model to the data
model.fit(X)

# Obtain the embedded data from the autoencoder layers
X_new = model.predict(X)

# Initialize USELM model for final embedding
layer = USELMLayer(number_neurons=5000, embedding_size=3, lam=0.001)
model = USELMModel(layer)

# Fit the USELM model to the embedded data
model.fit(X_new)

model.save('Saved Models/US_GELM_Model.h5')
model = model.load('Saved Models/US_GELM_Model.h5')

# Obtain the final embedding
pred = model.predict(X_new)

# Plot the 3D scatter plot with colored labels
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=y.flatten(), cmap=plt.cm.Paired, marker='o', edgecolors='k')

# Add labels and title
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('3D Scatter Plot with Colored Labels')

# Show the plot
plt.show()
