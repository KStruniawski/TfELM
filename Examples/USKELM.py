import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from Layers.USKELMLayer import USKELMLayer
from Models.USKELMModel import USKELMModel
from Resources.Kernel import CombinedProductKernel, Kernel

path = "../Data/G50C.txt"
df = pd.read_csv(path, delimiter='\t').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X = preprocessing.normalize(X)
observations, features = X.shape

# 2-dim embedding
'''
kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
layer = USKELMLayer(kernel=kernel, embedding_size=2, lam=0.1)
model = USKELMModel(layer)
model.fit(X)
pred = model.predict(X)
print(pred.numpy())
print(y)

# Create a scatter plot
plt.scatter(pred[:, 0], pred[:, 1], c=y.flatten(), cmap=plt.cm.Paired, marker='o', edgecolors='k')
# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot with Colored Labels')
# Add a colorbar
plt.colorbar()
# Show the plot
plt.show()
'''

# 3-dim embedding
kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
layer = USKELMLayer(kernel=kernel, embedding_size=3, lam=0.001)
model = USKELMModel(layer=layer)
model.fit(X)

model.save("Saved Models/USKELM_Model_1.h5")
model = model.load("Saved Models/USKELM_Model_1.h5")

pred = model.predict(X)

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

'''
# Clustering
kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
layer = USKELMLayer(kernel=kernel, embedding_size=3, lam=0.001)
model = USKELMModel(layer)
model.fit(X)
pred, cluster_labels = model.predict(X, clustering=True, k=10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=cluster_labels, cmap=plt.cm.Paired, marker='o', edgecolors='k')

# Add labels and title
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('3D Scatter Plot with Colored Labels')

# Show the plot
plt.show()
'''