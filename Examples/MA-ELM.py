import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from Layers.ELMLayer import ELMLayer
from Models.ELMModel import ELMModel
from Optimizers.ELMMAOptimizer import ELMMAOptimizer

# Hyperparameters:
num_neurons = 1000
n_splits = 10
n_repeats = 10

# Loading sample dataset from Data folder
path = "../Data/banana.txt"
df = pd.read_csv(path, delimiter='\t').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode class labels to numerical values
X = preprocessing.normalize(X)  # Normalize feature vectors

# Initialize an Extreme Learning Machine (ELM) layer
layer = ELMLayer(number_neurons=num_neurons, activation='tanh')
model = ELMModel(layer)

# Initialize an Extreme Learning Machine (ELM) Optimizer using Metaheuristic Algorithms from mealpy package
ma = ELMMAOptimizer(model)
# Run an Extreme Learning Machine (ELM) Optimizer using Metaheuristic Algorithms from mealpy package
model2, performance = ma.optimize(X, y, 'bio_based.SMA.BaseSMA', 5, 10, verbose=0)
print(f"Fitness function best value: {performance}")

# Compare the performance
kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
acc1, acc2 = [], []
for train_index, test_index in kf.split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(x_train, y_train)
    model2.fit(x_train, y_train)
    p1 = model.predict(x_test)
    p2 = model2.predict(x_test)
    acc1.append(accuracy_score(p1, y_test))
    acc2.append(accuracy_score(p2, y_test))
print(f"Basic ELM ACC: {np.mean(acc1)}")
print(f"MA-ELM Optimized ELM ACC:{np.mean(acc2)}")

# Save the optimized model to a file
model2.save("Saved Models/MA-ELM_Model.h5")

# Load the optimized model from the file
model2 = ELMModel.load("Saved Models/MA-ELM_Model.h5")

# Evaluate the accuracy of the model on the whole dataset
acc = accuracy_score(model2.predict(X), y)
print(acc)