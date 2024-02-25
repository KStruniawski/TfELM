import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from Layers.SSKELMLayer import SSKELMLayer
from Models.SSKELMModel import SSKELMModel
from Resources.Kernel import CombinedProductKernel, Kernel
from Resources.SSRepeatedKFold import SSRepeatedKFold
from Resources.ss_cross_val_score import ss_cross_val_score
from Resources.ss_split_dataset import ss_split_dataset

# Hyperparameters:
num_neurons = 1000
n_splits = 10
n_repeats = 10

# Loading sample dataset from Data folder
path = "../Data/G50C.txt"
df = pd.read_csv(path, delimiter='\t').fillna(0)
X = df.values[:, 1:]
y = df.values[:, 0]

# Label encoding and features normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode class labels to numerical values
X = preprocessing.normalize(X)  # Normalize feature vectors

# Splitting the dataset into labeled, validation, test, and unlabeled sets using semi-supervised split
X_labeled, X_val, X_test, X_unlabeled, y_labeled, y_val, y_test, y_unlabeled = ss_split_dataset(X, y, 50, 50, 136)

# Initialize a Kernel (it can be instanced as Kernel class and its subclasses like CombinedProductKernel)
kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])

# Initializing the Semi-Supervised Kernel Extreme Learning Machine (SS-KELM) layer
# Regularization parameter: 0.001
layer = SSKELMLayer(lam=0.01, kernel=kernel)

# Initializing the SS-KELM model with the defined layer
model = SSKELMModel(layer)

# Performing semi-supervised cross-validation using repeated k-fold
# Number of labeled, validation, test, and unlabeled samples are provided as parameters
cv = SSRepeatedKFold(n_splits=(50, 314, 50, 136), n_repeats=50)

# Scoring the model on validation and test sets using ROC AUC metric
scores_val, scores_test = ss_cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

# Printing mean scores for validation and test sets
print("Valid: " + str(np.mean(scores_val)))
print("Test: " + str(np.mean(scores_test)))

# Fitting the SS-ELM model to the labeled and unlabeled data
model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)

# Saving the trained model to a file
model.save("Saved Models/SS-KELM_Model_1.h5")

# Loading the saved model from the file
model = model.load("Saved Models/SS-KELM_Model_1.h5")

# Making predictions on the validation and test sets
pred_test = model.predict(X_test)
pred_val = model.predict(X_val)

# Printing accuracy scores on the whole dataset for validation and test sets
print(f"Valid on whole dataset: {str(accuracy_score(y_val, pred_val))}")
print(f"Test on whole dataset{str(accuracy_score(y_test, pred_test))}")
