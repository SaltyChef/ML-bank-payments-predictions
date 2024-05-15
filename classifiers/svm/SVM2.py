from sklearn.model_selection import GridSearchCV
from matplotlib import colors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale, StandardScaler
from sklearn.svm import SVR, SVC
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, \
    f1_score, roc_auc_score

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100],
              'kernel': ['rbf']}

# importing or loading the dataset
dataset = pd.read_csv('data.csv', header=1)

# distributing the dataset into two components X and Y
X = dataset.iloc[:, 1:-1].values  # Selects all rows and columns from the second to the penultimate column
y = dataset.iloc[:, -1].values   # Selects all rows for the last column only

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)

grid_clf = GridSearchCV(SVC(probability=True), param_grid, scoring = 'accuracy', n_jobs = -1, verbose = 3, cv = 3)
grid_clf.fit(X_train, y_train)

optimal_SVC_clf = grid_clf.best_estimator_

print(grid_clf.best_params_)


# Get the predicted classes
test_class_preds = optimal_SVC_clf.predict(X_test)

test_accuracy_SVC = accuracy_score(test_class_preds,y_test)
test_precision_score_SVC = precision_score(test_class_preds,y_test)
test_recall_score_SVC = recall_score(test_class_preds,y_test)
test_f1_score_SVC = f1_score(test_class_preds,y_test)
test_roc_score_SVC = roc_auc_score(test_class_preds,y_test)

print("The accuracy on test data is ", test_accuracy_SVC)
print("The precision on test data is ", test_precision_score_SVC)
print("The recall on test data is ", test_recall_score_SVC)
print("The f1 on test data is ", test_f1_score_SVC)
print("The roc_score on test data is ", test_roc_score_SVC)
