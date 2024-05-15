# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# importing or loading the dataset
dataset = pd.read_csv('../data.csv', header=1)

# distributing the dataset into two components X and Y
X = dataset.iloc[:, 1:-1].values  # Selects all rows and columns from the second to the penultimate column
y = dataset.iloc[:, -1].values   # Selects all rows for the last column only
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Fit Fisher Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_train, y_train)
X_lda_train = lda.transform(X_train)
X_lda_test = lda.transform(X_test)

# Calculate means for each class
mean_class_1 = np.mean(X_lda_train[y_train == 0], axis=0)
mean_class_2 = np.mean(X_lda_train[y_train == 1], axis=0)

# Calculate the pooled covariance matrix
covariance_matrix = np.cov(X_lda_train.T, aweights=(y_train == 0) * 1 + (y_train == 1) * 1)


# Calculate the minimum distance classifier parameters
w = mean_class_1 - mean_class_2
b = -0.5 * (np.dot(mean_class_1, mean_class_1) - np.dot(mean_class_2, mean_class_2))

# Calculate decision boundaries
x_min, x_max = X_lda_train.min() - 1, X_lda_train.max() + 1
xx = np.linspace(x_min, x_max, 1000)
decision_boundary = (-b - w * xx) / w

# Calculate its inverse
#covariance_matrix_inv = np.linalg.inv(covariance_matrix)

# Define a function to compute Mahalanobis distance
def mahalanobis_distance(x, mean, cov_inv):
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))

# Calculate Mahalanobis distances for the test set to each class mean
dist_class_1_test = np.array([mahalanobis_distance(x, mean_class_1, covariance_matrix) for x in X_lda_test])
dist_class_2_test = np.array([mahalanobis_distance(x, mean_class_2, covariance_matrix) for x in X_lda_test])

# Assign class labels based on minimum Mahalanobis distance
predictions = np.where(dist_class_1_test < dist_class_2_test, 0, 1)

# Plotting test instances and their predicted classes
plt.figure(figsize=(10, 6))
colors = ['r' if label == 0 else 'b' for label in predictions]
plt.plot(xx, decision_boundary, label='Decision Boundary')
plt.scatter(X_lda_test, X_lda_test, label='Test Instances', marker='^', edgecolors='k', c=colors)
plt.scatter(mean_class_1, mean_class_1, label='Mean Class 1', marker='x', c='y')  # Mean point of Class 1
plt.scatter(mean_class_2, mean_class_2, label='Mean Class 2', marker='x', c='w')  # Mean point of Class 2
plt.xlabel('Fisher Discriminant Direction')
plt.ylabel('Arbitrary Axis')
plt.title('Minimum Distance Classifier with Fisher Discriminant Analysis (Test) using Mahalanobis Distance')
plt.legend()
plt.show()



confusion_matrix = metrics.confusion_matrix(y_test, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Zeros", "Ones"])
cm_display.plot()
plt.show()


accuracy =  metrics.accuracy_score(y_test, predictions) * 100
specificity = metrics.roc_auc_score(y_test, predictions) * 100
recall = metrics.recall_score(y_test, predictions) * 100
precision = metrics.precision_score(y_test, predictions) * 100
f1 = metrics.f1_score(y_test, predictions) * 100

print("Accuracy = " + str(accuracy) + " %")
print("Specificity = " + str(specificity) + " %")
print("Recall (sensivity) = " + str(recall) + " %")
print("Precision = " + str(precision) + " %")
print("F1 score = " + str(f1) + " %")

print("Error = " + str(100 - accuracy) + " %")
print("False Zeros Rate = " + str( 100 -recall ) + " %")
print("False Ones Rate = " + str( 100 - specificity ) + " %")


