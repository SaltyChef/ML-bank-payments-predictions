# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import euclidean_distances
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, roc_curve, roc_auc_score

path = "../../data/data_modified.csv"
dataset = pd.read_csv(path)

# Dividir o conjunto de dados em features (X) e rótulos (y)
features = dataset.iloc[:, :-1]
target = dataset.iloc[:, -1]

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Fit Fisher Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
X_lda_train = lda.transform(X_train)
X_lda_test = lda.transform(X_test)

# Calculate means for each class
mean_class_1 = np.mean(X_lda_train[y_train == 0], axis=0)
mean_class_2 = np.mean(X_lda_train[y_train == 1], axis=0)

# Calculate the minimum distance classifier parameters
w = mean_class_1 - mean_class_2
b = -0.5 * (np.dot(mean_class_1, mean_class_1) - np.dot(mean_class_2, mean_class_2))

# Calculate decision boundaries
x_min, x_max = X_lda_train.min() - 1, X_lda_train.max() + 1
xx = np.linspace(x_min, x_max, 1000)
decision_boundary = (-b - w * xx) / w

# Use euclidean_distances to calculate distances for test set from the medium point
dist_class_1_test = euclidean_distances(X_lda_test, [mean_class_1])
dist_class_2_test = euclidean_distances(X_lda_test, [mean_class_2])

# Assign class labels based on minimum euclidian distance
predictions = np.where(dist_class_1_test < dist_class_2_test, 0, 1)

# Plotting
plt.figure(figsize=(10, 6))
colors = ['r' if label == 0 else 'b' for label in predictions]
plt.plot(xx, decision_boundary, label='Decision Boundary')
plt.scatter(X_lda_test, np.zeros_like(X_lda_test), label='Test Instances', marker='^', edgecolors='k', c=colors)
plt.scatter(mean_class_1, np.zeros_like(mean_class_1), label='Mean Class 1', marker='x', c='y')  # Plotting the mean point of Class 1
plt.scatter(mean_class_2, np.zeros_like(mean_class_2), label='Mean Class 2', marker='x', c='r')  # Plotting the mean point of Class 2
plt.xlabel('Fisher Discriminant Direction')
plt.ylabel('Arbitrary Axis')
plt.title('Minimum Distance Classifier with Fisher Discriminant Analysis (Test)')
plt.legend()
plt.show()


########### Prever os rótulos para o conjunto de teste
y_pred = lda.predict(X_test)

# Calcular as métricas de avaliação
accuracy = accuracy_score(y_test, y_pred) * 100
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp) * 100
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100
mse = mean_squared_error(y_test, y_pred) * 100
rmse = mean_squared_error(y_test, y_pred, squared=False) * 100
mae = mean_absolute_error(y_test, y_pred) * 100

# Imprimir as métricas
print(f"Accuracy: {accuracy}")
print(f"Specificity: {specificity}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()

# Prever os rótulos para o conjunto de teste
y_pred_proba = lda.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_value = roc_auc_score(y_test, y_pred_proba)

# Plotar a curva ROC com AUC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (ROC) Curve (AUC = {auc_value:.2f})')
plt.legend()
plt.show()

# Calculate F1 score for different threshold values
thresholds = np.arange(0, 1.01, 0.01)
f1_scores = [f1_score(y_test, y_pred_proba >= threshold) for threshold in thresholds]

# Plot F1 score against thresholds
plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1_scores, color='blue')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Threshold')
plt.grid(True)
plt.show()

