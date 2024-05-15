import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from CalculateThe_K import calculateTheK
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, roc_curve, roc_auc_score

# Carregar os dados
path = "../../data/data_kruskal.csv"
dataset = pd.read_csv(path)

# Dividir o conjunto de dados em features (X) e rótulos (y)
features = dataset.iloc[:, :-1]
target = dataset.iloc[:, -1]

top_features = ['X1', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X18', 'X19']  # Change this according to your analysis
X_top = dataset[top_features]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_top, target, test_size=0.2)
k = calculateTheK(X_train, X_test, y_train, y_test)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

########### Prever os rótulos para o conjunto de teste
y_pred = knn.predict(X_test)

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
y_pred_proba = knn.predict_proba(X_test)[:,1]
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

