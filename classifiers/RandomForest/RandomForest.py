import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, roc_curve, roc_auc_score

# Carregar os dados
path = "../../data/data_kruskal.csv"
dataset = pd.read_csv(path)

# Dividir o conjunto de dados em features (X) e rótulos (y)
features = dataset.iloc[:, :-1]
target = dataset.iloc[:, -1]

predictors = ['X1', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X18', 'X19']  # Change this according to your analysis

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# gini para usar as ROC
random_forest_classifier = RandomForestClassifier(random_state=42, criterion='log_loss', n_estimators=1000, verbose=False)
random_forest_classifier.fit(X_train, y_train)

predictions = random_forest_classifier.predict(X_test)

# Feature Importante
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': random_forest_classifier.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()

########### Prever os rótulos para o conjunto de teste
y_pred = random_forest_classifier.predict(X_test)

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
y_pred_proba = random_forest_classifier.predict_proba(X_test)[:,1]
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



