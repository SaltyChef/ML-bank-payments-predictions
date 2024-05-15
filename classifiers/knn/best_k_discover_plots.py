import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

error_rate = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed',
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))

acc = []
# Will take some time

for i in range(1, 50):
    neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 50), acc, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()
print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))