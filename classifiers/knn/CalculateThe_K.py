import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
np.random.seed(40)

def calculateTheK(X_train, X_test, y_train, y_test):
    error_rate = []
    for i in range(1, 50):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
    print("Minimum error:-", min(error_rate), "at K =", error_rate.index(min(error_rate)))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 50), error_rate, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()
    print("Minimum error:-", min(error_rate), "at K =", error_rate.index(min(error_rate)))


    acc = []


    for i in range(1, 50):
        neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
        yhat = neigh.predict(X_test)
        acc.append(metrics.accuracy_score(y_test, yhat))
    print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 50), acc, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='red', markersize=10)
    plt.title('accuracy vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()
    print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))

    if acc.index(max(acc)) == error_rate.index(min(error_rate)):
        return error_rate.index(min(error_rate))
    else:
        print(f"Correr again pois {error_rate.index(min(error_rate))} é diferente de {acc.index(max(acc))}")
        calculateTheK(X_train, X_test, y_train, y_test)


