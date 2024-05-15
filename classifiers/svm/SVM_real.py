#Load the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import colors
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

def metricas (clf_svm,X_test_scaled,y_test, y_pred):
    auc = roc_auc_score(y_test, y_pred) * 100
    accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) * 100
    recall = metrics.recall_score(y_test, y_pred) * 100
    precision = metrics.precision_score(y_test, y_pred) * 100
    f1 = metrics.f1_score(y_test, y_pred) * 100
    mae = metrics.mean_absolute_error(y_test, y_pred) * 100
    mse = metrics.mean_squared_error(y_test, y_pred) * 100
    rmse = metrics.mean_squared_error(y_test, y_pred, squared=False) * 100

    class_names = ['Did Not Default', 'Defaulted']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print("Accuracy = " + str(accuracy) + " %")
    print("Specificity = " + str(specificity) + " %")
    print("Recall (sensivity) = " + str(recall) + " %")
    print("Precision = " + str(precision) + " %")
    print("F1 score = " + str(f1) + " %")
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print("AUC = " + str(auc) + " %")
    print("--------------------------------------------------------------------------")



    # Prever os rótulos para o conjunto de teste
    y_pred_proba = clf_svm.predict_proba(X_test_scaled)[:,1]
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



# importing or loading the dataset
dataset = pd.read_csv('../../data/data_modified.csv', header=1)

# distributing the dataset into two components X and Y
features = dataset.iloc[:, :-1].values  # Selects all rows and columns from the second to the penultimate column
target = dataset.iloc[:, -1].values   # Selects all rows for the last column only


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#scale the data
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)


'''
####################### DEFAULT #######################
clf_svm = SVC(random_state = 42, probability=True)
clf_svm.fit(X_train_scaled, y_train)

#calculate overall accuracy
y_pred = clf_svm.predict(X_test_scaled)

metricas(clf_svm, X_test_scaled,y_test, y_pred)


####################### GRID #######################
print("-----------------------Escolher agora os melhores parametros com maior accuracy-------------------------------------")

param_grid = {'C':[0.5,0.1,1,10,100,1000],
              'gamma':['scale', 1,0.1, 0.01,0.001,0.0001],
              'kernel':['rbf', 'sigmoid']}

optimal_params = GridSearchCV(SVC(), param_grid, cv = 5, scoring='accuracy', verbose=3)
optimal_params.fit(X_train_scaled, y_train)

# see "best" parameters
print(optimal_params.best_params_)
# Get the best parameters from grid search
best_C = optimal_params.best_params_['C']
best_gamma = optimal_params.best_params_['gamma']
best_kernel = optimal_params.best_params_['kernel']
# refit model with optimal hyperparameters
grid_predictions = optimal_params.predict(X_test)


clf_svm = SVC(random_state = 42, C=1000, gamma=0.001, kernel= 'rbf', probability=True)
clf_svm.fit(X_train_scaled, y_train)

y_pred = clf_svm.predict(X_test_scaled)
metricas(clf_svm, X_test_scaled,y_test, y_pred)
'''

print("----------------------------Meter PCA em pratica para diminuir a dimensionalidade-----------------------------------")
pca = PCA(n_components = 9)
X_train_pca = pca.fit_transform(X_train_scaled)

train_pc1_coords = X_train_pca[:, 0]
train_pc2_coords = X_train_pca[:, 1]

pca_train_scaled = scale(np.column_stack((train_pc1_coords, train_pc2_coords)))

clf_svm = SVC(random_state=42, C=1000, gamma=0.001, kernel= 'rbf', probability=True)
clf_svm.fit(pca_train_scaled, y_train)

X_test_pca = pca.transform(X_train_scaled)
test_pc1_coords = X_test_pca[:, 0]
test_pc2_coords = X_test_pca[:, 1]

x_min = test_pc1_coords.min()-1
x_max = test_pc1_coords.max()+1
y_min = test_pc2_coords.min()-1
y_max = test_pc2_coords.max()+1

xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),np.arange(start=y_min, stop=y_max, step=0.1) )

Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))
Z = Z.reshape(xx.shape)

# visualizing the data
fig, ax = plt.subplots(figsize=(10,10))
ax.contourf(xx,yy, Z, alpha=0.1)
cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])
scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_train, cmap=cmap, s=100, edgecolors='k', alpha=0.7)
legend = ax.legend(scatter.legend_elements()[0], scatter.legend_elements()[1], loc='upper right')
legend.get_texts()[0].set_text('Did Not Default')
legend.get_texts()[1].set_text('Defaulted')
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_title('Visualizing the Decision Boundary Using Principal Components')
plt.show()

y_pred = clf_svm.predict(X_test_pca)
metricas(clf_svm, X_test_pca, y_test, y_pred)



