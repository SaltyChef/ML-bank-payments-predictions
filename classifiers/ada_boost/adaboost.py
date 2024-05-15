import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier


# importing or loading the dataset
dataset = pd.read_csv('../data/data_kruskal.csv', header=1)

# distributing the dataset into two components X and Y
X = dataset.iloc[:, :-1].values  # Selects all rows and columns from the second to the penultimate column
y = dataset.iloc[:, -1].values  # Selects all rows for the last column only

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = AdaBoostClassifier(random_state=42, algorithm='SAMME.R', learning_rate=0.8, n_estimators=100)
predictors = ['X1', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X18', 'X19']  # Change this according to your analysis

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()

cm = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm,
            xticklabels=['Not Default', 'Default'],
            yticklabels=['Not Default', 'Default'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


auc = roc_auc_score(y_test, predictions) * 100
accuracy = metrics.accuracy_score(y_test, predictions) * 100
conf_matrix = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)
recall = metrics.recall_score(y_test, predictions) * 100
precision = metrics.precision_score(y_test, predictions) * 100
f1 = metrics.f1_score(y_test, predictions) * 100
mae = metrics.mean_absolute_error(y_test, predictions)* 100
mse = metrics.mean_squared_error(y_test, predictions)* 100
rmse = metrics.mean_squared_error(y_test, predictions, squared=False)* 100

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

print("Accuracy = " + str(accuracy) + " %")
print("Specificity = " + str(specificity) + " %")
print("Recall (sensivity) = " + str(recall) + " %")
print("Precision = " + str(precision) + " %")
print("F1 score = " + str(f1) + " %")

print("Error = " + str(100 - accuracy) + " %")
print("False Zeros Rate = " + str( 100 -recall ) + " %")
print("False Ones Rate = " + str( 100 - specificity ) + " %")

