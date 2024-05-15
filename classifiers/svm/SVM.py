from matplotlib import colors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVR, SVC
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, ConfusionMatrixDisplay

# importing or loading the dataset
dataset = pd.read_csv('data.csv', header=1)

# distributing the dataset into two components X and Y
X = dataset.iloc[:, 1:-1].values  # Selects all rows and columns from the second to the penultimate column
y = dataset.iloc[:, -1].values   # Selects all rows for the last column only
top_features = ['BILL_AMT1', 'BILL_AMT2']  # Change this according to your analysis
X_top = dataset[top_features]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pca = PCA(n_components=9)
X_train_pca = pca.fit_transform(X_train)

per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = [str(x) for x in range(1, len(per_var)+1)]

#plot scree plot
plt.bar(x=range(1, len(per_var)+1), height=per_var)
plt.tick_params(axis='x', which = 'both', bottom=False, top=False, labelbottom=False)
plt.ylabel("Explained variance (%)")
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.show()

train_pc1_coords = X_train_pca[:, 0]
train_pc2_coords = X_train_pca[:, 1]

pca_train_scaled = scale(np.column_stack((train_pc1_coords, train_pc2_coords)))


# create an SVR model with a linear kernel
model = SVC(kernel="linear")

# Training the model
model.fit(pca_train_scaled, y_train)

#calculate overall accuracy
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print(f'Accuracy: {accuracy:.2%}')

class_names = ['Did Not Default', 'Defaulted']
disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues)


X_test_pca = pca.transform(X_train)
test_pc1_coords = X_test_pca[:, 0]
test_pc2_coords = X_test_pca[:, 1]

# Sample points from the principal component space
x_min, x_max = test_pc1_coords.min() - 1, test_pc1_coords.max() + 1
y_min, y_max = test_pc2_coords.min() - 1, test_pc2_coords.max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predict the class labels for each point in the sampled space
Z = model.predict(np.column_stack((xx.ravel(), yy.ravel())))
Z = Z.reshape(xx.shape)

# Visualize the decision boundary using sampled points
fig, ax = plt.subplots(figsize=(10, 10))
ax.contourf(xx, yy, Z, alpha=0.1)
cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])
scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_train, cmap=cmap, s=100, edgecolors='k', alpha=0.7)
legend = ax.legend(scatter.legend_elements()[0], scatter.legend_elements()[1], loc='upper right')
legend.get_texts()[0].set_text('Did Not Default')
legend.get_texts()[1].set_text('Defaulted')
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_title('Visualizing the Decision Boundary Using Principal Components')
plt.show()

