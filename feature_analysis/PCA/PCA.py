import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados
path = "../../data/data_modified.csv"
data = pd.read_csv(path)

# Dividir o conjunto de dados em features (X) e target (y)
features = data.iloc[:, :-1]
target = data.iloc[:, -1]

# standart nas features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Aplicar PCA
pca = PCA()
features_pca = pca.fit_transform(features_scaled)

# Plotar a variância por cada feature
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Variância')
plt.xlabel('Componentes')
plt.title('Variância por Componentes')
plt.show()

# Plotar o gráfico de dispersão dos dois primeiros componentes principais
plt.figure(figsize=(10, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=target, cmap='viridis', edgecolor='k', alpha=0.6)
plt.xlabel('Componente A')
plt.ylabel('Componente B')
plt.title('PCA - A vs. B')
plt.colorbar(label='Target')
plt.show()