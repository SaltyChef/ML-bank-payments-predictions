import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Carregar os dados
path = "../../data/data_modified.csv"
dataset = pd.read_csv(path)

# Dividir o conjunto de dados em features (X) e rótulos (y)
features = dataset.iloc[:, :-1]
target = dataset.iloc[:, -1]

# Calcula a média e a matriz de covariância das features 3 e 4
mean = features.iloc[:, 2:4].mean(axis=0)
covariance_matrix = np.cov(features.iloc[:, 4:6], rowvar=False)

# Cria a distribuição gaussiana bivariada com allow_singular=True
gaussian = multivariate_normal(mean=mean, cov=covariance_matrix, allow_singular=True)

# Amostra pontos da distribuição gaussiana
num_samples = 1000
samples = gaussian.rvs(size=num_samples)

# Faz o plot dos pontos amostrados
plt.figure(figsize=(10, 8))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6)
plt.xlabel('Feature 3')
plt.ylabel('Feature 4')
plt.title('Amostras da Distribuição Gaussiana Bivariada das Features 3 e 4')
plt.show()