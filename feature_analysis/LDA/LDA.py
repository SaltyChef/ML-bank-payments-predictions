import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

# Carregar os dados
path = "../../data/data_modified.csv"
data = pd.read_csv(path)

# Dividir o conjunto de dados em features (X) e target (y)
features = data.iloc[:, :-1]
target = data.iloc[:, -1]

# Padronizar as features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Aplicar LDA
lda = LDA()
features_lda = lda.fit_transform(features_scaled, target)

# Verificar a variância explicada pelo LDA
explained_variance_ratio = lda.explained_variance_ratio_
print("Variância explicada pelo LDA:", explained_variance_ratio)