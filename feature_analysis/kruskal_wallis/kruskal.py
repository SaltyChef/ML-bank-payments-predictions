import pandas as pd
from scipy.stats import kruskal

# Carregar os dados
path = "../../data/data_modified.csv"
data = pd.read_csv(path)

# Dividir o conjunto de dados em features (X) e target (y)
features = data.iloc[:, :-1]
target = data.iloc[:, -1]

# Lista para armazenar os resultados do teste de Kruskal-Wallis
results = []

# Executar o teste de Kruskal-Wallis para cada feature
for feature in features.columns:
    groups = [data[feature][target == label] for label in target.unique()]
    statistic, p_value = kruskal(*groups)
    results.append((feature, statistic, p_value))

# Organizar os resultados do maior para o menor valor de estatística de teste
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

# Imprimir os resultados ordenados
for feature, statistic, p_value in results_sorted:
    print("Feature:", feature)
    print("Estatística de teste de Kruskal-Wallis:", statistic)
    print("Valor p:", p_value)
    print()
