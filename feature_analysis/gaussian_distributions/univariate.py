import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados
path = "../../data/data_kruskal.csv"
data = pd.read_csv(path)

# Exibir a distribuição gaussiana univariada e plotar os gráficos correspondentes
for column in data.columns:
    sns.displot(data[column], kde=True)
    plt.title(f'Distribuição Gaussiana de {column}')
    plt.xlabel(column)
    plt.ylabel('Densidade')
    plt.show()