import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import os

"""
1. Desempenho do modelo: Quanto mais a curva ROC se aproxima do canto superior esquerdo 
do gráfico e maior for a AUC, melhor é o desempenho do modelo.

2. Trade-off entre TPR e FPR: A curva ROC permite visualizar o trade-off entre a taxa 
de verdadeiros positivos e a taxa de falsos positivos. À medida que aumentamos o TPR 
(detectamos mais verdadeiros positivos), normalmente também aumentamos o FPR (o número de falsos positivos).

3. Comparação entre diferentes features: Se estivermos comparando as curvas ROC de 
diferentes features, podemos determinar quais features têm um melhor poder de discriminação 
para a tarefa de classificação.
Portanto, analisando o gráfico, podemos avaliar o desempenho do modelo de classificação para 
cada feature individualmente e identificar quais features têm um maior impacto na capacidade 
de distinguir entre as classes positivas e negativas.
"""
class ROC:
    def __init__(self, y_test, y_score, path_for_graph=None):
        # Calcular a curva ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        # Plotar a curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC - Dataset Inteiro')
        plt.legend(loc="lower right")

        if path_for_graph:
            # Verificar se o diretório existe, senão, criar
            os.makedirs(os.path.dirname(path_for_graph), exist_ok=True)
            # Salvar o gráfico no caminho especificado
            plt.savefig(path_for_graph)
        else:
            plt.show()

#EXEMPLO
if __name__ == "__main__":
    path = "../../data/data_modified.csv"
    data = pd.read_csv(path)

    # Dividir o conjunto de dados em features (X) e rótulos (y)
    features = data.iloc[:, :-1]
    target = data.iloc[:, -1]

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Treinar um modelo (aqui estou usando uma regressão logística, você pode usar o seu modelo)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Obter as probabilidades previstas para os dados de teste
    y_score = model.predict_proba(X_test)[:, 1]

    ROC(y_test, y_score)
