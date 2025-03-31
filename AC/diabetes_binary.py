# Import das bibliotecas necessárias
import time
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score

# Leitura do ficheiro csv com os dados
df = pd.read_csv ('diabetes_multi.csv', delimiter = ",")

# Seleção das colunas das características
X = df.drop("Diabetes_binary", axis = 1)

# Seleção da coluna target
y = df.Diabetes_binary

# Divisão em conjunto de treino e de teste
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state = 42)

# Função que retorna as métricas de avaliação
def metricas(y_pred, y_true):
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred), precision_score(y_true, y_pred)

##---------- Análise inicial ----------##
# Informações sobre o Dataset
print(df.info(), "\n")

# Correlações entre todas as colunas 
correlation_matrix = df.corr()
plt.figure(figsize = (6, 4))
sns.heatmap(correlation_matrix,cmap='coolwarm', annot = False)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Distribuição de spam e não spam nos dados de treino antes do undersamplimg
sns.countplot(x = y_train)
plt.title("Spam distribution (train)")
plt.show()