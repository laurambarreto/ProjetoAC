# Import das bibliotecas necessárias
import time
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
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
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, classification_report

# Leitura do ficheiro csv com os dados
df = pd.read_csv ('diabetes_binary.csv', delimiter = ",")

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
sns.heatmap(correlation_matrix,cmap = 'coolwarm', annot = False)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Distribuição de diabetes e não diabetes do dataset
ax=sns.countplot(x = y, color = '#73D7FF') 
plt.title("Diabetes binary distribution", fontsize = 22)
plt.xlabel("Diabetes_binary", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
# Aumentar o tamanho dos números dos eixos
plt.tick_params(axis='both', which='major', labelsize=13)
# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis='both', zorder=0)
# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)
plt.show()

# Distribuição de diabetes e não diabetes nos dados de treino antes do undersamplimg
sns.countplot(x = y_train)
plt.title("Diabetes distribution (train)", fontsize = 20)
plt.xlabel("Diabetes", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
plt.show()

print(df['Diabetes_binary'].value_counts(), "\n")

##---------- Pré-processamento ----------##
# Reduzir o número de exemplos da classe dominante (undersampling) nos dados de treino
undersampler = RandomUnderSampler(sampling_strategy = 'auto', random_state = 42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Distribuição de diabetes e não diabetes nos dados de treino com undersampling
sns.countplot(x = y_train_under)
plt.title("Diabetes distribution (train with undersampling)", fontsize = 18)
plt.xlabel("Diabetes", fontsize = 14)
plt.ylabel("Count", fontsize = 14)
plt.ylim(0, 170000)
plt.show()

##---------- Neuronal Network ----------##
# Criar o MLP classifier
mlp = MLPClassifier(hidden_layer_sizes = (10, 5), activation = 'relu', solver = 'adam', max_iter = 1000, tol = 0.0001,random_state = 42)

# Treinar the classifier
mlp.fit(X_train_under, y_train_under)
y_pred = mlp.predict(X_test)

# Avaliar o classifier
print('Class labels:', np.unique(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Recall: %.2f' % recall_score(y_test, y_pred))
print('Precision: %.2f' % precision_score(y_test, y_pred))
print('F1: %.2f' % f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

##---------- SVM ----------##
