# Import das bibliotecas necessárias
import time
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
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

# Verificar dados nulos (NÃO HÁ NENHUM DADO A FALTAR)
print("Dados em falta por coluna:")
print(df.isnull().sum(), "\n")

# Verificar duplicatas completas (linhas idênticas)
duplicatas = df[df.duplicated(keep=False)]  # `keep=False` marca todas as ocorrências
print(f"Número de linhas duplicadas: {len(duplicatas)}") 
# Agrupa linhas idênticas e conta ocorrências
contagem_duplicatas = df.groupby(df.columns.tolist()).size().reset_index(name='Contagem')
# Mostra as linhas repetidas
print(contagem_duplicatas.sort_values('Contagem', ascending=False))

# Remover duplicados
df = df.drop_duplicates()

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
sns.countplot(x = y)
plt.title("Diabetes distribution")
plt.show()

# Distribuição de diabetes e não diabetes nos dados de treino antes de usar SMOTE
sns.countplot(x = y_train)
plt.title("Diabetes distribution (train)")
plt.show()
print(df['Diabetes_binary'].value_counts(), "\n")

# Verificar se são linearmente separáveis
# Normalizar os dados
scaler = StandardScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
# Reduzir a dimensionalidade para 2D para visualização (PCA)
pca = PCA(n_components = 2)
X_reduced = pca.fit_transform(X_norm)

#Visualizar os dados reduzidos em 2D
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c = y, cmap = 'coolwarm', alpha = 0.5)
plt.title("Classes Reais (diabetes vs. não diabetes)")
plt.show()


##---------- Pré-processamento ----------##
# Aplicar SMOTE aos dados de treino
smote = SMOTE(sampling_strategy = 'auto', random_state = 42)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

# Distribuição de diabetes e não diabetes nos dados de treino com SMOTE
sns.countplot(x = y_train_SMOTE)
plt.title("Diabetes distribution (train with undersampling)", fontsize = 18)
plt.xlabel("Diabetes", fontsize = 14)
plt.ylabel("Count", fontsize = 14)
plt.ylim(0, 170000)
plt.show()

##---------- Neuronal Network ----------##
# Create a MLP classifier
mlp = MLPClassifier(hidden_layer_sizes = (10, 5), activation = 'relu', solver = 'adam', max_iter = 1000, tol = 0.0001,random_state = 42)

# Train the classifier
mlp.fit(X_train_SMOTE, y_train_SMOTE)
y_pred = mlp.predict(X_test)

# Evaluate the classifier
print('Class labels:', np.unique(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Recall: %.2f' % recall_score(y_test, y_pred))
print('Precision: %.2f' % precision_score(y_test, y_pred))
print('F1: %.2f' % f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))