# Import das bibliotecas necessárias
import time
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
df = pd.read_csv ('diabetes_012.csv', delimiter = ",")

# Verificar dados nulos (NÃO HÁ NENHUM DADO A FALTAR)
print("Missing data per column:")
print(df.isnull().sum(), "\n")

# Verificar réplicas completas (linhas idênticas)
duplicated = df[df.duplicated(keep = False)]  # `keep = False` marca todas as ocorrências
print(f"Number of duplicated lines: {len(duplicated)}") 

# Agrupa linhas idênticas e conta ocorrências
count_duplicated = df.groupby(df.columns.tolist()).size().reset_index(name = 'Count')

# Mostra as linhas repetidas
print(count_duplicated.sort_values('Count', ascending = False))

# Remover duplicados
df = df.drop_duplicates()
print("Distribution after removing duplicates:")
print(df['Diabetes_012'].value_counts(), "\n")
# Seleção das colunas das características
X = df.drop("Diabetes_012", axis = 1)

# Seleção da coluna target
y = df.Diabetes_012

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

# Distribuição dos dados de treino antes do SMOTE
sns.countplot(x = y_train)
plt.title("Diabetes distribution (train with SMOTE)")
plt.show()
print(df['Diabetes_012'].value_counts(), "\n")

# Verificar se são linearmente separáveis
# Normalizar os dados
scaler = StandardScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

# Reduzir a dimensionalidade para 2D para visualização (PCA)
pca = PCA(n_components = 2)
X_reduced = pca.fit_transform(X_norm)
# Define um array de cores fixas: ex. vermelho, verde, azul
colors = {0: '#ffc0dc', 1: 'blue', 2: '#ffff00'}
# Mapeia as cores com base nas classes
maped_colors = [colors[classe] for classe in y]
legenda_cores = [Patch(color = cor, label = f'Class {classe}') for classe, cor in colors.items()]
# Visualizar os dados reduzidos em 2D
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c = maped_colors, alpha = 0.5)
plt.title("Real classes (0 vs. 1 vs. 2)", fontsize = 18)
plt.legend(handles = legenda_cores, fontsize = 12, title_fontsize = 13)
plt.xlabel("1st component", fontsize = 14)
plt.ylabel("2nd component", fontsize = 14)
plt.show()

##---------- Pré-processamento ----------##
# Aplicar SMOTE aos dados de treino
smote = SMOTE(sampling_strategy = 'auto', random_state = 42)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

# Distribuição de diabetes e não diabetes nos dados de treino com SMOTE
sns.countplot(x = y_train_SMOTE)
plt.title("Diabetes distribution (train with SMOTE)", fontsize = 18)
plt.xlabel("Diabetes", fontsize = 14)
plt.ylabel("Count", fontsize = 14)
plt.ylim(0, 170000)
plt.show()

##---------- Neuronal Network ----------##
# Criar o MLP classifier
mlp = MLPClassifier(hidden_layer_sizes = (10, 5), activation = 'relu', solver = 'adam', max_iter = 1000, tol = 0.0001,random_state = 42)

# Treinar o classifier
mlp.fit(X_train_SMOTE, y_train_SMOTE)
y_pred = mlp.predict(X_test)

# Avaliar o classifier
# Macro-Average (igual peso para todas classes)
macro_precision = precision_score(y_test, y_pred, average = 'macro')
macro_recall = recall_score(y_test, y_pred, average = 'macro')
macro_f1 = f1_score(y_test, y_pred, average = 'macro')

# Weighted-Average (ponderado pelo número de amostras)
weighted_precision = precision_score(y_test, y_pred, average = 'weighted')
weighted_recall = recall_score(y_test, y_pred, average = 'weighted')
weighted_f1 = f1_score(y_test, y_pred, average = 'weighted')

print(classification_report(y_test, y_pred))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}\n")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}\n")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['No Diabetes', 'Prediabetes', 'Diabetes'], yticklabels = ['No Diabetes', 'Prediabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix')
plt.show()