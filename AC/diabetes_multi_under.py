# Import das bibliotecas necessárias
import time
from sklearn.neural_network import MLPClassifier
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
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, classification_report

# Leitura do ficheiro csv com os dados
df = pd.read_csv ('diabetes_multi.csv', delimiter = ",")

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

# Distribuição de spam e não spam nos dados de treino antes do undersamplimg
sns.countplot(x = y_train)
plt.title("Diabetes distribution (train with undersampling)")
plt.show()
print(df['Diabetes_012'].value_counts(), "\n")

##---------- Pré-processamento ----------##
# Reduzir o número de exemplos da classe dominante (undersampling) nos dados de treino
undersampler = RandomUnderSampler(sampling_strategy = 'auto', random_state = 42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Distribuição de diabetes e não diabetes nos dados de treino com underSampling
sns.countplot(x = y_train_under)
plt.title("Diabetes distribution (train with underSampling)", fontsize = 18)
plt.xlabel("Diabetes", fontsize = 14)
plt.ylabel("Count", fontsize = 14)
plt.ylim(0, 170000)
plt.show()


##---------- Neuronal Network ----------##
# Create a MLP classifier
mlp = MLPClassifier(hidden_layer_sizes = (10, 5), activation = 'relu', solver = 'adam', max_iter = 1000, tol = 0.0001,random_state = 42)

# Train the classifier
mlp.fit(X_train_under, y_train_under)
y_pred = mlp.predict(X_test)

# Evaluate the classifier
# Macro-Average (igual peso para todas classes)
macro_precision = precision_score(y_test, y_pred, average='macro')
macro_recall = recall_score(y_test, y_pred, average='macro')
macro_f1 = f1_score(y_test, y_pred, average='macro')

# Weighted-Average (ponderado pelo número de amostras)
weighted_precision = precision_score(y_test, y_pred, average='weighted')
weighted_recall = recall_score(y_test, y_pred, average='weighted')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

print(classification_report(y_test, y_pred))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}\n")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}\n")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Diabetes', 'Diabetes Tipo 1', 'Diabetes Tipo 2'], yticklabels=['Não Diabetes', 'Diabetes Tipo 1', 'Diabetes Tipo 2'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()