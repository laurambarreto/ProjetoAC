# Import das bibliotecas necessárias
import time
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
df = pd.read_csv ('diabetes_01.csv', delimiter = ",")


# Seleção das colunas das características
X = df.drop("Diabetes_binary", axis = 1)

# Seleção da coluna target
y = df.Diabetes_binary

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
plt.xticks(ticks = np.arange(len(df.columns)) + 0.5, labels = df.columns, rotation = 45, ha = 'right', fontsize = 8)
plt.yticks(ticks = np.arange(len(df.columns)) + 0.5, labels = df.columns, rotation = 0, fontsize = 8)
plt.tight_layout()
plt.show()

# Distribuição de diabetes e não diabetes do dataset
ax = sns.countplot(x = y, color = '#73D7FF')
plt.title("Diabetes distribution before")
# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)
# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)

plt.ylim(0, 225000)
plt.show()

# Verificar se são linearmente separáveis
# Normalizar os dados
scaler = StandardScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

# Reduzir a dimensionalidade para 2D para visualização (PCA)
pca = PCA(n_components = 2)
X_reduced = pca.fit_transform(X_norm)

# Define um array de cores fixas: ex. vermelho, verde, azul
colors = {0: '#ffc0dc', 1: '#ffff00'}

# Mapeia as cores com base nas classes
maped_colors = [colors[classe] for classe in y]

# Cria os elementos da legenda manualmente
legenda_cores = [Patch(color = cor, label = f'Class {classe}') for classe, cor in colors.items()]
 
# Visualizar os dados reduzidos em 2D
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c = maped_colors, alpha = 0.5)
plt.title("Real classes (0 vs. 1)", fontsize = 20)
plt.xlabel("1st component", fontsize = 14)
plt.ylabel("2nd component", fontsize = 14)
plt.legend(handles = legenda_cores, fontsize = 12, title_fontsize = 13)
plt.show()

# Variância explicada pelas duas e três primeiras componentes
var_2 = sum(pca.explained_variance_ratio_[:2])
print(f"Variance explained by the first two components: {var_2:.4f}")

##---------- Pré-processamento ----------##
# Verificar dados nulos (NÃO HÁ NENHUM DADO A FALTAR)
print("Missing data per column:")
print(df.isnull().sum(), "\n")

# Verificar duplicatas completas (linhas idênticas)
duplicated = df[df.duplicated(keep = False)]  # `keep = False` marca todas as ocorrências
print(f"Number of duplicated lines: {len(duplicated)}") 

# Agrupa linhas idênticas e conta ocorrências
count_duplicated = df.groupby(df.columns.tolist()).size().reset_index(name = 'Count')

# Mostra as linhas repetidas
print(count_duplicated.sort_values('Count', ascending = False))

# Remover duplicados
df = df.drop_duplicates()
df_l0 = df[df['Diabetes_binary'] == 0]  # classe maioritária
df_l1 = df[df['Diabetes_binary'] == 1]  # classe minoritária

# Calcular IQR e remover outliers **apenas** da classe 0
Q1 = df_l0.quantile(0.25)
Q3 = df_l0.quantile(0.75)
IQR = Q3 - Q1
cond = ~((df_l0 < (Q1 - 1.5 * IQR)) | (df_l0 > (Q3 + 1.5 * IQR))).any(axis = 1)
df_l0_clean = df_l0[cond]
# Juntar as duas classes
df = pd.concat([df_l0_clean, df_l1], axis = 0)

X = df.drop("Diabetes_binary", axis = 1)
y = df.Diabetes_binary

# Divisão em conjunto de treino e de teste
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state = 42)

# Distribuição das classes nos dados de treino no final da remoção de outliers e linhas duplicadas
ax = sns.countplot(x = y_train, color = '#73D7FF')
plt.title("Diabetes binary distribution after")

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)

# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)

plt.ylim(0, 225000)
plt.show()
print(y_train.value_counts(), "\n")

# Aplicar SMOTE aos dados de treino
smote = SMOTE(sampling_strategy = 'auto', random_state = 42)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

# Distribuição das classes nos dados de treino depois de usar SMOTE 
ax = sns.countplot(x = y_train_SMOTE, color = '#73D7FF')
plt.title("Diabetes distribution (balanced with SMOTE)", fontsize = 20)
plt.xlabel("Diabetes binary", fontsize = 16)
plt.ylabel("Count", fontsize = 16)

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)
# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)

plt.ylim(0, 225000)
plt.show()

print(y_train_SMOTE.value_counts(), "\n")

##---------- MODELIZAÇÃO ----------##
# Normalizar os dados
scaler = StandardScaler()
X_train_scaled_SMOTE = scaler.fit_transform(X_train_SMOTE)
X_test_scaled = scaler.transform(X_test)

#---------- REDES NEURONAIS ----------#
# Criar o MLP classifier
mlp = MLPClassifier(hidden_layer_sizes = (10,5), activation = 'relu', solver = 'adam', max_iter = 1000, tol = 0.0001, random_state = 42)

# Para contar o tempo de treino
start_time = time.time()
mlp.fit(X_train_scaled_SMOTE, y_train_SMOTE)
time_total = time.time() - start_time
print(f"Total training time: {time_total:.2f} seconds")

# Previsões
y_pred_mlp = mlp.predict(X_test_scaled)

# Avaliar o classifier
# Macro-Average (igual peso para todas classes)
macro_precision = precision_score(y_test, y_pred_mlp, average = 'macro')
macro_recall = recall_score(y_test, y_pred_mlp, average = 'macro')
macro_f1 = f1_score(y_test, y_pred_mlp, average = 'macro')

print ("MLP CLASSIFIER RESULTS")
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_mlp))
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}\n")
print(classification_report(y_test, y_pred_mlp))

# Matriz de confusão do classificador MLP
cm = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['No Diabetes', 'Diabetes'], yticklabels = ['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix - Using MLP classifier')
plt.show()

#---------- SVM ----------# 
# Criar e treinar o modelo SVM
svm = SVC(kernel = 'linear')  # Pode usar 'rbf', 'poly', etc.

# Para contar o tempo de treino
start_time = time.time()
svm.fit(X_train_scaled_SMOTE, y_train_SMOTE)
time_total = time.time() - start_time
print(f"Total training time: {time_total:.2f} seconds")

# Previsões
y_pred_svm = svm.predict(X_test_scaled)

# Avaliar o modelo SVM
print ("SVM RESULTS")
print('Class labels:', np.unique(y_test))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_svm))
print('Recall: %.2f' % recall_score(y_test, y_pred_svm))
print('Precision: %.2f' % precision_score(y_test, y_pred_svm))
print('F1: %.2f' % f1_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Matriz de confusão do método SVM
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['No Diabetes', 'Diabetes'], yticklabels = ['No Diabetes','Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix - Using SVM')
plt.show()