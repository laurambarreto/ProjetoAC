# Import das bibliotecas necessárias
import time
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
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
df = pd.read_csv ('diabetes_01.csv', delimiter = ",")

# Seleção das colunas das características
X = df.drop("Diabetes_binary", axis = 1)

# Seleção da coluna target
y = df.Diabetes_binary

# Divisão em conjunto de treino e de teste
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state = 42)

# Função que retorna as métricas de avaliação
def metricas(y_pred, y_true):
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred), precision_score(y_true, y_pred)

##---------- ANÁLISE INICIAL ----------##
# Informações sobre o Dataset
print(df.info(), "\n")

# Distribuição de diabetes e não diabetes do dataset
ax = sns.countplot(x = y, color = '#73D7FF') 
plt.title("Diabetes binary distribution before", fontsize = 20)
plt.xlabel("Diabetes_binary", fontsize = 16)
plt.ylabel("Count", fontsize = 16)

# Aumentar o tamanho dos números dos eixos
plt.tick_params(axis = 'both', which = 'major', labelsize = 13)

# Definir os ticks do eixo Y de 25.000 em 25.000
max_val = y.value_counts().max()
plt.yticks(ticks=range(0, max_val + 25000, 25000))

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)

# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)
plt.ylim(0, 225000)
plt.show()



print(df['Diabetes_binary'].value_counts(), "\n")

##---------- PRÉ-PROCESSAMENTO ----------##
# Remover duplicados
df = df.drop_duplicates()
print("Distribution after removing duplicates:")
print(df['Diabetes_binary'].value_counts(), "\n")

df_l0 = df[df['Diabetes_binary'] == 0]  # classe maioritária
df_l1 = df[df['Diabetes_binary'] == 1]  # classe minoritária
# Calcular IQR e remover outliers **apenas** da classe 0
Q1 = df_l0.quantile(0.25)
Q3 = df_l0.quantile(0.75)
IQR = Q3 - Q1
cond = ~((df_l0 < (Q1 - 1.5 * IQR)) | (df_l0 > (Q3 + 1.5 * IQR))).any(axis=1)
df_l0_clean = df_l0[cond]

# Juntar as duas classes
df = pd.concat([df_l0_clean, df_l1], axis=0)

# Seleção das colunas das características
X = df.drop("Diabetes_binary", axis = 1)

# Seleção da coluna target
y = df.Diabetes_binary

# Distribuição de diabetes e não diabetes nos dados de treino depois da remoção de outliers e linhas duplicadas
ax = sns.countplot(x = y, color = '#73D7FF')
plt.title("Diabetes binary distribution after", fontsize = 20)
plt.xlabel("Diabetes binary", fontsize = 16)
plt.ylabel("Count", fontsize = 16)

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)

# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)
plt.ylim(0, 225000)
plt.show()

# Divisão em conjunto de treino e de teste
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state = 42)

# Reduzir o número de exemplos da classe dominante (undersampling) nos dados de treino
# Escolher o novo tamanho alvo
objective_len = 40000

# Aplica UNDERSAMPLING à classe 0
rus = RandomUnderSampler(sampling_strategy = {0: objective_len}, random_state = 42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)

# Aplica SMOTE a tudo o que saiu do RUS para aumentar as classes 1 e 2
smote = SMOTE(sampling_strategy = {1: objective_len}, random_state = 42)
X_train_bal, y_train_bal = smote.fit_resample(X_rus, y_rus)

# Juntar tudo num novo DataFrame
df_final = pd.concat([pd.DataFrame(X_train_bal, columns = X.columns),
                      pd.Series(y_train_bal, name = "Diabetes_01")], axis = 1)

# Distribuição de diabetes e não diabetes nos dados de treino com undersampling
ax  = sns.countplot(x = y_train_bal, color = '#73D7FF')
plt.title("Diabetes distribution (balanced with SMOTE + under)", fontsize = 20)
plt.xlabel("Diabetes binary", fontsize = 16)
plt.ylabel("Count", fontsize = 16)

# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)

# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)
plt.ylim(0, 160000)
plt.show()

##---------- MODELIZAÇÃO ----------##
# Normalizar os dados
scaler = StandardScaler()
X_train_scaled_bal = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

#---------- REDES NEURONAIS ----------#
# Criar o MLP classifier
mlp = MLPClassifier(hidden_layer_sizes = (10,5), activation = 'relu', solver = 'adam', max_iter = 1000, tol = 0.0001, random_state = 42)

# Para contar o tempo de treino
start_time = time.time()
mlp.fit(X_train_scaled_bal, y_train_bal)
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
svm.fit(X_train_scaled_bal, y_train_bal)
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