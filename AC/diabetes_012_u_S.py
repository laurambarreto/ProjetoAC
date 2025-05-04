# Import das bibliotecas necessárias
import time
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from imblearn.under_sampling import RandomUnderSampler
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

# Remover duplicados
df = df.drop_duplicates()
print("Distribution after removing duplicates:")
print(df['Diabetes_012'].value_counts(), "\n")

df_l0 = df[df['Diabetes_012'] == 0]  # classe maioritária
df_l1 = df[df['Diabetes_012'] == 1]  # classe minoritária
df_l2 = df[df['Diabetes_012'] == 2]  # classe minoritária
# Calcular IQR e remover outliers **apenas** da classe 0
Q1 = df_l0.quantile(0.25)
Q3 = df_l0.quantile(0.75)
IQR = Q3 - Q1
cond = ~((df_l0 < (Q1 - 1.5 * IQR)) | (df_l0 > (Q3 + 1.5 * IQR))).any(axis=1)
df_l0_clean = df_l0[cond]
# Juntar as duas classes
df = pd.concat([df_l0_clean, df_l1,df_l2], axis=0)
print(df['Diabetes_012'].value_counts(), "\n")

# Seleção das colunas das características
X = df.drop("Diabetes_012", axis = 1)

# Seleção da coluna target
y = df.Diabetes_012

# Divisão em conjunto de treino e de teste
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state = 42)

# Escolher o novo tamanho alvo
objective_len = 40000

# Aplica UNDERSAMPLING à classe 0
rus = RandomUnderSampler(sampling_strategy = {0: objective_len}, random_state = 42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)

# Aplica SMOTE a tudo o que saiu do RUS para aumentar as classes 1 e 2
smote = SMOTE(sampling_strategy = {1: objective_len, 2: objective_len}, random_state = 42)
X_train_bal, y_train_bal = smote.fit_resample(X_rus, y_rus)

# Juntar tudo num novo DataFrame
df_final = pd.concat([pd.DataFrame(X_train_bal, columns = X.columns),
                      pd.Series(y_train_bal, name = "Diabetes_012")], axis = 1)

# Distribuição de diabetes e não diabetes nos dados de treino depois do SMOTE 
ax = sns.countplot(x = y_train_bal, color = '#73D7FF')
plt.title("Diabetes distribution (balanced with SMOTE + under)", fontsize = 20)
plt.xlabel("Diabetes 012", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
# Colocar grelha nos dois eixos, atrás das barras
plt.grid(True, axis = 'both', zorder = 0)
# Colocar as barras à frente da grelha
for bar in ax.patches:
    bar.set_zorder(3)
plt.ylim(0, 160000)
plt.show()


##---------- Neuronal Network ----------##
# Criar o MLP classifier
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes = (10,5), activation = 'relu', solver = 'adam', max_iter = 1000, tol = 0.0001, random_state = 42)

start_time = time.time()
mlp.fit(X_train_scaled, y_train_bal)
tempo_total = time.time() - start_time
print(f"Total training time: {tempo_total:.2f} seconds")

y_pred_mlp = mlp.predict(X_test)

# Avaliar o classifier
# Macro-Average (igual peso para todas classes)
macro_precision = precision_score(y_test, y_pred_mlp, average = 'macro')
macro_recall = recall_score(y_test, y_pred_mlp, average = 'macro')
macro_f1 = f1_score(y_test, y_pred_mlp, average = 'macro')

# Weighted-Average (ponderado pelo número de amostras)
weighted_precision = precision_score(y_test, y_pred_mlp, average = 'weighted')
weighted_recall = recall_score(y_test, y_pred_mlp, average = 'weighted')
weighted_f1 = f1_score(y_test, y_pred_mlp, average = 'weighted')

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_mlp))
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}\n")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}\n")
print(classification_report(y_test, y_pred_mlp))

cm = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['No Diabetes', 'Prediabetes', 'Diabetes'], yticklabels = ['No Diabetes', 'Prediabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix - Using MLP classifier')
plt.show()

##---------- SVM ----------## 
# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar e treinar o modelo SVM
svm = SVC(kernel = 'linear')  # Pode usar 'rbf', 'poly', etc.
svm.fit(X_train, y_train)

# Fazer previsões
y_pred_svm = svm.predict(X_test)

# Avaliar o modelo SVM
print ("SVM RESULTS")
print('Class labels:', np.unique(y_test))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_svm))
print('Recall: %.2f' % recall_score(y_test, y_pred_svm))
print('Precision: %.2f' % precision_score(y_test, y_pred_svm))
print('F1: %.2f' % f1_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['No Diabetes', 'Prediabetes', 'Diabetes'], yticklabels = ['No Diabetes', 'Prediabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix - Using SVM')
plt.show()