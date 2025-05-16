# Import das bibliotecas necessárias
import time
from sklearn.svm import SVC
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
import optuna

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

##---------- Análise inicial ----------##
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

##---------- Pré-processamento ----------##
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
undersampler = RandomUnderSampler(sampling_strategy = 'auto', random_state = 42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Distribuição de diabetes e não diabetes nos dados de treino com undersampling
ax = sns.countplot(x = y_train_under, color = '#73D7FF')
plt.title("Diabetes distribution (balanced with undersampling)", fontsize = 20)
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
X_train_scaled_under = scaler.fit_transform(X_train_under)
X_test_scaled = scaler.transform(X_test)

##---------- REDES NEURONAIS ----------##
# Criar o MLP classifier
mlp = MLPClassifier(hidden_layer_sizes = (10, 5), activation = 'relu', solver = 'adam', max_iter = 1000, tol = 0.0001, random_state = 42)

# Para contar o tempo de treino
start_time = time.time()
mlp.fit(X_train_scaled_under, y_train_under)
time_total = time.time() - start_time
print(f"Total training time: {time_total:.2f} seconds")

# Previsões
y_pred_mlp = mlp.predict(X_test_scaled)

# Avaliar o classifier
print ("MLP CLASSIFIER RESULTS")
print('Class labels:', np.unique(y_test))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_mlp))
print('Recall: %.2f' % recall_score(y_test, y_pred_mlp))
print('Precision: %.2f' % precision_score(y_test, y_pred_mlp))
print('F1: %.2f' % f1_score(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))

# Matriz de confusão do classificador MLP
cm = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['No Diabetes', 'Diabetes'], yticklabels = ['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix - Using MLP classifier')
plt.show()

##---------- SVM ----------##
def objective(trial):
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    C = trial.suggest_float('C', 1e-3, 1e3, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    
    if kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 5)
    else:
        degree = 3
    
    svm = SVC(kernel=kernel, C=C, degree=degree, max_iter=max_iter)
    svm.fit(X_train, y_train)
    preds = svm.predict(X_train)
    score = f1_score(y_train, preds, average='weighted')
    return score

# Criar o estudo Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print(f"Melhor trial: {study.best_trial.params}")
print(f"Melhor F1-score na validação: {study.best_value:.4f}")

# Treinar o melhor modelo com os dados de treino + validação
best_params = study.best_trial.params

#Definir degree corretamente
degree = best_params['degree'] if best_params['kernel'] == 'poly' else 3

best_svm = SVC(
    kernel=best_params['kernel'],
    C=best_params['C'],
    degree=degree,
    max_iter=best_params['max_iter']
)

# Treinar com treino + validação
X_train_val = np.concatenate((X_train, X_val), axis=0)
y_train_val = np.concatenate((y_train, y_val), axis=0)

best_svm.fit(X_train_val, y_train_val)

# Avaliar no teste
test_preds = best_svm.predict(X_test)
test_f1 = f1_score(y_test, test_preds, average='weighted')

print(f"F1-score no conjunto de teste: {test_f1:.4f}")

# Avaliar o modelo SVM
print ("SVM RESULTS")
print('Class labels:', np.unique(y_test))
print('Accuracy: %.2f' % accuracy_score(y_test, test_preds))
print('Recall: %.2f' % recall_score(y_test, test_preds))
print('Precision: %.2f' % precision_score(y_test, test_preds))
print('F1: %.2f' % f1_score(y_test, test_preds))
print(classification_report(y_test, test_preds))

# Matriz de confusão do método SVM
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['No Diabetes', 'Diabetes'], yticklabels = ['No Diabetes','Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix - Using SVM')
plt.show()