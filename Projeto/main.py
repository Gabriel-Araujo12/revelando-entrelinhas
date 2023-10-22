import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

path = (r"C:\Users\gabri\Downloads\TCC\Projeto\sinresp.csv")
df = pd.read_csv(path, on_bad_lines='skip', sep=';', low_memory=False)

#Pré-processamento dos dados da base
limite = len(df) * 0.5
df = df.dropna(thresh = limite, axis = 1)
df = df.dropna(subset = ['EVOLUCAO'])

colunas_apagadas = ['SG_UF_NOT', 'ID_REGIONA', 'ID_MUNICIP', 'ID_UNIDADE', 'ID_PAIS', 'SG_UF', 'ID_RG_RESI', 'ID_MN_RESI', 'DT_NOTIFIC', 'DT_NASC', 'DT_NOTIFIC', 'DT_SIN_PRI', 'DT_NASC', 'DT_INTERNA', 'DT_COLETA', 'DT_PCR', 'DT_EVOLUCA', 'DT_ENCERRA', 'DT_DIGITA', 'CS_SEXO', 'SG_UF_INTE', 'ID_RG_INTE', 'ID_MN_INTE', 'CS_ESCOL_N', 'NU_IDADE_N', 'CO_MUN_RES', 'AVE_SUINO', 'ID_RG_INTE', 'CO_RG_RESI']
X = df.drop(columns = colunas_apagadas)

#Convertendo todas as colunas restantes do DataFrame para valores numéricos
X = X.apply(pd.to_numeric, errors='coerce')

#Preenchendo os valores ausentes das colunas com sua média
X = X.fillna(X.mean())

#Aplicação do KNN
Y = df.EVOLUCAO

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 10)

clf = KNeighborsClassifier(n_neighbors = 11)
clf.fit(x_train, y_train)

pred_scores = clf.predict_proba(x_test)
print(pred_scores)

y_pred = clf.predict(x_test)
te_acc= (sklearn.metrics.accuracy_score(y_test, y_pred)) 
print('Acurácia obtida: ', te_acc)

print(X.shape)