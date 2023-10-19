import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

colunas_apagadas = ['SG_UF_NOT', 'ID_REGIONA', 'ID_MUNICIP', 'ID_UNIDADE', 'ID_PAIS', 'SG_UF', 'ID_RG_RESI', 'ID_MN_RESI', 'DT_NOTIFIC', 'DT_NASC']
X = df.drop(columns = colunas_apagadas)

#Aplicação do KNN

Y = df.EVOLUCAO

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 10)

clf = KNeighborsClassifier(n_neighbors = 11)
clf.fit(x_train, y_train)

pred_scores = clf.predict_proba(x_test)
print(pred_scores)