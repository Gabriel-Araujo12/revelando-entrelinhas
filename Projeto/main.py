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

colunas_apagadas = ['SG_UF_NOT', 'ID_REGIONA', 'ID_MUNICIP', 'ID_UNIDADE', 'ID_PAIS', 'SG_UF', 'ID_RG_RESI', 'ID_MN_RESI', 'DT_NOTIFIC', 'DT_NASC', 'DT_NOTIFIC', 'DT_SIN_PRI', 'DT_NASC', 'DT_INTERNA', 'DT_COLETA', 'DT_PCR', 'DT_EVOLUCA', 'DT_ENCERRA', 'DT_DIGITA', 'CS_SEXO', 'SG_UF_INTE', 'ID_RG_INTE', 'ID_MN_INTE']
X = df.drop(columns = colunas_apagadas)

#Sugestão de colunas para apagar: ['NU_NOTIFIC', 'DT_NOTIFIC', 'SG_UF_NOT', 'ID_MUNICIP', 'ID_REGIONA', 'ID_UNIDADE', 'TEM_CPF', 'NU_CPF', 'ESTRANG', 'NU_CNS', 'NM_PACIENT', 'DT_NASC', 'CS_RACA', 'CS_ETINIA', 'POV_CT', 'TP_POV_CT', 'CS_ESCOL_CT', 'PAC_COCBO', 'NM_MAE_PAC', 'NU_CEP', 'SG_UF', 'ID_RG_RESI', 'ID_MN_RESI', 'NM_BAIRRO', 'NM_LOGRADO', 'NU_NUMERO', 'NM_COMPLEM', 'NU_DDD_TEL', 'CS_ZONA', 'ID_PAIS', 'SG_UF_INTE', 'ID_RG_INTE', 'ID_MN_INTE', 'ID_UN_INTE', 'LAB_AN', 'CO_LAB_AN', 'LAB_PCR', 'DT_EVOLUCA', 'DT_ENCERRA', 'NOME_PROF', 'REG_PROF', 'DT_DIGITA', 'DT_INTERNA', 'DT_COLETA', 'DT_PCR', 'CS_ESCOL_N', 'NU_IDADE_N', 'CO_MUN_RES', 'AVE_SUINO', 'ID_RG_INTE', 'CO_RG_RESI']

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