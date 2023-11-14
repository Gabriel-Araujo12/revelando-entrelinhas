import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics

path = (r"C:\Users\gabri\Downloads\TCC\Projeto\sinresp.csv")
df = pd.read_csv(path, on_bad_lines='skip', sep=';', low_memory=False)

#Reduzindo aleatoriamente a base de dados
col_base = 0.5
df = df.sample(frac=1, axis=1).iloc[:, :int(col_base * df.shape[1])]

#Convertendo colunas não numéricas para numéricas
le = LabelEncoder()
colunas_nn = df.select_dtypes(exclude=['number']).columns

while len(colunas_nn) > 0:
    column = colunas_nn[0]
    df[column] = le.fit_transform(df[column])
    colunas_nn = df.select_dtypes(exclude=['number']).columns

#Preenchendo valores nulos como 0
df = df.fillna(0)

X = df

# Aplicação do KNN
Y = df.EVOLUCAO
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=10)

clf = KNeighborsClassifier(n_neighbors=11)
clf.fit(x_train, y_train)

pred_scores = clf.predict_proba(x_test)
print(pred_scores)

y_pred = clf.predict(x_test)
te_acc = metrics.accuracy_score(y_test, y_pred)
print('Acurácia obtida: ', te_acc)

print(X.shape)
