
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
df = pd.read_csv('dataset.csv', sep=',', engine='python')
X = df.drop(['posee_auto'],axis=1).values   
y = df['posee_auto'].values
#plt.scatter(X,y)
	
#Separa el corpus cargado en el DataFrame en el 90% para entrenamiento y el 10% para pruebas
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.2, shuffle = True, random_state=0)
	
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) #la clase predicha
print ('\n Clase real', y_test)
print ('\n Clase predicha', y_pred,'\n\n')

print('\nMatriz de confusión')
print(confusion_matrix(y_test, y_pred))
print('\nAccuracy')
print('Porcentaje de instancias predichas correctamente',accuracy_score(y_test, y_pred)) 
print('Cantidad de instancias predichas correctamente',accuracy_score(y_test, y_pred, normalize=False), '\n\n') 

               #~ #Clase predicha
#~ #               0 1
#~ #           -------
#~ #Clase real 0 | 0 1
#~ #           1 | 0 3
cm = confusion_matrix(y_test, y_pred,labels= clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()

y_pred_proba = clf.predict_proba(X_test) 
print ('\n Probabilidad de pertenecer a una clase\n', y_pred_proba ,'\n\n')
plt.show()
