import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df = pd.read_csv('iris.csv', sep=',', engine='python')
X = df.drop(['species'],axis=1).values   
y = df['species'].values

clf = GaussianNB()
clf.fit(X, y)

y_predict = clf.predict(X)
print ('------------Gaussian NB------------')
print (y_predict)
print (clf.predict_proba(X))
# ~ print (clf.predict_log_proba(X))

print (accuracy_score(y, y_predict))
print (accuracy_score(y, y_predict, normalize=False))

target_names =clf.classes_
print (target_names)

print(classification_report(y, y_predict, target_names=target_names))
print (confusion_matrix(y, y_predict, labels=target_names))


# ~ print ('\n------------Multinomial NB------------')
# ~ clf = MultinomialNB()
# ~ clf.fit(X, y)

# ~ y_predict = clf.predict(X)
# ~ print (accuracy_score(y, y_predict))
# ~ print (accuracy_score(y, y_predict, normalize=False))

# ~ print(classification_report(y, y_predict, target_names=target_names))
# ~ cm = confusion_matrix(y, y_predict, labels=target_names)
# ~ print (cm)
# ~ disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
# ~ disp.plot()
# ~ plt.show()
