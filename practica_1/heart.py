import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import csv


#importar el dataset
dataset = pd.read_csv("heart.csv")
#dataset

y = dataset.iloc[:, -1:].values #todas menos la ultima columna
x  = dataset.iloc[:, :-1].values#solo la ultima columna


## dividir el dataset en testing y en entrenamiento
from sklearn.model_selection import train_test_split
x_train , x_test ,y_train , y_test = train_test_split(x,y,test_size= 0.4, random_state= 0,shuffle = False)

#target --> etiqueta



headers_x = list(dataset.columns)
headers_x.remove('target')


#importar a csv ->

with open('x_e.csv', 'w', newline = '') as csvfile:
    my_writer = csv.writer(csvfile)
    my_writer.writerow(headers_x)
    for x in x_train:
        my_writer.writerow(x)


with open('y_e.csv', 'w', newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ' ')
    my_writer.writerow(["target"])
    for x in y_train:
        my_writer.writerow(x)

with open('x_p.csv', 'w', newline = '') as csvfile:
    my_writer = csv.writer(csvfile)
    my_writer.writerow(headers_x)
    for x in x_test:
        my_writer.writerow(x)

with open('y_p.csv', 'w', newline = '') as csvfile:
    my_writer = csv.writer(csvfile)
    my_writer.writerow(["target"])
    for x in y_test:
        my_writer.writerow(x)
    
    



