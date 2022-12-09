import matplotlib.pyplot as plt
import sys
import pandas as pd
import csv



def F(w, X, y):
	return sum((w * x - y)**2 for x, y in zip(X, y))/len(y)


def dF(w, X, y):
	return sum(2*(w * x - y) * x for x, y in zip(X, y))/len(y)


def print_line(points, w, iteration, line_color = None, line_style = 'dotted'):
	list_x = []
	list_y = []
	for index, tuple in enumerate(points):
		x = tuple[0]
		y = x * w
		list_x.append(x)
		list_y.append(y)
	ax1.text(x,y, iteration, horizontalalignment='right')
	ax1.plot(list_x, list_y, color = line_color, linestyle= line_style)

if __name__=='__main__':
	X = [65,150,120,250,70,180,130,200,150,100,230,40,80,240,90]
	y = [2,3.5,1.8,4.2,3,3.7,2.8,5,4.5,2.5,4.7,3.6,3.2,5,3.5]

	#dataset = pd.read_csv("dataset_ejercicio_I_regresion_lineal.csv")	
	
	from sklearn.model_selection import train_test_split
	x_train , x_Test ,y_train , y_test = train_test_split(X,y,test_size= 0.1, random_state= 0,shuffle = True)

	list_error = []
	list_w = []	
	iterations = int(sys.argv[1])
	
	fig = plt.figure(figsize=(15, 9))
	ax1 = fig.add_subplot(1, 2, 1)
	ax1.set_title("Linear regression")
	ax1.set(xlabel="size", ylabel="price")
	ax2 = fig.add_subplot(1, 2, 2)
	ax2.set_title("Loss function")
	ax2.set(xlabel="weight", ylabel="error")
	
	ax1.scatter(x_Test, y_test)
	
	w = 0.02265
	alpha = 0.00001
	# ~ alpha = 0.05 #Efecto similar al de no sacar el promedio
	for t in range(iterations):
		error = F(w, x_Test, y_test)
		gradient = dF(w, x_Test, y_test)
		print ('gradient = {}'.format(gradient))
		ax2.scatter(w, error)
		ax2.text(w, error, t, horizontalalignment='right')
		list_w.append(w)
		list_error.append(error)
		
		w = w - alpha * gradient
		print ('iteration {}: w = {}, F(w) = {}'.format(t, w, error))
		print_line(zip(x_Test, y_test), w, t)
			
	print_line(zip(x_Test, y_test), w, t, 'red', 'solid')
	ax2.plot(list_w, list_error, color = 'red', linestyle = 'solid')
	
	plt.show()
