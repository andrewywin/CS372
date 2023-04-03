#Visualization for decision_vis.pdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_decision_tree_classifier import MyDecisionTreeClassifier
from matplotlib.colors import ListedColormap

iris = pd.read_csv("iris_data.csv")
x = np.array(iris.drop(["Species"],axis = 1))
y = np.array(iris["Species"])

model = MyDecisionTreeClassifier()

model.fit(x,y)
x = iris.drop(["Species"],axis = 1)
xnew = x.sample(n = 10000,replace = True)
xnew = np.array(xnew)

#print(len(xnew))
color = model.predict(xnew)


#plt.scatter(xnew[0],xnew[1])


#plt.show()

sepal = np.empty(0)
petal = np.empty(0)
for i in xnew:
	sepal =np.append(sepal, i[0])
	petal = np.append(petal, i[1])
plt.scatter(sepal,petal, c = color,cmap = ListedColormap(['green', 'orange', 'purple']))
plt.xlim(5,35)
plt.ylim(-5,20)
plt.xlabel("Sepal Area")
plt.ylabel("Petal Area")
plt.show()

x = np.linspace(-5,35,1000)