import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_decision_tree_classifier import MyDecisionTreeClassifier
from matplotlib.colors import ListedColormap

iris = pd.read_csv("iris_data.csv")
x = np.array(iris.drop(["Species"], axis=1))
y = np.array(iris["Species"])

model = MyDecisionTreeClassifier()
model.fit(x, y)


xnew = x[np.random.choice(x.shape[0], 10000, replace=True)]


xx, yy = np.meshgrid(np.linspace(5, 35, 1000),np.linspace(-5, 20, 1000))


Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array(Z)
Z = Z.reshape(xx.shape)


plt.xlim(5, 35)
plt.ylim(-5, 20)


cmap = ListedColormap(['green', 'orange', 'purple'])
plt.pcolormesh(xx, yy, Z, cmap=cmap)
plt.scatter(xnew[:, 0], xnew[:, 1], c=model.predict(xnew), edgecolors='k', cmap=cmap)
plt.xlabel("Sepal Area")
plt.ylabel("Petal Area")
plt.show()