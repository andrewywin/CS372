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
color = model.predict(xnew)

color_dict = {1: "green", 2: "orange", 3: "purple"}
point_colors = [color_dict[s] for s in color]

sepal = np.empty(0)
petal = np.empty(0)
for i in xnew:
    sepal = np.append(sepal, i[0])
    petal = np.append(petal, i[1])

# Create the colormap
cmap = ListedColormap(['green', 'orange', 'purple'])

# Create the meshgrid
h = 0.1
x_min, x_max = xnew[:, 0].min() - 1, xnew[:, 0].max() + 1
y_min, y_max = xnew[:, 1].min() - 1, xnew[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the color of each point in the meshgrid
Z = np.array(model.predict(np.c_[xx.ravel(), yy.ravel()]))

# Reshape the predicted color array to match the meshgrid dimensions
Z = Z.reshape(xx.shape)

# Compute the average color of the points
point_colors_rgb = [plt.get_cmap()(c) for c in point_colors if c in color_dict.values()]
avg_color = np.mean(np.array(point_colors_rgb), axis=0)

# Create the scatter plot with the specified point colors and the decision boundary
fig, ax = plt.subplots()
ax.scatter(sepal, petal, c=point_colors, cmap=cmap)
ax.set_xlim(5, 35)
ax.set_ylim(-5, 20)
ax.set_xlabel("Sepal Area")
ax.set_ylabel("Petal Area")
ax.pcolormesh(xx, yy, Z, cmap=cmap)

# Set the background color of the plot to the average color of the points
ax.set_facecolor(avg_color)

plt.show()