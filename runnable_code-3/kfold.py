import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_decision_tree_classifier import MyDecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold

iris = pd.read_csv("iris_data.csv")
x = np.array(iris.drop(["Species"], axis=1))
y = np.array(iris["Species"])

# create 5-fold cross-validation object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# create a single decision tree
model = MyDecisionTreeClassifier()

# loop over the folds
for fold, (train_idx, test_idx) in enumerate(kf.split(x)):
    # split the data into training and test sets
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # train the model on the training set
    model.fit(x_train, y_train)

    # create mesh grid
    
    xx, yy = np.meshgrid(np.linspace(5, 35, 1000),np.linspace(-5, 20, 1000))

    # predict class for each point in mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)

    # plot mesh grid and data points
    cmap = ListedColormap(['green', 'orange', 'purple'])
    plt.subplot(2, 3, fold+1)
    plt.pcolormesh(xx, yy, Z, cmap=cmap)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=model.predict(x_test), edgecolors='k', cmap=cmap)
    plt.xlim(5, 35)
    plt.ylim(-5, 20)
    plt.xlabel("Sepal Area")
    plt.ylabel("Petal Area")
    plt.title(f"Fold {fold+1}")

plt.tight_layout()
plt.show()