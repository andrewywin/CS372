import pandas as pd
from sklearn.model_selection import train_test_split

def trial():
	iris = pd.read_csv("iris_data.csv")
	train,test = train_test_split(iris, test_size=0.2)
	train_X = train[["Sepal Area", "PetalArea"]]
	train_y = train[["Species"]]
	test_X = test[["Sepal Area", "PetalArea"]]
	test_y = test[["Species"]]
	return train_X, train_y, test_X, test_y

print(trial()[0])
print(trial()[1])
print(trial()[2])
print(trial()[3])