import pandas as pd
from sklearn.model_selection import train_test_split
from my_random_forest_classifier import MyRandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Load the cleaned data
def run(filename="clean_diabetes.csv"):
	file = pd.read_csv(filename)
	x = file.drop(["Outcome"],axis = 1)
	y = file["Outcome"]
	train_X,test_X, train_y,test_y= train_test_split(np.array(x),np.array(y),test_size = .2)
	model = MyRandomForestClassifier()
	model.fit(train_X, train_y)
	my_random_forest_predictions = model.predict(np.array(test_X))

	target_names = ["Prediction", "True Values"]
	print(classification_report(my_random_forest_predictions, test_y, target_names=target_names))
	

run()