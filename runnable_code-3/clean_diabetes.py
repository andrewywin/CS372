import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np


def clean():

	diabetes = pd.read_csv("diabetes.csv")

	x = diabetes.drop(["Outcome"],axis = 1)
	y = diabetes["Outcome"]

	#train_X,test_X, train_y,test_y= train_test_split(np.array(x),np.array(y),test_size = .2)

	#training = pd.DataFrame(train_X)
	#training.columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

	imputer = SimpleImputer(strategy='mean',missing_values=0)
	x = imputer.fit_transform(np.array(x))
	#test_X = imputer.fit_transform(test_X)

	scaler = StandardScaler()
	x = scaler.fit_transform(x)
	#test_X = scaler.fit_transform(test_X)

	x = pd.DataFrame(x)
	x.columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

	#testing = pd.DataFrame(test_X)
	#testing.columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

	x["Outcome"] = y
	#testing["Outcome"] = test_y
	x.to_csv('clean_diabetes.csv', index=False, header=True)
	return
print(clean())

#https://www.kaggle.com/datasets/whenamancodes/predict-diabities