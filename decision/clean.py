import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
# Extract
loan_train = pd.read_csv('loan_sanction_train.csv')
loan_test = pd.read_csv('loan_sanction_test.csv')
loan_train_y = loan_train["Loan_Status"]
loan_train_x = loan_train.drop(["Loan_Status","Loan_ID"],axis = 1)
def clean(file):
	dependent = np.empty(0)
	for i in file["Dependents"]:
		cur = i
		if cur == "3+":
			cur = 3
		dependent = np.append(dependent, cur)
	# gender male = 1 female = 0
	gender = np.empty(0)
	for i in file["Gender"]:
		gen = i
		if gen == "Female":
			gen = 1
		if gen == "Male":
			gen = 0
		gender = np.append(gender,gen)
	#education graduate = 1 not graduate =0
	education = np.empty(0)
	for i in file["Education"]:
		edu = i
		if edu == "Graduate":
			edu = 1
		if edu == "Not Graduate":
			edu = 0
		education = np.append(education,edu)
	#self employed yes =1 no = 1, NULL/NAN = 1

	employment= np.empty(0)
	for i in file["Self_Employed"]:
		emp = i
		if i == "Yes":
			emp = 1
		if i  == "No":
			emp = 1
		employment = np.append(employment, emp)
	# Property Area urban = 1 semiurban = 2, rural = 3
	prop = np.empty(0)
	for i in file["Property_Area"]:
		if i == "Urban":
			prop = np.append(prop, 1)
		elif i == "Semiurban":
			prop = np.append(prop,2)
		elif i  == "Rural":
			prop = np.append(prop,3)
		else:
			prop = np.append(prop,i)
	#married yes = 1 no = 0
	married = np.empty(0)
	for i in file["Married"]:
		marry = i
		if marry ==  "Yes":
			marry = 1
		if marry == "No":
			marry = 1
		married = np.append(married,marry)
	train_df = pd.DataFrame({"Gender": gender, "Married": married, "Dependents": dependent, "Education": education, "Self_Employed": employment,  
							"ApplicantIncome": np.array(file["ApplicantIncome"]), "CoapplicantIncome": np.array(file["CoapplicantIncome"]),
							"LoanAmount": np.array(file["LoanAmount"]), "Loan_Amount_Term": np.array(file["Loan_Amount_Term"]),
							"Credit_History": np.array(file["Credit_History"]), "Property_Area": prop})
	#print(train_df)
	#train_df.to_csv("example.csv", index = True, header = True)
	return train_df
num_train = clean(loan_train_x)

num_test = clean(loan_test)
#print(num_train)
#print(num_test)

imputer = SimpleImputer(strategy='median')
num_train = imputer.fit_transform(num_train)
num_test = imputer.fit_transform(num_test)

# Standardize the data
scaler = StandardScaler()
num_train = scaler.fit_transform(num_train)
num_test = scaler.fit_transform(num_test)
# Load
# Save cleaned data to a new CSV file
cleaned_train = pd.DataFrame(num_train)
cleaned_test = pd.DataFrame(num_test)
cleaned_train["Loan_Status"] = loan_train_y
print(cleaned_train)
print(cleaned_test)
cleaned_train.to_csv('cleaned_train_loan.csv', index=True, header=True)
cleaned_test.to_csv('cleaned_test_loan.csv', index=True, header=True)
