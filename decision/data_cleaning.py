import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
# Extract
loan_train = pd.read_csv('loan_sanction_train.csv')
loan_test = pd.read_csv('loan_sanction_test.csv')
loan_train_y = loan_train["Loan_Status"]
loan_train_x = loan_train.drop(["Loan_Status","Loan_ID"],axis = 1)
dependent = np.empty(0)
for i in loan_train_x["Dependents"]:
	cur = i
	if cur == "3+":
		cur = 3
	dependent = np.append(dependent, cur)
# gender male = 1 female = 0
gender = np.empty(0)
for i in loan_train_x["Gender"]:
	gen = i
	if gen == "Female":
		gen = 1
	if gen == "Male":
		gen = 0
	gender = np.append(gender,gen)
#education graduate = 1 not graduate =0
education = np.empty(0)
for i in loan_train_x["Education"]:
	edu = i
	if edu == "Graduate":
		edu = 1
	if edu == "Not Graduate":
		edu = 0
	education = np.append(education,edu)
#self employed yes =1 no = 1, NULL/NAN = 1

employment= np.empty(0)
for i in loan_train_x["Self_Employed"]:
	emp = i
	if i == "Yes":
		emp = 1
	if i  == "No":
		emp = 1
	employment = np.append(employment, emp)
# Property Area urban = 1 semiurban = 2, rural = 3
prop = np.empty(0)
for i in loan_train_x["Property_Area"]:
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
for i in loan_train_x["Married"]:
	marry = i
	if marry ==  "Yes":
		marry = 1
	if marry == "No":
		marry = 1
	married = np.append(married,marry)
train_df = pd.DataFrame({"Gender": gender, "Married": married, "Dependents": dependent, "Education": education, "Self_Employed": employment,  
						"ApplicantIncome": np.array(loan_train_x["ApplicantIncome"]), "CoapplicantIncome": np.array(loan_train_x["CoapplicantIncome"]),
						"LoanAmount": np.array(loan_train_x["LoanAmount"]), "Loan_Amount_Term": np.array(loan_train_x["Loan_Amount_Term"]),
						"Credit_History": np.array(loan_train_x["Credit_History"]), "Property_Area": prop, "Loan_Status":  np.array(loan_train_y)})
print(train_df)
train_df.to_csv("example.csv", index = True, header = True)
# Transform
# Replace missing values with median of the column
'''
imputer = SimpleImputer(strategy='median')
loan_train_x = imputer.fit_transform(loan_train_x)
loan_test = imputer.fit_transform(loan_test)

# Standardize the data
scaler = StandardScaler()
loan_train_x = scaler.fit_transform(loan_train_x)
loan_test = scaler.fit_transform(loan_test)
# Load
# Save cleaned data to a new CSV file
loan_train_x["Loan_Status"] = np.array(loan_train_y)
cleaned_train = pd.DataFrame(loan_train_x)
cleaned_test = pd.DataFrame(loan_test)
cleaned_train.to_csv('cleaned_train_loan.csv', index=False, header=False)
cleaned_test.to_csv('cleaned_test_loan.csv', index=False, header=False)

'''
