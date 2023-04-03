import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


abalone = pd.read_csv("abalone.csv", header = None)
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', "WholeWt", "ShuckedWt", "VisceraWt", "ShellWt", "Rings"]

def preprocess(df):
    numeric = df.iloc[:, 0]
    #print(numeric)
    numeric_array_female = np.empty(0)
    numeric_array_male = np.empty(0)
    numeric_array_infant = np.empty(0)
    for i in numeric:
        if i == "M":
            numeric_array_female = np.append(numeric_array_female, 0)
            numeric_array_male = np.append(numeric_array_male, 1)
            numeric_array_infant = np.append(numeric_array_infant, 0)
        elif i == "F":
            numeric_array_female = np.append(numeric_array_female, 1)
            numeric_array_male = np.append(numeric_array_male, 0)
            numeric_array_infant = np.append(numeric_array_infant, 0)
        else:
            numeric_array_female = np.append(numeric_array_female, 0)
            numeric_array_male = np.append(numeric_array_male, 0)
            numeric_array_infant = np.append(numeric_array_infant, 1)
    #print(numeric_array_female, numeric_array_male, numeric_array_infant)
    return numeric_array_female, numeric_array_male, numeric_array_infant
#print(preprocess(abalone))
abalone["IsFemale"] = preprocess(abalone)[0].astype(int)
abalone["IsMale"] = preprocess(abalone)[1].astype(int)
abalone["IsInfant"] = preprocess(abalone)[2].astype(int)
abalone = abalone[["IsFemale", "IsMale", "IsInfant", 'Length', 'Diameter', 'Height', "WholeWt", "ShuckedWt", "VisceraWt", "ShellWt", "Rings"]]


train, test = train_test_split(abalone, test_size=0.1)
#print(train)
#print(test)
train.to_csv('abalone_train.csv', index=False)
test.to_csv('abalone_test.csv', index=False)