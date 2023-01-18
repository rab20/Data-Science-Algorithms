# Demonstrate various Feature Selection Techniques on the "Covid" dataset
# URL for reference - https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/

import pandas as pd
import numpy as np
 
df = pd.read_csv("Covid_data.csv")
print(df)

# Information Gain - Filter Method
# It can be used for feature selection by evaluating the Information gain of each variable in the context of the target variable.
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# Encode the Categorical data of Gender - using Dummy Column - Creates a new column called Gender_M
# Use Dummy Variables
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

print(df.columns)
print(df.columns[0])
# Columns - Age [0], Co_Morbid [1], Admit_date [2], Discharge_date [3], Remdesevir_given [4], 
# DaysofStay [5], DischargeType [6], Covid_Severity [7], 
# Covid_SeverityDescription [8], DischargeTypeCategorical [9], Gender_M [10]

X = df.iloc[:,[0,1,4,5,7,10]]   #Co_Morbid, Remdesevir_given, DaysofStay, Covid_Severity
Y = df.iloc[:,[6]]  #DischargeType

importances = mutual_info_classif(X,Y)
feat_importances = pd.Series(importances,index=[0,1,4,5,7,10])
feat_importances.plot(kind='barh', color='teal')
plt.show()
