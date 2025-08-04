import pandas as pd
import numpy as np

boston = pd.read_csv(r"C:\Users\Biju\Desktop\Jetlearn\ML and AI\Datasets\HousingData.csv")
print(boston.info())

#Preprocessing data:
X = boston[["RM" , "LSTAT"]]
y = boston["MEDV"]
print(X.isnull().sum())

#Removing the missing data
X.dropna(inplace = True)
print(X.isnull().sum())

#Spliting the data (Training and testing)
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 56)
print(xtrain)
print(xtest)
print(ytrain)
print(ytest)
