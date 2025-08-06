import pandas as pd
import numpy as np

boston = pd.read_csv(r"C:\Users\Biju\Desktop\Jetlearn\ML and AI\Datasets\HousingData.csv")
print(boston.info())

boston = boston[["RM", "LSTAT", "MEDV"]]

#Removing the missing data
boston.dropna(inplace = True)
print(boston.isnull().sum())

#Preprocessing data:
X = boston[["RM" , "LSTAT"]]
y = boston["MEDV"]
print(X.isnull().sum())

#Spliting the data (Training and testing)
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 56)
print(xtrain)
print(xtest)
print(ytrain)
print(ytest)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(xtrain, ytrain)

predict = model.predict(xtest)
print(predict)

from sklearn.metrics import root_mean_squared_error

error = root_mean_squared_error(ytest, predict)
print(error)
print("Gradient = ", model.coef_, ", Intercept = ", model.intercept_)


