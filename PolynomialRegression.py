import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

#x = np.arange(1,21)
# y = x**2 + 3

# plt.scatter(x, y, marker = "*")
# plt.show()

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

#Transforming features to higher degree
from sklearn.preprocessing import PolynomialFeatures
bob = PolynomialFeatures(degree = 2)
D2X = bob.fit_transform(X)
print(X)
print(D2X)

#Spliting the data (Training and testing)

xtrain, xtest, ytrain, ytest = train_test_split(D2X, y, test_size = 0.3, random_state = 56)

model = LinearRegression()
model.fit(xtrain, ytrain)
predict = model.predict(xtest)
print(predict)
print("Gradient = ", model.coef_, ", Intercept = ", model.intercept_)

error = root_mean_squared_error(ytest, predict)
print(error)
