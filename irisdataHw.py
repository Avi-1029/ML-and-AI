import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error

data = pd.read_csv(r"C:\Users\Biju\Desktop\Jetlearn\ML and AI\Datasets\iris.csv")
print(data.info())

#Seperating the features from the target:
X = data[["sepal_length","sepal_width","petal_length","petal_width"]]
y = data["species"]

#Encoding the species:
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)

#Splitting the data into train and test:
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 56)

#Training the model:
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)
print(prediction)

#Error:
error = root_mean_squared_error(ytest, prediction)
print(error)

#Polynomial Regression:
from sklearn.preprocessing import PolynomialFeatures

polyfeatures = PolynomialFeatures(degree = 3)
D2X = polyfeatures.fit_transform(X)
print(X)
print(D2X)

xtrain, xtest, ytrain, ytest = train_test_split(D2X, y, test_size = 0.3, random_state = 56)

model1 = LinearRegression()
model1.fit(xtrain, ytrain)
predict = model1.predict(xtest)
print(predict)

error = root_mean_squared_error(ytest, predict)
print(error)
