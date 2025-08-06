import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

sales = pd.read_csv(r"C:\Users\Biju\Desktop\Jetlearn\ML and AI\Datasets\Salary.csv")
print(sales.info())

x = sales["YearsExperience"]
y = sales["Salary"]

x = x.values.reshape(-1,1)

model = LinearRegression()

model.fit(x, y)

predict = model.predict(x)
print(predict)

error = root_mean_squared_error(y, predict)
print(error)

