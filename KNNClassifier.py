import pandas as pd

data = pd.read_csv(r"C:\Users\Biju\Desktop\Jetlearn\ML and AI\Datasets\iris.csv")
#print(data.info())

#seperating features and target:
X = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = data["species"]

#encoding:
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)

#scaling:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(X)