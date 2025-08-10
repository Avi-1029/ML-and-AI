import pandas as pd

data = pd.read_csv(r"C:\Users\Biju\Desktop\Jetlearn\ML and AI\Datasets\Data.csv")
print(data.info())

#Replacing the missing valueswith the appropriate statistical value
from sklearn.impute import SimpleImputer 
bob = SimpleImputer(missing_values= pd.NA , strategy= "mean")
data[["Age" , "Salary"]] = bob.fit_transform(data[["Age" , "Salary"]])
#print(data)

#Seperating feature and target:
x = data[["Country","Age","Salary"]]
y = data["Purchased"]

#Encoding catagorical values in features:
x = pd.get_dummies(x, columns = ["Country"], dtype= int)
print(x)

#Encoding the target values:
from sklearn.preprocessing import LabelEncoder
tom = LabelEncoder()
y = tom.fit_transform(y)
print(y)
