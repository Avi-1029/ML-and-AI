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
x_dummies = pd.get_dummies(x, columns = ["Country"], dtype= int)
print(x_dummies)

#Encoding using onehotencoder:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ben = ColumnTransformer(transformers = [("encoder", OneHotEncoder(), [0])], remainder = 'passthrough')
x_transformed = ben.fit_transform(x)
print(x_transformed)

filtered_france = data[data["Country"] == "France"]
filtered_dummies = pd.get_dummies(filtered_france, columns = ["Country"], dtype= int)
print(filtered_dummies)

print(ben.transform(filtered_france))

#Encoding the target values:
from sklearn.preprocessing import LabelEncoder
tom = LabelEncoder()
y = tom.fit_transform(y)
print(y)

#Scaling:
#Standard Scaling : (Xi-Xmean)/Xstd:
from sklearn.preprocessing import StandardScaler
# tim = StandardScaler()
# x_dummies[["Age","Salary"]] = tim.fit_transform(x_dummies[["Age", "Salary"]])
# print(x_dummies)

#MinMax Scaling : (values range from 0 and 1) : (X - X_min) / (X_max - X_min):
from sklearn.preprocessing import MinMaxScaler
sam = MinMaxScaler()
x_dummies[["Age", "Salary"]] = sam.fit_transform(x_dummies[["Age", "Salary"]])
print(x_dummies)

