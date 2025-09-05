import pandas as pd
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv(r"C:\Users\Biju\Desktop\Jetlearn\ML and AI\Datasets\car.csv")
data.info()
print(data)
print(data["class"].value_counts())

X = data.drop(columns = "class")
y = data["class"]

#sampling:
from imblearn.over_sampling import RandomOverSampler
sampler = RandomOverSampler(sampling_strategy= "not majority", random_state=48)
X, y = sampler.fit_resample(X, y)
print(y.value_counts())

#encoding:
encoder = LabelEncoder()

columns = X.columns

for column in columns:
    X[column] = encoder.fit_transform(X[column])

print(X)

y = encoder.fit_transform(y)
print(y)

#splitting the data:
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size= 0.3, random_state= 5)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
result = model.predict(xtest)
print(result)

#testing accuracy
from sklearn.metrics import classification_report
print(classification_report(ytest, result))

