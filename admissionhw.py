import pandas as pd
data = pd.read_csv(r"C:\Users\Biju\Desktop\Jetlearn\ML and AI\Datasets\admission.csv")
print(data.info())
print(data.head(5))

#defining features and target:
X = data[["gre","gpa","rank"]]
y = data["admit"]

#scaling:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X["gre"] = scaler.fit_transform(X[["gre"]])
print(X)

#splitting data:
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=80)

#initiating the model:
from sklearn.linear_model import LogisticRegression
model =  LogisticRegression()
model.fit(xtrain, ytrain)
prediction = model.predict(xtest)
print(prediction)

#accuracy
from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, prediction))

from sklearn.metrics import classification_report
print(classification_report(ytest, prediction))