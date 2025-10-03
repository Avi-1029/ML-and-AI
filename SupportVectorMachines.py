from sklearn import datasets
import pandas as pd

cancerdictionary = datasets.load_breast_cancer()
print(cancerdictionary.keys())

X = pd.DataFrame(cancerdictionary.data, columns= cancerdictionary.feature_names)
print(X)

y = pd.Series(cancerdictionary.target)
print(y)

#Splitting the data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=24)

#deploying the model
from sklearn.svm import SVC
model = SVC(kernel= 'linear')
model.fit(xtrain, ytrain)
prediction = model.predict(xtrain)
print(prediction)

#Error checking
from sklearn.metrics import classification_report
print(classification_report(ytrain, prediction))

