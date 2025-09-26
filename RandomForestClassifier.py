import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv(r"C:\Users\Biju\Desktop\Jetlearn\ML and AI\Datasets\adult_income.csv", sep = ", ")
print(data.info())
print(data)

#encoding:
list = ["workclass","income","native-country","gender","race","relationship","occupation","marital-status","education"]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for i in list: 
    data[i] = encoder.fit_transform(data[i])
#print(data.info())

#print(data[["education","educational-num"]])

#Finding the corelation between features and target:
cormatrix = data.corr()
print(cormatrix)
sns.heatmap(cormatrix, annot = True)
plt.show()

#Deciding features nd target:
X = data[["age", "workclass", "educational-num", "marital-status" ,"occupation", "relationship" ,"gender", "race" ,"hours-per-week"]]
y = data["income"]

#Oversampling to balance the data
from imblearn.over_sampling import RandomOverSampler
sampler = RandomOverSampler(sampling_strategy= "auto", random_state=24)
resampledx, resampledy = sampler.fit_resample(X, y)

#Splitting the data:
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(resampledx, resampledy, test_size = 0.3, random_state= 24)

#Initiating the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain, ytrain)
prediction = model.predict(xtest)
print(prediction)

#Accuracy of the model
from sklearn.metrics import classification_report
print(classification_report(ytest, prediction))
