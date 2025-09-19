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

#Finding the corelation between features and target:
cormatrix = data.corr()
print(cormatrix)
sns.heatmap(cormatrix, annot = True)
plt.show()

#Deciding features nd target:
X = data[[]]

