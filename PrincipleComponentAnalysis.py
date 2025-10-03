from sklearn import datasets
import pandas as pd

cancerdictionary = datasets.load_breast_cancer()
print(cancerdictionary.keys())

X = pd.DataFrame(cancerdictionary.data, columns= cancerdictionary.feature_names)
print(X.info())

y = pd.Series(cancerdictionary.target)
print(y)

#Scaling the data:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaledX = scaler.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components= 3)
analysedX = pca.fit_transform(scaledX)

print(analysedX)