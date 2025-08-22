import pandas as pd
import numpy as np

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

#splitting the data:
from sklearn.model_selection import train_test_split
trainx, testx, trainy, testy = train_test_split(X, y, test_size= 0.3, random_state= 87)

#applying the KNNclassifier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(1)
model.fit(trainx, trainy)
prediction = model.predict(testx)
print(prediction)

#accuracy
from sklearn.metrics import confusion_matrix
print(confusion_matrix(testy, prediction))

from sklearn.metrics import classification_report
print(classification_report(testy, prediction))

#finding best value for K
max_k = int(np.sqrt(trainx.shape[0]))

#finding optimal K:
from sklearn.metrics import f1_score
scores = []
for k in range(1, max_k+1):
    model1 = KNeighborsClassifier(k)
    model1.fit(trainx, trainy)
    pred = model1.predict(testx)
    scores.append(f1_score(testy, pred, average='macro'))

print(scores)
maxscore = max(scores)
print(scores.index(maxscore) + 1)
