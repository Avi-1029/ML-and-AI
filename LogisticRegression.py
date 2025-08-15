import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Biju\Desktop\Jetlearn\ML and AI\Datasets\titanic.csv")
print(data.info())

X = data[["Pclass", "Sex", "Age", "Siblings/Spouses Aboard", "Parents/Children Aboard"]]
y = data["Survived"]

#Encoding the values:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

encoder = ColumnTransformer(transformers = [("encoder", OneHotEncoder(), [1])], remainder = 'passthrough')
x_encoded = encoder.fit_transform(X)
print(x_encoded)

#Splitting the data:
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x_encoded, y, test_size = 0.3, random_state = 97)

#Initating the model:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain, ytrain)
prediction = model.predict(xtest)
print(prediction)

#Evaluating the model:
from sklearn.metrics import confusion_matrix
results = confusion_matrix(ytest, prediction)

from sklearn.metrics import classification_report
print(classification_report(ytest, prediction))

sns.heatmap(results, annot= True, fmt = 'd')
plt.show()


