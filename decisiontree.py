import pandas as pd

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
