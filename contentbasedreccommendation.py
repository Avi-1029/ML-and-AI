import pandas as pd

dataset = pd.read_csv(r"C:\Users\Biju\Desktop\Jetlearn\ML and AI\Datasets\movies_metadata.csv")
print(dataset.info())

new_data = dataset[["overview", "title"]]
new_data = new_data.dropna()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")
bob = vectorizer.fit_transform(new_data["overview"])
print(bob.shape)

