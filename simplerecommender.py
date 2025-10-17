import pandas as pd

dataset = pd.read_csv(r"C:\Users\Biju\Desktop\Jetlearn\ML and AI\Datasets\movies_metadata.csv")
print(dataset.info())

#(v/(v+m))*R + (m/(v+m))*C
#v = vote count, m = minimum amount of votes required to be listed or considered, R = Vote Average, C = Average votes across all movies in the whole platform

C = dataset["vote_average"].mean()
v = dataset["vote_count"]
R = dataset["vote_average"]
m = dataset["vote_count"].quantile(0.95)
#print(m)

dataset["Weighted_Rating"] = (v/(v+m))*R + (m/(v+m))*C

print(dataset)

sorteddata = dataset.sort_values(by = "Weighted_Rating", ascending= False)
print(sorteddata)

print(sorteddata["title"].head(20))

