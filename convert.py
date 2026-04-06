import urllib.request
import pandas as pd

url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv"
urllib.request.urlretrieve(url, "heart.csv")
df = pd.read_csv("heart.csv")
print("heart.csv created! Rows:", len(df))
print(df.head())