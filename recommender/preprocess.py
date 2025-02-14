import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
movies = pd.read_csv('data/movies.csv')

movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))

def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)  
    return int(match.group(1)) if match else np.nan

movies["year"] = movies["title"].apply(extract_year)

movies.dropna(subset=["year", "genres"], inplace=True)
# one hot encoding
genres_list = sorted(set([genre for sublist in movies["genres"] for genre in sublist]))
for genre in genres_list:
    movies[genre] = movies["genres"].apply(lambda x: 1 if genre in x else 0)

print(movies.head())

movies.to_csv('data/movies_processed.csv', index=False)