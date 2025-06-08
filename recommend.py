import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
dataframe = pd.merge(ratings, movies, on="movieId")
dataframe["genre"] = dataframe["genres"].apply(lambda x: x.split('|')[0])
le = LabelEncoder()
dataframe["genreencoded"] = le.fit_transform(dataframe["genre"])
features = dataframe[["userId", "movieId", "rating", "genreencoded"]]
train_data, test_data = train_test_split(features, test_size=0.2, random_state=42)
scaler = StandardScaler()
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)
X_train = scaled_train[:, [0, 1, 3]]
y_train = scaled_train[:, 2]

def gradient_descent(X, y, lr=0.01, epochs=100):
    m, n = X.shape
    weights = np.ones(n)
    for i in range(epochs):
        predictions = X @ weights
        error = predictions - y
        gradient = (1/m) * X.T @ error
        weights = weights - lr * gradient
    return weights

weights = gradient_descent(X_train, y_train)
X_train_weighted = X_train * weights
knn = NearestNeighbors(n_neighbors=15)
knn.fit(X_train_weighted)

sample = test_data.iloc[100]
sample_scaled = scaler.transform([sample])
sample_features = sample_scaled[:, [0, 1, 3]]
sample_weighted = sample_features * weights
distances, indices = knn.kneighbors(sample_weighted)

neighbor_ratings = []
print("Recommended Movies and Neighbor Ratings:")
for i, dist in zip(indices[0], distances[0]):
    neighbor = train_data.iloc[i]
    movie_Id = int(neighbor["movieId"])
    actual_rating = neighbor["rating"]
    neighbor_ratings.append(actual_rating)
    title = movies[movies["movieId"] == movie_Id]["title"].values[0]
    print(f"Movie: {title}, Neighbor Rating: {actual_rating:.2f}, Distance: {dist:.2f}")
print("\nAverage Neighbor Rating:", np.mean(neighbor_ratings))
