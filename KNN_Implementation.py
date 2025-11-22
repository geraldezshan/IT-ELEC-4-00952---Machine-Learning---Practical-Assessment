import math
from collections import Counter
import numpy as np

def min_max_normalize(X):
    X = np.array(X, dtype=float)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1    # avoid divide-by-zero
    return (X - mins) / ranges, mins, ranges

def predict_knn(new_point, data, k=3, normalize=True):
    # Separate features and labels
    X = np.array([np.array(f) for f, _ in data])
    y = [label for _, label in data]
    new_point = np.array(new_point)

    # Normalization (recommended)
    if normalize:
        combined = np.vstack([X, new_point])
        combined_norm, mins, ranges = min_max_normalize(combined)
        X = combined_norm[:-1]        # training points
        new_point = combined_norm[-1] # normalized test point

    # Compute Euclidean distances
    distances = np.linalg.norm(X - new_point, axis=1)

    # Select k nearest neighbors
    k = min(k, len(data))
    nearest_ids = distances.argsort()[:k]

    # Majority vote
    neighbor_labels = [y[i] for i in nearest_ids]
    vote_counts = Counter(neighbor_labels)
    prediction = vote_counts.most_common(1)[0][0]

    return prediction

data = [
    ([1.0, 2.0], "A"),
    ([1.5, 1.8], "A"),
    ([5.0, 8.0], "B"),
    ([6.0, 8.0], "B"),
    ([1.2, 0.8], "A"),
    ([9.0, 11.0], "B")
]

new_point = [2.0, 2.0]

print("Predicted class:", predict_knn(new_point, data, k=3))
