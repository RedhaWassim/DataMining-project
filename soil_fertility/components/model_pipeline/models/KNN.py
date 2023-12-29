import numpy as np
import pandas as pd

from collections import Counter


def euclidean_distance(row1, row2):
    if np.array_equal(row1, row2):
        return 0
    return np.sqrt(np.sum((row1 - row2) ** 2))


def minkowski_distance(row1, row2):
    if np.array_equal(row1, row2):
        return 0
    return np.sqrt(np.sum((row1 - row2) ** 2))


def cosine_distance(row1, row2):
    if np.array_equal(row1, row2):
        return 0
    return np.dot(row1, row2) / (
        np.sqrt(np.sum(row1**2)) * np.sqrt(np.sum(row2**2))
    )


def manhattan_distance(row1, row2):
    if np.array_equal(row1, row2):
        return 0
    return np.sum(np.abs(row1 - row2))


def Hamming_distance(row1, row2):
    if np.array_equal(row1, row2):
        return 0
    return np.sum(row1 != row2)


class KNN:
    def __init__(self, k: int = 5, strategy: str = "euclidean"):
        self.k = k
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = X
        self.y_train = y

    def predict(self, X: pd.DataFrame):
        predictions = X.apply(lambda x: self._calculate(x), axis=1)
        return np.array(predictions)

    def _calculate(self, row: pd.Series):
        if self.strategy == "euclidean":
            distances = self.X_train.apply(
                lambda row1: euclidean_distance(row1, row), axis=1
            )
        elif self.strategy == "minkowski":
            distances = self.X_train.apply(
                lambda row1: minkowski_distance(row1, row), axis=1
            )
        elif self.strategy == "cosine":
            distances = self.X_train.apply(
                lambda row1: cosine_distance(row1, row), axis=1
            )
        elif self.strategy == "manhattan":
            distances = self.X_train.apply(
                lambda row1: manhattan_distance(row1, row), axis=1
            )
        elif self.strategy == "Hamming":
            distances = self.X_train.apply(
                lambda row1: Hamming_distance(row1, row), axis=1
            )

        k_indices = np.argsort(distances)[: self.k]
        k_neighbor_labels = self.y_train.iloc[k_indices].tolist()

        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
