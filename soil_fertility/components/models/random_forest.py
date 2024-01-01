import numpy as np
from soil_fertility.components.models.decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=15, max_depth=10, max_features=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names

        for _ in range(self.n_trees):
            # Randomly sample data with replacement (bootstrapping)
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_bootstrap = X[indices, :]
            y_bootstrap = y[indices]
 
            # Randomly select a subset of features if max_features is specified
            if self.max_features is not None:
                selected_features = np.random.choice(
                    X.shape[1], self.max_features, replace=False
                )
                X_bootstrap = X_bootstrap[:, selected_features]

            # Train a Decision Tree on the bootstrapped data
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap, feature_names=self.feature_names)
            self.trees.append(tree)

    def predict(self, X):
        # Ensure that the input array aligns with the selected features during training
        X_subset = X[:, self.feature_names] if self.feature_names is not None else X

        # Make predictions using each tree and return the majority vote
        predictions = np.array([tree.predict(X_subset) for tree in self.trees])

        # Explicitly cast to integers before applying np.bincount
        predictions = predictions.astype(int)

        # Apply np.bincount along the columns to get the majority vote
        majority_votes = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )

        return majority_votes

    def get_params(self, deep=True):
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
