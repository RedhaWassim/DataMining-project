import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
import random
from sklearn.metrics import pairwise_distances

def euclidean_distance(row1, row2):
    if np.array_equal(row1,row2):
        return 0
    return np.sqrt(np.sum((row1-row2)**2))

def minkowski_distance(row1, row2):
    if np.array_equal(row1,row2):
        return 0
    return np.sqrt(np.sum((row1-row2)**2))

def cosine_distance(row1, row2):
    if np.array_equal(row1, row2):
        return 0
    cosine_similarity = np.dot(row1, row2) / (np.sqrt(np.sum(row1**2)) * np.sqrt(np.sum(row2**2)))
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def manhattan_distance(row1, row2):
    if np.array_equal(row1, row2):
        return 0
    return np.sum(np.abs(row1-row2))


class KMeans:
    def __init__(self, K=5, max_iters=100,strategy : str="euclidean",plot_steps = False):
        self.K = K
        self.max_iters = max_iters
        self.strategy=strategy
        self.plot_steps = plot_steps
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean vector) for each cluster
        self.centroids = []
    def _initialize_centroids(self, X):
        min_values = np.min(X, axis=0)
        max_values = np.max(X, axis=0)
        centroids = []

        for _ in range(self.K):
            centroid = [random.uniform(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]
            centroids.append(centroid)

        return np.array(centroids)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values  

        self.X = X
        self.n_samples, self.n_features = X.shape
        # initialize centroids randomly within the range of feature values
        self.centroids = self._initialize_centroids(X)

        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples) # a changer 
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        if self.strategy=="euclidean":
            distances =  [euclidean_distance(sample, point) for point in centroids]
        elif self.strategy=="minkowski":
            distances =  [minkowski_distance(sample, point) for point in centroids]
        elif self.strategy=="cosine":
            distances =  [cosine_distance(sample, point) for point in centroids]
        elif self.strategy=="manhattan":
            distances =  [manhattan_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        if self.strategy=="euclidean":
            distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        elif self.strategy=="minkowski":
            distances =  [minkowski_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        elif self.strategy=="cosine":
            distances =  [cosine_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        elif self.strategy=="manhattan":
            distances =  [manhattan_distance(centroids_old[i], centroids[i]) for i in range(self.K)]

        return sum(distances) == 0
    
    def dissimilarity_matrix(self):
        n = len(self.X)
        # Initialize the dissimilarity matrix with zeros
        dissimilarity_matrix = np.zeros((n, n))
        # Compute pairwise distances
        for i in range(n):
            for j in range(n):
                if self.strategy=="euclidean":
                    dissimilarity_matrix[i, j] = euclidean_distance(self.X[i], self.X[j]) 
                elif self.strategy=="minkowski":            
                    dissimilarity_matrix[i, j] = minkowski_distance(self.X[i], self.X[j]) 
                elif self.strategy=="cosine":            
                    dissimilarity_matrix[i, j] = cosine_distance(self.X[i], self.X[j]) 
                elif self.strategy=="manhattan":            
                    dissimilarity_matrix[i, j] = manhattan_distance(self.X[i], self.X[j]) 

        return dissimilarity_matrix

    def silhouette_coefficient(self):
        dissimilarity_matrix = self.dissimilarity_matrix()
        clusters = self.predict(self.X)
        n_samples = self.X.shape[0]

        silhouette_coefficients = np.zeros(n_samples)
        for i in range(n_samples):
            # Calculate the average dissimilarity of point i to all other points in the same cluster
            in_cluster_distance = np.sum(dissimilarity_matrix[i, clusters == clusters[i]])
            in_cluster_size = np.sum(clusters == clusters[i]) - 1  # Subtract 1 to exclude the point itself

            if in_cluster_size == 0:
                silhouette_coefficients[i] = 0
            else:
                # Calculate the average dissimilarity of point i to points in all other clusters
                dissimilarities_to_other_clusters = [
                    np.sum(dissimilarity_matrix[i, clusters == c]) / np.sum(clusters == c)
                    for c in range(self.K) if c != clusters[i]
                ]

                # Calculate silhouette coefficient for point i
                silhouette_coefficients[i] = (
                    np.min(dissimilarities_to_other_clusters) - in_cluster_distance / in_cluster_size
                ) / max(np.min(dissimilarities_to_other_clusters), in_cluster_distance / in_cluster_size)

        avg_silhouette_coefficient = np.mean(silhouette_coefficients)

        # print("Silhouette Coefficients:", silhouette_coefficients)
        # print("Average Silhouette Coefficient:", avg_silhouette_coefficient)

        return avg_silhouette_coefficient,silhouette_coefficients

    def davies_bouldin_index(self,X, labels):
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_centers = [np.mean(X[labels == i], axis=0) for i in range(n_clusters)]
        pairwise_distances_clusters = pairwise_distances(cluster_centers)

        max_avg_ratio = 0
        for i in range(n_clusters):
            ratios = []
            for j in range(n_clusters):
                if i != j:
                    ratios.append((pairwise_distances_clusters[i, i] + pairwise_distances_clusters[j, j]) /
                                  pairwise_distances_clusters[i, j])

            max_avg_ratio += max(ratios)

        return max_avg_ratio / n_clusters

    def calinski_harabasz(self ,X, labels):
    # Calculate the total number of data points and clusters
        n_samples, _ = X.shape
        k = len(np.unique(labels))
    
        # Calculate the centroid of the entire dataset
        overall_centroid = np.mean(X, axis=0)
    
        # Calculate the within-cluster dispersion matrix
        within_dispersion = 0
        for cluster_label in np.unique(labels):
            cluster_points = X[labels == cluster_label]
            cluster_centroid = np.mean(cluster_points, axis=0)
            within_dispersion += np.sum(pairwise_distances(cluster_points, [cluster_centroid])**2)
    
        # Calculate the between-cluster dispersion matrix
        between_dispersion = 0
        for cluster_label in np.unique(labels):
            cluster_points = X[labels == cluster_label]
            cluster_size = len(cluster_points)
            between_dispersion += cluster_size * pairwise_distances([overall_centroid], [np.mean(cluster_points, axis=0)])**2
    
        # Calculate the Calinski-Harabasz Index
        ch_index = (between_dispersion / (k - 1)) / (within_dispersion / (n_samples - k))
    
        return ch_index


    def inertia(self, X):
        labels = np.argmin(np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids]), axis=0)
        return np.sum([np.sum(np.linalg.norm(X[labels == k] - self.centroids[k], axis=1)**2) for k in range(self.K)])

    def plot(self, X):
        if self.centroids is None:
            raise Exception('You must fit the model first')

        pca = PCA(n_components=2)
        reduced_X = pca.fit_transform(X)
        reduced_centroids = pca.transform(self.centroids)

        sns.scatterplot(x=reduced_X[:, 0], y=reduced_X[:, 1], hue=self._get_cluster_labels(self.clusters), palette='bright')
        sns.scatterplot(x=reduced_centroids[:, 0], y=reduced_centroids[:, 1], color='black', marker='x', s=100)

        plt.show()

    def plot_thickness(self,X,labels):
        
        # Calculate silhouette scores for each sample
        silhouette_avg , sample_silhouette_values = self.silhouette_coefficient()
        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(8, 4)

        # The 1st subplot is the silhouette plot
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (self.K + 1) * 10])

        y_lower = 10
        for i in range(self.K):
            # Aggregate the silhouette scores for samples belonging to cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # Fill the silhouette plot
            color = plt.cm.nipy_spectral(float(i) / self.K)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for the next plot
            y_lower = y_upper + 10

        ax1.set_title(f"Silhouette Plot for {self.K} Clusters")
        ax1.set_xlabel("Silhouette Coefficient Values")
        ax1.set_ylabel("Cluster Label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--", label="Average Silhouette Score")
        ax1.legend()

        plt.show()