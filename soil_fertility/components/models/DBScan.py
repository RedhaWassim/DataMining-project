import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
import seaborn as sns
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

class DBScan:
    def __init__(self, min_samples:int,eps =float ,strategy : str="euclidean"):
        self.min_samples = min_samples
        self.eps = eps
        self.strategy=strategy
    def calculate_distance(self,X, point_index, y_index):
        if self.strategy=="euclidean":
            distance = euclidean_distance(X[point_index], X[y_index]) 
        elif self.strategy=="minkowski":
            distance =  minkowski_distance(X[point_index], X[y_index]) 
        elif self.strategy=="cosine":
            distance =  cosine_distance(X[point_index], X[y_index]) 
        elif self.strategy=="manhattan":
            distance =  manhattan_distance(X[point_index], X[y_index]) 
        else:
            raise ValueError(f"Unsupported distance metric: {self.strategy}")
        return distance
    def get_nieghbors(self,point_index:int,X:np.ndarray):
        nieghbors_index = []
        for y_index ,point_y in enumerate(X):
            distance = self.calculate_distance(X, point_index, y_index)
            if y_index != point_index and distance <= self.eps :
                nieghbors_index.append(y_index)
        return(nieghbors_index)

    def is_core(self, point_index:int,X:np.array):
        return len(self.get_nieghbors(point_index,X)) >= self.min_samples
    def visit_neighbors(self, point_index:int, X:np.array,cluster_index=int):
        for neighbor_index in self.get_nieghbors(point_index,X):
            if self.cluster_per_points[neighbor_index] == -1:
                self.cluster_per_points[neighbor_index] = cluster_index
                self.visit_neighbors(neighbor_index, X, cluster_index)

    def predict(self, X :np.ndarray):
        if isinstance(X, pd.DataFrame):
            X = X.values 
        cluster_index = 0
        self.cluster_per_points = [-1]*len(X)
        for x_index,point_x in enumerate(X):
            if self.cluster_per_points[x_index] != -1 :
                continue
            if self.is_core(x_index,X) :
                self.cluster_per_points[x_index]=cluster_index
                # loop through all the neighbors
                self.visit_neighbors(x_index,X,cluster_index)
                cluster_index += 1
        return self.cluster_per_points
    
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
    
    def calinski_harabasz(self,X, labels):
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
    def plot(self, X):
        plt.figure(figsize=(8, 6))
        if self.cluster_per_points is None:
            raise Exception('You must fit the model first')
        pca = PCA(n_components=2)
        reduced_X = pca.fit_transform(X)

   
        for label in np.unique(self.cluster_per_points):
            if label == -1:
                outliers_indices = np.where(np.array(self.cluster_per_points) == label)
                plt.scatter(reduced_X[outliers_indices, 0], reduced_X[outliers_indices, 1], c='black', marker='x', label=f'Cluster {label} (Outliers)')
            else:
                cluster_indices = np.where(np.array(self.cluster_per_points) == label)
                plt.scatter(reduced_X[cluster_indices, 0], reduced_X[cluster_indices, 1], label=f'Cluster {label}')

        plt.legend()
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