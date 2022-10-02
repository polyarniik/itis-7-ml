import math

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, X, K=5, max_iters=100, show_steps=False):
        self.X = X
        self.K = K
        self.max_iters = max_iters
        self.show_steps = show_steps
        self.n_samples, self.n_features = self.X.shape
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = self.init_centers(self.K)
        self.predict()

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def init_centers(self, k):
        x_mean = sum([x[0] for x in self.X]) / len(self.X)
        y_mean = sum([x[1] for x in self.X]) / len(self.X)

        R = -1000
        for x in self.X:
            distance = self.euclidean_distance(x, (x_mean, y_mean))
            R = max(distance, R)
            
        centroids = []
        for i in range(k):
            x_c = R * np.cos(2 * np.pi * i / k) + x_mean
            y_c = R * np.sin(2 * np.pi * i / k) + y_mean
            centroids.append(np.array([x_c, y_c]))
        return centroids

    def predict(self):
        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)

            if self.show_steps:
                self.plot()

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.show_steps:
                self.plot()

    def _get_cluster_labels(self):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(self.clusters):
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
        distances = [self.euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        # print(sample, closest_idx)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [self.euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for index in self.clusters:
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()


if __name__ == "__main__":
    X, y = make_blobs(n_samples=250)
    optimal_k = 2
    dists = []
    for k in range(2, 10):
        k_means = KMeans(X, K=k, max_iters=150, show_steps=False)
        labels = k_means._get_cluster_labels()
        dist_sum = 0
        
        for i in range(len(X)):
            centroid = k_means.centroids[math.ceil(labels[i])]

            dist_sum += ((abs(centroid[0]) - abs(X[i][0])) ** 2) + ((abs(centroid[1]) - abs(X[i][1])) ** 2)

        dists.append(dist_sum)

    for i in range(1, len(dists)): 
        if dists[i] / dists[i-1] < 0.5:
            optimal_k = i + 2
            break
            
    k_means = KMeans(X, K=optimal_k, max_iters=150, show_steps=False)
    k_means.plot()

