import numpy as np
import pandas as pd

class KMeansManual:
    def __init__(self, n_clusters=3, max_iter=300, init='random'):
        self.n_clusters = n_clusters  
        self.max_iter = max_iter  
        self.init = init  
        self.centroids = None  
        self.labels_ = None 

    def fit(self, X):
        if self.init == 'random':
            random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.centroids = X.iloc[random_indices].to_numpy()
        elif self.init == 'k-means++':  # inisialisasi centroid menggunakan metode K-means++
            self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            distances = self._compute_distances(X)
            self.labels_ = np.argmin(distances, axis=1) # tentukan label cluster berdasarkan jarak terdekat
            centroids_old = self.centroids.copy() # simpan centroid sebelumnya untuk perbandingan konvergensi
            self.centroids = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)]) # perbarui centroid berdasarkan rata-rata titik dalam cluster
            if np.all(centroids_old == self.centroids):
                break  # hentikan iterasi jika konvergen

    def _initialize_centroids(self, X):
        centroids = [X.iloc[np.random.randint(X.shape[0])].to_numpy()]
        for _ in range(1, self.n_clusters):
            distances = self._compute_distances(X, centroids)  # hitung jarak dari centroid yang sudah ada
            probabilities = distances.min(axis=1) ** 2  
            probabilities /= probabilities.sum()  # normalisasi probabilitas
            centroid = X.iloc[np.random.choice(X.shape[0], p=probabilities)].to_numpy()
            centroids.append(centroid)
        return np.array(centroids)

    def _compute_distances(self, X, centroids=None):
        # hitung jarak dari setiap titik ke centroid
        if centroids is None:
            centroids = self.centroids
        return np.linalg.norm(X.to_numpy()[:, np.newaxis] - centroids, axis=2)

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
