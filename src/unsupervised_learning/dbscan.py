import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class DBSCANManual:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', p=2):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.p = p
        self.labels_ = None

    def fit(self, X):
        X = X.to_numpy()
        n_samples = X.shape[0]
        self.labels_ = -1 * np.ones(n_samples)  # label -1 digunakan untuk noise (data yang tidak masuk cluster)
        visited = np.zeros(n_samples, dtype=bool)  # tandai apakah titik sudah dikunjungi
        cluster_id = 0  # inisialisasi ID cluster

        for i in range(n_samples):
            if visited[i]:  # jika titik sudah dikunjungi, lewati
                continue
            visited[i] = True  # tandai titik sudah dikunjungi
            neighbors = self._region_query(X, i)  # cari tetangga dalam radius eps

            if len(neighbors) < self.min_samples:  # jika jumlah tetangga kurang dari min_samples, tandai sebagai noise
                self.labels_[i] = -1
            else:  # jika cukup tetangga, mulai cluster baru
                cluster_id += 1
                self._expand_cluster(X, i, neighbors, cluster_id, visited)  # Perluas cluster

    def _region_query(self, X, point_idx):
        if self.metric == 'minkowski':
            distances = cdist(X[point_idx:point_idx+1], X, metric=self.metric, p=self.p).flatten()
        elif self.metric == 'manhattan':
            # normalisasi data untuk mengatasi perbedaan skala jika menggunakan manhattan
            X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-6)
            distances = cdist(X_normalized[point_idx:point_idx+1], X_normalized, metric='cityblock').flatten()  # Manhattan distance
        else:
            distances = cdist(X[point_idx:point_idx+1], X, metric=self.metric).flatten()
        
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        queue = list(neighbors)
        self.labels_[point_idx] = cluster_id  # tetapkan label cluster untuk titik awal
        
        while queue:
            neighbor_idx = queue.pop(0)

            if not visited[neighbor_idx]:  # jika tetangga belum dikunjungi
                visited[neighbor_idx] = True  # tandai sebagai sudah dikunjungi
                new_neighbors = self._region_query(X, neighbor_idx)
                
                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors) 

            if self.labels_[neighbor_idx] == -1:  # jika tetangga belum masuk cluster mana pun
                self.labels_[neighbor_idx] = cluster_id  # tetapkan ke cluster_id saat ini