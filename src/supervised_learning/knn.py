import numpy as np
from collections import Counter

class KNN:
    def __init__(self, n_neighbors=3, metric='euclidean', p=3, weighted=False):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p  # Minkowski parameter
        self.weighted = weighted

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # prediksi kelas untuk setiap sampel di data uji
        predictions = []
        for x in X_test:
            predictions.append(self._predict(x))
        return predictions

    def _predict(self, x):
        # hitung jarak antara x dan seluruh sampel di data latih
        distances = []
        for x_train in self.X_train:
            if self.metric == 'euclidean':
                distance = self._euclidean_distance(x, x_train)
            elif self.metric == 'manhattan':
                distance = self._manhattan_distance(x, x_train)
            elif self.metric == 'minkowski':
                distance = self._minkowski_distance(x, x_train, self.p)
            else:
                raise ValueError("Invalid metric")
            distances.append(distance)

        # urutkan jarak dan mendapatkan indeks dari k tetangga terdekat
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.n_neighbors]

        # ambil label dari k tetangga terdekat
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]

        # return label yang paling umum di antara tetangga, dengan pembobotan berdasarkan jarak jika diaktifkan
        return self._most_common_label(k_nearest_labels, k_nearest_distances)

    def _euclidean_distance(self, x1, x2):
        # hitung jarak Euclidean antara dua titik
        return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))

    def _manhattan_distance(self, x1, x2):
        # hitung jarak Manhattan antara dua titik
        return np.sum(np.abs(np.array(x1) - np.array(x2)))

    def _minkowski_distance(self, x1, x2, p):
        # hitung jarak Minkowski antara dua titik dengan parameter p
        return np.sum(np.abs(np.array(x1) - np.array(x2)) ** p) ** (1 / p)

    def _most_common_label(self, labels, distances):
        # tentukan label paling umum di antara tetangga
        if self.weighted:
            # gunakan pembobotan dengan jarak terbalik
            weights = [1 / (d + 1e-5) for d in distances]  # tambahkan nilai kecil untuk menghindari pembagian dengan nol
            weighted_votes = {}
            for label, weight in zip(labels, weights):
                if label in weighted_votes:
                    weighted_votes[label] += weight
                else:
                    weighted_votes[label] = weight
            return max(weighted_votes, key=weighted_votes.get)
        else:
            # voting sederhana berdasarkan jumlah label terbanyak
            return Counter(labels).most_common(1)[0][0]

    def get_params(self, deep=False):
        return {
            "n_neighbors": self.n_neighbors,
            "metric": self.metric,
            "p": self.p,
            "weighted": self.weighted
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
