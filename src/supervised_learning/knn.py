class KNN:
    def __init__(self, n_neighbors=3, metric='euclidean', p=3):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p  # Minkowski

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            predictions.append(self._predict(x))
        return predictions

    def _predict(self, x):
        # Calculate distances between x and all training samples
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

        # Sort distances and get indices of k nearest neighbors
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.n_neighbors]

        # Get labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common label among the neighbors
        return self._most_common_label(k_nearest_labels)

    def _euclidean_distance(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i]) ** 2
        return distance ** 0.5

    def _manhattan_distance(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            distance += abs(x1[i] - x2[i])
        return distance

    def _minkowski_distance(self, x1, x2, p):
        distance = 0
        for i in range(len(x1)):
            distance += abs(x1[i] - x2[i]) ** p
        return distance ** (1 / p)

    def _most_common_label(self, labels):
        label_count = {}
        for label in labels:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        # Get label with the highest count
        most_common = max(label_count, key=label_count.get)
        return most_common
