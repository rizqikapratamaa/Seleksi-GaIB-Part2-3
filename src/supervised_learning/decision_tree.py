import numpy as np

class DecisionTreeManual:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def _build_tree(self, X, y, depth=0):
        # bangun pohon keputusan
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # kondisi berhenti
        if (depth >= self.max_depth) or (num_labels == 1) or (num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # mencari split terbaik
        best_split = self._best_split(X, y, num_features)
        if not best_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # bangun subtree kiri dan kanan
        left_subtree = self._build_tree(best_split['X_left'], best_split['y_left'], depth+1)
        right_subtree = self._build_tree(best_split['X_right'], best_split['y_right'], depth+1)
        return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree)

    def _best_split(self, X, y, num_features):
        # cari split terbaik berdasarkan Gini impurity
        best_gini = 1e9
        best_split = {}
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                # pisahkan data berdasarkan threshold
                X_left, X_right, y_left, y_right = self._split(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # hitung Gini impurity
                gini = self._gini(y_left, y_right)
                if gini < best_gini:
                    best_gini = gini
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'X_left': X_left,
                        'y_left': y_left,
                        'X_right': X_right,
                        'y_right': y_right
                    }
        return best_split if best_split else None

    def _split(self, X, y, feature_index, threshold):
        # pisahkan data berdasarkan threshold
        X_left = X[X[:, feature_index] <= threshold]
        y_left = y[X[:, feature_index] <= threshold]
        X_right = X[X[:, feature_index] > threshold]
        y_right = y[X[:, feature_index] > threshold]
        return X_left, X_right, y_left, y_right

    def _gini(self, y_left, y_right):
        # hitung Gini impurity
        def gini_impurity(y):
            m = len(y)
            return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

        m_left = len(y_left)
        m_right = len(y_right)
        m_total = m_left + m_right

        gini_left = gini_impurity(y_left)
        gini_right = gini_impurity(y_right)

        # hitung rata-rata dari Gini impurity
        return (m_left / m_total) * gini_left + (m_right / m_total) * gini_right

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        # telusuri pohon untuk prediksi
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def get_params(self, deep=True):
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        # cek apakah node adalah node daun
        return self.value is not None
