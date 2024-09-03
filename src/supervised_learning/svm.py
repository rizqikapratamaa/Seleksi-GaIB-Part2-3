import numpy as np

class SVMManual:
    def __init__(self, learning_rate=0.001, C=1.0, n_iters=1000, kernel='linear'):
        self.learning_rate = learning_rate
        self.C = C  # parameter regularisasi
        self.n_iters = n_iters
        self.kernel = kernel
        self.w = None
        self.b = None

    def _kernel(self, X1, X2):
        # fungsi kernel untuk mendukung kernel linear dan RBF (Radial Basis Function)
        if self.kernel == 'linear':
            return np.dot(X1, X2)
        elif self.kernel == 'rbf':
            gamma = 0.1  # nilai gamma bisa dituning
            return np.exp(-gamma * np.linalg.norm(X1 - X2) ** 2)
        else:
            raise ValueError("Unknown kernel")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1) # konversi label dari 0/1 menjadi -1/1 sesuai dengan SVM
        
        self.w = np.zeros(n_features) # inisialisasi weight dan bias
        self.b = 0

        # implementasi Stochastic Gradient Descent (SGD) untuk optimasi SVM
        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # periksa apakah kondisi margin sudah terpenuhi
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.C * self.w) # bila margin terpenuhi, hanya menerapkan L2 regularization
                else:
                    self.w -= self.learning_rate * (2 * self.C * self.w - np.dot(x_i, y_[idx])) # bila margin tidak terpenuhi, update weight dan bias
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b # prediksi kelas berdasarkan tanda dari hasil dot product antara input X dan weight
        return np.sign(approx)
    
    def get_params(self, deep=True):
        return {
            "learning_rate": self.learning_rate,
            "C": self.C,
            "n_iters": self.n_iters,
            "kernel": self.kernel
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
