import numpy as np

class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, reg_lambda=0.01, method='gradient_descent', batch_size=None):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.method = method
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # fungsi aktivasi sigmoid untuk menghasilkan probabilitas
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, n_features):
        # inisialisasi weight dan bias dengan nilai awal 0
        self.weights = np.zeros(n_features)
        self.bias = 0

    def compute_cost(self, y_true, y_pred):
        # hitung cost atau loss function (cross-entropy) dengan opsi regularisasi
        m = y_true.shape[0]
        cost = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # tambahkan penalti regularisasi jika diterapkan (L1 atau L2)
        if self.regularization == 'l2':
            cost += (self.reg_lambda / (2 * m)) * np.sum(np.square(self.weights))
        elif self.regularization == 'l1':
            cost += (self.reg_lambda / m) * np.sum(np.abs(self.weights))
        
        return cost

    def compute_gradients(self, X, y_true, y_pred):
        # hitung gradien untuk weight dan bias
        m = X.shape[0]
        dw = 1/m * np.dot(X.T, (y_pred - y_true))
        db = 1/m * np.sum(y_pred - y_true)
        
        # tambahkan penalti regularisasi pada gradien jika diterapkan
        if self.regularization == 'l2':
            dw += (self.reg_lambda / m) * self.weights
        elif self.regularization == 'l1':
            dw += (self.reg_lambda / m) * np.sign(self.weights)
        
        return dw, db

    def compute_hessian(self, X, y_pred):
        # hitung matriks Hessian untuk Newton's method
        m = X.shape[0]
        S = np.diag(y_pred * (1 - y_pred))
        H = np.dot(np.dot(X.T, S), X) / m
        
        # tambahkan penalti regularisasi pada Hessian jika menggunakan L2
        if self.regularization == 'l2':
            H += (self.reg_lambda / m) * np.eye(X.shape[1])
        
        return H

    def fit(self, X, y):
        # latih model dengan menggunakan iterasi dan metode yang dipilih
        n_features = X.shape[1]
        self.initialize_parameters(n_features)

        for i in range(self.num_iterations):
            # mini-batch gradient descent jika batch_size diatur
            if self.batch_size:
                indices = np.random.choice(X.shape[0], self.batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y

            z = np.dot(X_batch, self.weights) + self.bias # hitung output model
            y_pred = self.sigmoid(z)

            cost = self.compute_cost(y_batch, y_pred) # hitung cost untuk pemantauan

            # update weight dan bias berdasarkan metode optimasi yang dipilih
            if self.method == 'gradient_descent':
                dw, db = self.compute_gradients(X_batch, y_batch, y_pred)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            elif self.method == 'newton':
                dw, db = self.compute_gradients(X_batch, y_batch, y_pred)
                H = self.compute_hessian(X_batch, y_pred)
                H_inv = np.linalg.inv(H)
                self.weights -= np.dot(H_inv, dw)
                self.bias -= db  # update bias

            # tampilkan cost setiap 100 iterasi untuk monitoring
            if i % 100 == 0:
                print(f"Iteration {i}: Cost {cost}")

    def predict(self, X):
        # prediksi output biner berdasarkan probabilitas
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return np.where(y_pred > 0.5, 1, 0)
    
    def get_params(self, deep=True):
        return {
            "learning_rate": self.learning_rate,
            "num_iterations": self.num_iterations,
            "regularization": self.regularization,
            "reg_lambda": self.reg_lambda,
            "method": self.method,
            "batch_size": self.batch_size
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
