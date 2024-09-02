import numpy as np

# kelas untuk membuat layer fully connected beserta aktivasinya
class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, activation, weight_init='he', regularization=None, reg_lambda=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.regularization = regularization
        self.reg_lambda = reg_lambda

        # inisialisasi weights dengan He initialization untuk aktivasi ReLU
        if weight_init == 'he':
            self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        else:
            self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))

    # aktivasi untuk forward propagation
    def activate(self, X):
        z = np.dot(X, self.weights) + self.biases
        if self.activation == 'relu':
            a = np.maximum(0, z)
        elif self.activation == 'sigmoid':
            a = 1 / (1 + np.exp(-z))
        elif self.activation == 'softmax':
            e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            a = e_z / np.sum(e_z, axis=1, keepdims=True)
        else:
            a = z
        return a, z

    # turunan fungsi aktivasi untuk backward propagation
    def activation_derivative(self, z):
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        else:
            return np.ones_like(z)

# kelas untuk membangun ANN
class ANN:
    def __init__(self, layer_sizes, activations, weight_init='he', loss_function='binary_crossentropy', regularization=None, reg_lambda=0.01, learning_rate=0.01, num_iterations=1000, batch_size=32):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weight_init = weight_init
        self.loss_function = loss_function
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.layers = []

        # inisialisasi layer sesuai ukuran dan aktivasi yang diinginkan
        for i in range(len(layer_sizes) - 1):
            self.layers.append(FullyConnectedLayer(layer_sizes[i], layer_sizes[i+1], activations[i], weight_init, regularization, reg_lambda))

    # forward propagation: menghitung aktivasi untuk setiap layer
    def forward_propagation(self, X):
        activations = [X]
        pre_activations = []
        for layer in self.layers:
            a, z = layer.activate(activations[-1])
            activations.append(a)
            pre_activations.append(z)
        return activations, pre_activations

    # backward propagation: menghitung gradien dan memperbarui bobot serta bias
    def backward_propagation(self, X, y, activations, pre_activations):
        m = X.shape[0]
        if self.loss_function == 'binary_crossentropy':
            d_activations = activations[-1] - y.reshape(-1, 1)  # untuk binary crossentropy
        else:
            d_activations = activations[-1] - y.reshape(-1, 1)  # untuk mean squared error

        # proses backpropagation untuk setiap layer
        for i in reversed(range(len(self.layers))):
            d_z = d_activations * self.layers[i].activation_derivative(pre_activations[i])
            d_weights = np.dot(activations[i].T, d_z) / m
            d_biases = np.sum(d_z, axis=0, keepdims=True) / m

            if self.layers[i].regularization == 'l2':
                d_weights += self.layers[i].reg_lambda * self.layers[i].weights / m

            # update weights dan bias
            self.layers[i].weights -= self.learning_rate * d_weights
            self.layers[i].biases -= self.learning_rate * d_biases

            # update d_activations untuk layer sebelumnya
            d_activations = np.dot(d_z, self.layers[i].weights.T)

    def fit(self, X, y):
        for iteration in range(self.num_iterations):
            for batch_start in range(0, X.shape[0], self.batch_size):
                X_batch = X[batch_start:batch_start + self.batch_size]
                y_batch = y[batch_start:batch_start + self.batch_size]

                activations, pre_activations = self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, y_batch, activations, pre_activations)

            if iteration % 100 == 0:
                y_pred = self.predict(X)
                if self.loss_function == 'binary_crossentropy':
                    loss = binary_crossentropy(y, y_pred)
                else:
                    loss = mean_squared_error(y, y_pred)

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        if self.layers[-1].activation == 'softmax':
            return activations[-1]
        return (activations[-1] > 0.5).astype(int)
    
    def get_params(self, deep=True):
        return {
            "layer_sizes": self.layer_sizes,
            "activations": self.activations,
            "weight_init": self.weight_init,
            "loss_function": self.loss_function,
            "regularization": self.regularization,
            "reg_lambda": self.reg_lambda,
            "learning_rate": self.learning_rate,
            "num_iterations": self.num_iterations,
            "batch_size": self.batch_size
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
