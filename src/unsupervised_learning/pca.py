import numpy as np
import matplotlib.pyplot as plt

class PCAManual:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None  
        self.explained_variance_ = None  
        self.explained_variance_ratio_ = None  
        self.mean_ = None  

    def fit(self, X):
        # fit model. Proses mencakup penghitungan mean, covariance matrix, dan komponen utama melalui eigen decomposition

        self.mean_ = np.mean(X, axis=0)  
        X_centered = X - self.mean_  # mean subtraction
        covariance_matrix = np.cov(X_centered, rowvar=False)  # hitung covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)  # eigen decomposition
        sorted_idx = np.argsort(eigenvalues)[::-1]  # urutkan eigenvalues dan eigenvectors
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # jika n_components ditentukan, pilih hanya sejumlah komponen tersebut
        if self.n_components is not None:
            eigenvectors = eigenvectors[:, :self.n_components]
            eigenvalues = eigenvalues[:self.n_components]

        self.components_ = eigenvectors
        self.explained_variance_ = eigenvalues
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance  # Hitung rasio varians

    def transform(self, X):
        # transformasikan data ke ruang dimensi rendah
        X_centered = X - self.mean_  
        return np.dot(X_centered, self.components_)  

    def inverse_transform(self, X_transformed):
        # kembalikan data yang sudah ditransformasikan ke ruang dimensi asli
        return np.dot(X_transformed, self.components_.T) + self.mean_

    def fit_transform(self, X):
        self.fit(X.to_numpy())
        return self.transform(X.to_numpy())
