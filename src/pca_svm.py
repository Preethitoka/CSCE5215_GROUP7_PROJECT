"""PCA via SVD (from scratch) followed by RBF-kernel SVM.
Owner: Afnan (PCA + RBF-SVM).
"""
import numpy as np
from sklearn.svm import SVC


class PCASVD:
    def __init__(self, n_components):
        self.k = n_components

    def fit(self, X):
        self.mu = X.mean(axis=0)
        Xc = X - self.mu
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.W = Vt[: self.k].T  # d x k principal axes
        return self

    def transform(self, X):
        return (X - self.mu) @ self.W


class PCARBFSVM:
    def __init__(self, n_components=100, C=1.0, gamma="scale", seed=0):
        self.pca = PCASVD(n_components)
        self.svm = SVC(kernel="rbf", C=C, gamma=gamma, random_state=seed)

    def fit(self, X, y):
        self.pca.fit(X)
        self.svm.fit(self.pca.transform(X), y)
        return self

    def predict(self, X):
        return self.svm.predict(self.pca.transform(X))
