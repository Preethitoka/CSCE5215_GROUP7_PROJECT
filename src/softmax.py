"""SoftMax classifier from scratch: cross-entropy + L2, mini-batch SGD.
Owner: Preethi (SoftMax).

Loss:  J(W,b) = -(1/n) sum_i sum_k Y_ik log P_ik + (lam/2) ||W||_F^2   (bias not penalized)
Grad:  dW = (1/m) X_B^T (P_B - Y_B) + lam W,   db = (1/m) sum_B (P_B - Y_B)
"""
import numpy as np


class SoftmaxSGD:
    def __init__(self, lam=1e-3, lr=0.1, epochs=30, batch_size=64, seed=0, K=6):
        self.lam = lam
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.K = K

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)  # numerical stability
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.W = np.zeros((d, self.K))
        self.b = np.zeros(self.K)
        Y = np.eye(self.K)[y]
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                Xb, Yb = X[idx], Y[idx]
                P = self._softmax(Xb @ self.W + self.b)
                G = (P - Yb) / len(idx)
                self.W -= self.lr * (Xb.T @ G + self.lam * self.W)
                self.b -= self.lr * G.sum(axis=0)
        return self

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)

    def predict(self, X):
        return np.argmax(X @ self.W + self.b, axis=1)

    def loss(self, X, y):
        P = self.predict_proba(X)
        ce = -np.mean(np.log(P[np.arange(len(y)), y] + 1e-12))
        return ce + 0.5 * self.lam * np.sum(self.W ** 2)


def gradient_check(eps=1e-5, seed=0):
    """Finite-difference check of the analytic gradient on a small random problem."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(20, 5))
    y = rng.integers(0, 6, size=20)
    model = SoftmaxSGD(lam=0.1, seed=seed)
    model.W = rng.normal(size=(5, 6)) * 0.1
    model.b = np.zeros(6)
    Y = np.eye(6)[y]
    P = model._softmax(X @ model.W + model.b)
    G = (P - Y) / len(y)
    analytic = X.T @ G + model.lam * model.W
    numeric = np.zeros_like(model.W)
    for i in range(model.W.shape[0]):
        for j in range(model.W.shape[1]):
            old = model.W[i, j]
            model.W[i, j] = old + eps
            up = model.loss(X, y)
            model.W[i, j] = old - eps
            down = model.loss(X, y)
            model.W[i, j] = old
            numeric[i, j] = (up - down) / (2 * eps)
    return np.max(np.abs(analytic - numeric))
