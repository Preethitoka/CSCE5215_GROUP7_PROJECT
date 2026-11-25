"""Data loading, standardization, and subject-grouped folds.
Owner: Earl (pipeline). Standardization is fit on training data only.
"""
from pathlib import Path

import numpy as np


def load_split(base, split):
    base = Path(base)
    X = np.loadtxt(base / split / f"X_{split}.txt")
    y = np.loadtxt(base / split / f"y_{split}.txt", dtype=int) - 1  # labels 0..5
    g = np.loadtxt(base / split / f"subject_{split}.txt", dtype=int)  # participant IDs
    return X, y, g


def load_official(base):
    """Return (X_train, y_train, g_train), (X_test, y_test, g_test)."""
    return load_split(base, "train"), load_split(base, "test")


def fit_standardizer(X_train):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0  # guard against constant features
    return mu, sd


def apply_standardizer(X, mu, sd):
    return (X - mu) / sd


def subject_folds(groups, n_splits=5, seed=0):
    """Yield (train_idx, val_idx) with whole subjects held out together."""
    subjects = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    fold_of = {s: i % n_splits for i, s in enumerate(subjects)}
    fold = np.array([fold_of[s] for s in groups])
    for k in range(n_splits):
        val = fold == k
        yield np.where(~val)[0], np.where(val)[0]
