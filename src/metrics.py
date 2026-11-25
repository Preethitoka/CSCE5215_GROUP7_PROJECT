"""Evaluation metrics written from scratch and checked against scikit-learn in check_metrics.py.
Owner: Earl (pipeline).
"""
import numpy as np


def confusion_matrix(y_true, y_pred, K=6):
    C = np.zeros((K, K), dtype=int)
    np.add.at(C, (y_true, y_pred), 1)  # rows = true class, cols = predicted class
    return C


def macro_f1(y_true, y_pred, K=6):
    C = confusion_matrix(y_true, y_pred, K)
    tp = np.diag(C).astype(float)
    fp = C.sum(axis=0) - tp
    fn = C.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.zeros(K)
    nz = denom > 0
    f1[nz] = 2 * tp[nz] / denom[nz]
    return f1.mean()


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def per_subject_macro_f1(y_true, y_pred, groups, K=6):
    return {int(s): macro_f1(y_true[groups == s], y_pred[groups == s], K)
            for s in np.unique(groups)}
