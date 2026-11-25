"""Checks that every student-written component is correct. Run before run_all.py.

  1. Data/split checks (Earl): counts, no subject overlap, feasibility of the focused study.
  2. Metrics vs scikit-learn (Earl).
  3. SoftMax gradient check + comparison with scikit-learn LogisticRegression (Preethi).
  4. PCA via SVD vs scikit-learn PCA (Afnan).

Usage: python verify.py "data/UCI HAR Dataset"
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix as sk_cm

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data import load_official, fit_standardizer, apply_standardizer, subject_folds  # noqa: E402
from metrics import macro_f1, accuracy, confusion_matrix  # noqa: E402
from softmax import SoftmaxSGD, gradient_check  # noqa: E402
from pca_svm import PCASVD  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "data/UCI HAR Dataset"
(X_tr, y_tr, g_tr), (X_te, y_te, g_te) = load_official(BASE)

print("1. Data checks")
print("   train", X_tr.shape, " test", X_te.shape)
print("   train subjects", len(np.unique(g_tr)), " test subjects", len(np.unique(g_te)))
print("   subject overlap:", set(g_tr) & set(g_te) or "none")
print("   train class counts", np.bincount(y_tr), " test", np.bincount(y_te))
subs = np.unique(g_tr)
min_cell = min(np.sum((g_tr == s) & (y_tr == c)) for s in subs for c in range(6))
worst3 = min(min(np.sum(np.isin(g_tr, trio) & (y_tr == c)) for c in range(6))
             for trio in itertools.combinations(subs, 3))
print(f"   min windows per subject per activity: {min_cell}")
print(f"   min per-activity pool over all 3-subject groups: {worst3} (need >= 100)")

print("\n2. Metrics vs scikit-learn")
rng = np.random.default_rng(1)
yt = rng.integers(0, 6, 500)
yp = np.where(rng.random(500) < 0.7, yt, rng.integers(0, 6, 500))
print(f"   macro-F1 ours {macro_f1(yt, yp):.10f}  sklearn {f1_score(yt, yp, average='macro'):.10f}")
print(f"   accuracy ours {accuracy(yt, yp):.4f}  sklearn {accuracy_score(yt, yp):.4f}")
print("   confusion matrices equal:", np.array_equal(confusion_matrix(yt, yp), sk_cm(yt, yp)))

print("\n3. SoftMax")
print(f"   gradient check max abs error: {gradient_check():.2e} (should be < 1e-6)")
# Compare on a held-out VALIDATION fold of training subjects, so the test set stays untouched.
tr_idx, va_idx = next(subject_folds(g_tr, n_splits=5, seed=0))
mu, sd = fit_standardizer(X_tr[tr_idx])
Xa, Xb = apply_standardizer(X_tr[tr_idx], mu, sd), apply_standardizer(X_tr[va_idx], mu, sd)
ya, yb = y_tr[tr_idx], y_tr[va_idx]
lam = 1e-3
ours = SoftmaxSGD(lam=lam, lr=0.1, epochs=30, seed=0).fit(Xa, ya)
# Same objective: ours = mean CE + (lam/2)||W||^2  <=>  sklearn C = 1 / (n * lam)
sk = LogisticRegression(C=1.0 / (len(ya) * lam), max_iter=3000).fit(Xa, ya)
sk_obj = SoftmaxSGD(lam=lam)
sk_obj.W, sk_obj.b = sk.coef_.T, sk.intercept_
print(f"   training objective ours {ours.loss(Xa, ya):.4f}  sklearn optimum {sk_obj.loss(Xa, ya):.4f}")
p_ours, p_sk = ours.predict(Xb), sk.predict(Xb)
print(f"   validation macro-F1 ours {macro_f1(yb, p_ours):.4f}  sklearn {macro_f1(yb, p_sk):.4f}")
print(f"   prediction agreement: {np.mean(p_ours == p_sk):.4f}")

print("\n4. PCA via SVD vs scikit-learn")
k = 50
pca_ours = PCASVD(k).fit(Xa)
pca_sk = PCA(n_components=k, svd_solver="full").fit(Xa)
Z_ours, Z_sk = pca_ours.transform(Xa), pca_sk.transform(Xa)
signs = np.sign(np.sum(Z_ours * Z_sk, axis=0))  # align axis signs (sign is arbitrary)
print(f"   max abs difference after sign alignment: {np.max(np.abs(Z_ours * signs - Z_sk)):.2e}")
s = np.linalg.svd(Xa - Xa.mean(0), compute_uv=False)
print(f"   explained variance with {k} components: {np.sum(s[:k]**2) / np.sum(s**2):.4f} "
      f"(sklearn {pca_sk.explained_variance_ratio_.sum():.4f})")
