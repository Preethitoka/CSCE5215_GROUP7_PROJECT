"""Core comparison: three methods, same folds, same metrics.

Steps:
  1. 5-fold grouped CV on the 21 training subjects to choose hyperparameters (test set untouched).
  2. Refit each method on all 21 training subjects with the selected setting.
  3. Evaluate ONCE on the 9 test subjects.

Usage:  python run_all.py "data/UCI HAR Dataset"
        python run_all.py "data/UCI HAR Dataset" --quick   (small grids, for a smoke test)
"""
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data import load_official, fit_standardizer, apply_standardizer, subject_folds  # noqa: E402
from metrics import macro_f1, accuracy, per_subject_macro_f1, confusion_matrix  # noqa: E402
from softmax import SoftmaxSGD  # noqa: E402
from pca_svm import PCARBFSVM  # noqa: E402
from random_forest import make_rf  # noqa: E402
from plots import plot_confusion, plot_per_subject  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "data/UCI HAR Dataset"
QUICK = "--quick" in sys.argv
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
ACTIVITIES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

(X_tr, y_tr, g_tr), (X_te, y_te, g_te) = load_official(BASE)

if QUICK:
    GRIDS = {
        "softmax": [dict(lam=1e-3, lr=0.1)],
        "pca_rbf_svm": [dict(n_components=100, C=10)],
        "random_forest": [dict(max_features="sqrt", max_depth=None)],
    }
else:
    GRIDS = {
        "softmax": [dict(lam=l, lr=r) for l, r in itertools.product([1e-4, 1e-3, 1e-2, 1e-1], [0.1, 0.01])],
        "pca_rbf_svm": [dict(n_components=k, C=c) for k, c in itertools.product([50, 100, 200], [0.1, 1, 10])],
        "random_forest": [dict(max_features=f, max_depth=d) for f, d in itertools.product(["sqrt", 0.3], [None, 20])],
    }


def build(method, params):
    if method == "softmax":
        return SoftmaxSGD(**params, epochs=30, batch_size=64, seed=0)
    if method == "pca_rbf_svm":
        return PCARBFSVM(**params, seed=0)
    if method == "random_forest":
        return make_rf(**params, seed=0)
    raise ValueError(method)


def fit_predict(method, params, Xa, ya, Xb):
    """Standardize with Xa statistics only, fit on Xa, predict Xb. Returns (pred, train_s, predict_s)."""
    mu, sd = fit_standardizer(Xa)
    Xa_s, Xb_s = apply_standardizer(Xa, mu, sd), apply_standardizer(Xb, mu, sd)
    t0 = time.perf_counter()
    model = build(method, params).fit(Xa_s, ya)
    t1 = time.perf_counter()
    pred = model.predict(Xb_s)
    t2 = time.perf_counter()
    return pred, t1 - t0, t2 - t1


def cv_score(method, params):
    scores = []
    for tr_idx, va_idx in subject_folds(g_tr, n_splits=5, seed=0):
        pred, _, _ = fit_predict(method, params, X_tr[tr_idx], y_tr[tr_idx], X_tr[va_idx])
        scores.append(macro_f1(y_tr[va_idx], pred))
    return float(np.mean(scores)), float(np.std(scores))


hardware = {"python": platform.python_version(), "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(), "system": platform.platform()}
print("Hardware:", hardware)

summary, best_params, cv_tables = {}, {}, {}
for method, grid in GRIDS.items():
    print(f"\n=== {method}: grouped CV over {len(grid)} setting(s) ===")
    rows = []
    for params in grid:
        mean, std = cv_score(method, params)
        rows.append({"params": params, "cv_macro_f1": mean, "cv_std": std})
        print(f"  {params}  macro-F1 {mean:.4f} ± {std:.4f}")
    best = max(rows, key=lambda r: r["cv_macro_f1"])
    best_params[method] = best["params"]
    cv_tables[method] = rows
    print(f"  selected: {best['params']}")

    # ---- refit on all 21 training subjects, evaluate the test set ONCE ----
    pred_te, train_s, pred_s = fit_predict(method, best["params"], X_tr, y_tr, X_te)
    C = confusion_matrix(y_te, pred_te)
    per_subj = per_subject_macro_f1(y_te, pred_te, g_te)
    summary[method] = {
        "params": best["params"],
        "cv_macro_f1": best["cv_macro_f1"], "cv_std": best["cv_std"],
        "test_macro_f1": macro_f1(y_te, pred_te), "test_accuracy": accuracy(y_te, pred_te),
        "train_seconds": train_s, "predict_seconds": pred_s,
        "test_confusion": C.tolist(), "test_per_subject_macro_f1": per_subj,
    }
    plot_confusion(C, ACTIVITIES, f"{method} (test)", OUT / f"confusion_{method}.png")
    np.save(OUT / f"test_pred_{method}.npy", pred_te)

plot_per_subject({m: s["test_per_subject_macro_f1"] for m, s in summary.items()}, OUT / "per_subject_macro_f1.png")

with open(OUT / "best_params.json", "w") as f:
    json.dump(best_params, f, indent=2, default=str)
with open(OUT / "core_comparison.json", "w") as f:
    json.dump({"hardware": hardware, "methods": summary, "cv_tables": cv_tables}, f, indent=2, default=str)

print("\n=== Test results (evaluated once) ===")
print(f"{'method':15s} {'CV F1':>7s} {'test F1':>8s} {'test acc':>9s} {'train s':>8s} {'pred s':>7s}")
for m, s in summary.items():
    print(f"{m:15s} {s['cv_macro_f1']:7.4f} {s['test_macro_f1']:8.4f} {s['test_accuracy']:9.4f} "
          f"{s['train_seconds']:8.2f} {s['predict_seconds']:7.3f}")
