"""Focused study: how does the number of training participants affect SoftMax macro-F1?
Owner: Earl (focused runs), using Preethi's SoftMax.

Fixed: 600 training windows (100 per activity), same hyperparameters, same test set.
Varied: k training participants in {3, 6, 11, 16, 21}. 10 random draws per k (seeds 0-9).
Compare each k to k = 3 with Welch's t-test.

The learning rate comes from results/best_params.json (written by run_all.py). Lambda is
re-tuned once at the 600-window budget on validation folds, as the proposal states, then frozen.
The test set is never used to choose anything. Usage: python focused.py /path/to/"UCI HAR Dataset"
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data import load_official, fit_standardizer, apply_standardizer, subject_folds  # noqa: E402
from metrics import macro_f1  # noqa: E402
from softmax import SoftmaxSGD  # noqa: E402
from plots import plot_focused  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "data/UCI HAR Dataset"
OUT = Path(__file__).parent / "results"
(X_tr, y_tr, g_tr), (X_te, y_te, g_te) = load_official(BASE)
best = json.load(open(OUT / "best_params.json"))["softmax"]

KS = [3, 6, 11, 16, 21]
DRAWS = 10
PER_CLASS = 100
K_CLASSES = 6
all_subjects = np.unique(g_tr)


def draw_training_set(k, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_subjects, size=k, replace=False)
    pool = np.isin(g_tr, chosen)
    idx = []
    for c in range(K_CLASSES):
        cand = np.where(pool & (y_tr == c))[0]
        idx.append(rng.choice(cand, size=PER_CLASS, replace=False))
    return np.concatenate(idx), chosen


# ---- Tune lambda ONCE at the study's data budget (600 windows), on validation folds only ----
def sample_budget(pool_mask, rng):
    return np.concatenate([rng.choice(np.where(pool_mask & (y_tr == c))[0], PER_CLASS, replace=False)
                           for c in range(K_CLASSES)])


lam_scores = {}
for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(subject_folds(g_tr, n_splits=5, seed=0)):
        pool = np.zeros(len(y_tr), bool)
        pool[tr_idx] = True
        idx = sample_budget(pool, np.random.default_rng(fold))
        mu, sd = fit_standardizer(X_tr[idx])
        m = SoftmaxSGD(lam=lam, lr=best["lr"], epochs=30, batch_size=64, seed=0)
        m.fit(apply_standardizer(X_tr[idx], mu, sd), y_tr[idx])
        scores.append(macro_f1(y_tr[va_idx], m.predict(apply_standardizer(X_tr[va_idx], mu, sd))))
    lam_scores[lam] = float(np.mean(scores))
    print(f"lambda {lam:g}: validation macro-F1 at 600 windows {lam_scores[lam]:.4f}")
best = {"lam": max(lam_scores, key=lam_scores.get), "lr": best["lr"]}
print(f"Frozen for the study: {best}\n")

results = {k: [] for k in KS}
for k in KS:
    for seed in range(DRAWS):
        idx, chosen = draw_training_set(k, seed)
        mu, sd = fit_standardizer(X_tr[idx])  # training draw only
        model = SoftmaxSGD(**best, epochs=30, batch_size=64, seed=seed)
        model.fit(apply_standardizer(X_tr[idx], mu, sd), y_tr[idx])
        pred = model.predict(apply_standardizer(X_te, mu, sd))  # full official test set
        results[k].append(macro_f1(y_te, pred))
    print(f"k={k:2d}  macro-F1 {np.mean(results[k]):.4f} ± {np.std(results[k]):.4f}")

base = results[KS[0]]
print("\nWelch t-test vs k=3:")
summary = {}
for k in KS:
    mean, std = np.mean(results[k]), np.std(results[k])
    if k == KS[0]:
        p = None
    else:
        p = stats.ttest_ind(results[k], base, equal_var=False).pvalue
    summary[k] = {"mean": mean, "std": std, "p_vs_k3": p, "draws": results[k]}
    print(f"  k={k:2d}  mean diff {mean - np.mean(base):+.4f}  p={'-' if p is None else f'{p:.2e}'}")

with open(OUT / "focused_study.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

plot_focused(KS, [summary[k]["mean"] for k in KS], [summary[k]["std"] for k in KS],
             OUT / "focused_study.png")
print("\nNote: at k=21 every draw uses all training participants, so only the sampled windows vary.")
