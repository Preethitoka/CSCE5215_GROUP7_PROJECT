"""Error analysis on the test predictions saved by run_all.py.
Owner: Jaydeep (random forest + error analysis).

Reports: per-class F1, most-confused activity pairs, weakest test participants.
Usage: python error_analysis.py "data/UCI HAR Dataset"
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data import load_official  # noqa: E402
from metrics import confusion_matrix  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "data/UCI HAR Dataset"
OUT = Path(__file__).parent / "results"
ACTIVITIES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
_, (X_te, y_te, g_te) = load_official(BASE)
core = json.load(open(OUT / "core_comparison.json"))["methods"]

report = {}
for method in core:
    pred = np.load(OUT / f"test_pred_{method}.npy")
    C = confusion_matrix(y_te, pred)
    tp = np.diag(C)
    f1 = 2 * tp / (C.sum(0) + C.sum(1))
    off = [(int(C[i, j]), ACTIVITIES[i], ACTIVITIES[j]) for i in range(6) for j in range(6) if i != j]
    top = sorted(off, reverse=True)[:3]
    per_subj = core[method]["test_per_subject_macro_f1"]
    weakest = sorted(per_subj.items(), key=lambda kv: kv[1])[:3]
    report[method] = {"per_class_f1": dict(zip(ACTIVITIES, f1.round(4).tolist())),
                      "top_confusions": top, "weakest_subjects": weakest}
    print(f"\n=== {method} ===")
    for a, v in zip(ACTIVITIES, f1):
        print(f"  F1 {a:11s} {v:.4f}")
    print("  Most confused (count, true -> predicted):")
    for n, t, p in top:
        print(f"    {n:4d}  {t} -> {p}")
    print("  Weakest test participants (macro-F1):", ", ".join(f"{s}: {v:.3f}" for s, v in weakest))

with open(OUT / "error_analysis.json", "w") as f:
    json.dump(report, f, indent=2)
