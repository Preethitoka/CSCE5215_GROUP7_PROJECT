# CSCE 4205/5215 Group 7: Human Activity Recognition (UCI HAR)

We compare three classifiers on recognizing six activities for participants never seen in training,
then use our from-scratch SoftMax classifier for one focused study: does the number of distinct
training participants change macro-F1 on unseen participants when the total training data is fixed?

## Team and code ownership

| Member | Owns | Files |
|---|---|---|
| Afnan Abdul Razzack | PCA via SVD (from scratch) + RBF-SVM | `src/pca_svm.py`, PCA check in `verify.py` |
| Preethi Toka | SoftMax classifier from scratch (cross-entropy, L2, mini-batch SGD) | `src/softmax.py`, SoftMax checks in `verify.py` |
| Jaydeep Reddy Chintham | Random forest baseline + error analysis | `src/random_forest.py`, `error_analysis.py` |
| Earl Amematekpo | Data pipeline, subject-grouped folds, metrics, participant sampling, focused study runs | `src/data.py`, `src/metrics.py`, `run_all.py`, `focused.py` |

Student-written components: SoftMax (loss, gradient, SGD loop, prediction), PCA via SVD,
metric functions (macro-F1, accuracy, confusion matrix, per-participant F1), subject-grouped folds,
and the participant-stratified sampler. External libraries: NumPy, SciPy (Welch t-test),
scikit-learn (random forest, SVC, and reference implementations used only for checking), Matplotlib.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset from https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones,
unzip it (it contains a second zip), and place the folder so this path exists:

```
data/UCI HAR Dataset/train/X_train.txt
```

The `data/` folder is in `.gitignore`; each member downloads the dataset once.

## Run order

```bash
python verify.py         "data/UCI HAR Dataset"   # correctness checks (about 5 s)
python run_all.py        "data/UCI HAR Dataset"   # core comparison: CV tuning + one test evaluation (several minutes)
python focused.py        "data/UCI HAR Dataset"   # focused study (about 5 s)
python error_analysis.py "data/UCI HAR Dataset"   # per-class F1, confusions, weakest participants
```

`python run_all.py "data/UCI HAR Dataset" --quick` runs one setting per method as a smoke test.
Quick-mode numbers are not the reported results.

## Protocol

- **Split:** official UCI HAR split, 21 training participants (7,352 windows), 9 test participants (2,947 windows), no overlap.
- **Validation:** 5-fold grouped CV over the 21 training participants (`subject_folds`, seed 0). All hyperparameters are chosen here.
- **Leakage control:** standardization and PCA are fit on the training portion of each fold only, and on the full training set for the final model. The test set is evaluated once per method, after selection.
- **Metrics:** macro-F1 (primary), accuracy, confusion matrix, per-participant macro-F1, training and prediction time with hardware details (printed by `run_all.py` and saved in `results/core_comparison.json`).
- **Seeds:** SoftMax, SVM and random forest use seed 0 for the core comparison; the focused study uses seeds 0 to 9 for its 10 draws.

### Hyperparameter grids

| Method | Grid |
|---|---|
| SoftMax | λ ∈ {1e-4, 1e-3, 1e-2, 1e-1}, learning rate ∈ {0.1, 0.01}; 30 epochs, batch 64 |
| PCA + RBF-SVM | components ∈ {50, 100, 200}, C ∈ {0.1, 1, 10}, gamma = "scale" |
| Random forest | 300 trees, max_features ∈ {sqrt, 0.3}, max_depth ∈ {None, 20} |

## Focused study

- **Varied:** number of training participants k ∈ {3, 6, 11, 16, 21}.
- **Fixed:** 600 training windows (100 per activity) drawn from the k chosen participants, the same frozen hyperparameters, training procedure, hardware, and the 9 test participants.
- **Tuning:** λ is tuned once at the 600-window budget on the validation folds, then frozen; the learning rate comes from the core comparison.
- **Design:** 10 random draws per k (seeds 0 to 9), 50 models, each tested once. Each k is compared with k = 3 using Welch's t-test.
- **Feasibility:** every training participant has at least 36 windows per activity, and every group of 3 has at least 113 per activity (checked by `verify.py`).
- **Limitation:** with total windows fixed, more participants also means fewer windows per participant (about 200 at k = 3, about 29 at k = 21), so any gain is a joint effect. At k = 21 every draw uses all participants, so only the sampled windows vary.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `core_comparison.json` | CV tables, selected settings, test metrics, timings, hardware |
| `best_params.json` | Selected hyperparameters per method |
| `confusion_<method>.png` | Test confusion matrix per method |
| `per_subject_macro_f1.png` | Per-participant macro-F1 for all three methods |
| `focused_study.json`, `focused_study.png` | Focused-study draws, means, standard deviations, p-values |
| `error_analysis.json` | Per-class F1, most-confused pairs, weakest participants |
