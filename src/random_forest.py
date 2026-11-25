"""Random forest baseline (scikit-learn). Owner: Jaydeep (random forest + error analysis)."""
from sklearn.ensemble import RandomForestClassifier


def make_rf(max_features="sqrt", max_depth=None, seed=0):
    return RandomForestClassifier(
        n_estimators=300,
        max_features=max_features,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=seed,
    )
