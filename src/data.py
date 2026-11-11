import pandas as pd
from .config import DATA


def load():
    """Load train/test/sample submission; id & target columns from sample_submission."""
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    id_col, target = sample.columns[0], sample.columns[1]
    return train, test, sample, id_col, target


def prepare(train, test, id_col, target):
    """Minimal, model-agnostic preparation: bools -> int, strings -> shared category dtype."""
    X = train.drop(columns=[id_col, target])
    y = train[target].astype(int)
    X_test = test.drop(columns=[id_col])
    for df in (X, X_test):
        for c in df.select_dtypes(include="bool").columns:
            df[c] = df[c].astype(int)
    cats = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in cats:
        levels = sorted(set(X[c]) | set(X_test[c]))
        X[c] = pd.Categorical(X[c], categories=levels)
        X_test[c] = pd.Categorical(X_test[c], categories=levels)
    return X, y, X_test, cats
