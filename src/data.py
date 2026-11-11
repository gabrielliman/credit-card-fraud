import pandas as pd
from .config import DATA


def load():
    """Load train/test/sample submission; infer id & target columns from sample_submission."""
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    id_col, target = sample.columns[0], sample.columns[1]
    return train, test, sample, id_col, target


def prepare(train, test, id_col, target):
    X = train.drop(columns=[id_col, target])
    y = train[target]
    X_test = test.drop(columns=[id_col])
    cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cats:
        both = pd.concat([X[c], X_test[c]]).astype("category")
        X[c] = pd.Categorical(X[c], categories=both.cat.categories)
        X_test[c] = pd.Categorical(X_test[c], categories=both.cat.categories)
    return X, y, X_test, cats
