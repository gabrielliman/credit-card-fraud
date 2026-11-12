"""Model factory. Each entry returns an object with fit / predict_proba."""
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build(name, params, cats, columns):
    if name == "logreg":
        # D004: one-hot categoricals + standardized numerics (bools cast to 0/1 in data.prepare)
        nums = [c for c in columns if c not in cats]
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
            ("num", StandardScaler(), nums),
        ])
        return make_pipeline(pre, LogisticRegression(**{"max_iter": 1000, **params}))
    if name == "lgbm":
        # D005: native categorical handling (pandas 'category' dtype is auto-detected)
        # metric="auc": track ONLY AUC, otherwise early stopping also stops on binary_logloss (bug found in exp04a)
        return lgb.LGBMClassifier(**{"verbose": -1, "random_state": 42, "metric": "auc", **params})
    raise ValueError(name)
