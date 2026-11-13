"""Model factory + uniform early-stopping interface.

build(name, params, cats, columns) -> estimator with fit / predict_proba
fit(model, name, X, y, Xv=None, yv=None, patience=None) -> best iteration count (or None)
"""
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SEED


def build(name, params, cats, columns):
    if name == "logreg":
        # D004: one-hot categoricals + standardized numerics (bools cast to 0/1 in data.prepare)
        nums = [c for c in columns if c not in cats]
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats),
                                 ("num", StandardScaler(), nums)])
        return make_pipeline(pre, LogisticRegression(**{"max_iter": 1000, **params}))
    if name == "lgbm":
        # D005 native categoricals. metric="auc": track ONLY AUC, otherwise early stopping
        # also stops on binary_logloss (bug found in exp04a)
        return lgb.LGBMClassifier(**{"verbose": -1, "random_state": SEED, "metric": "auc", **params})
    if name == "xgb":
        # D014 M3: native categorical support (consistent with D005)
        from xgboost import XGBClassifier
        # device="cuda": GPU histogram algorithm, same model as CPU hist (D014 compute note)
        return XGBClassifier(**{"tree_method": "hist", "device": "cuda", "enable_categorical": True, "eval_metric": "auc",
                                "random_state": SEED, "n_jobs": -1, **params})
    if name == "catboost":
        # ordered target statistics for categoricals (Prokhorenkova et al., 2018)
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**{"eval_metric": "AUC", "cat_features": cats, "verbose": 0,
                                     "random_seed": SEED, "thread_count": -1, **params})
    raise ValueError(name)


def _cb(X, cats):
    # CatBoost wants str/int categoricals, not pandas Categorical
    X = X.copy()
    for c in cats:
        X[c] = X[c].astype(str)
    return X


def fit(model, name, X, y, cats, Xv=None, yv=None, patience=None):
    es = patience is not None
    if name == "lgbm":
        kw = dict(eval_X=Xv, eval_y=yv, callbacks=[lgb.early_stopping(patience, verbose=False)]) if es else {}
        model.fit(X, y, **kw)
        return int(model.best_iteration_) if es else None
    if name == "xgb":
        if es:
            model.set_params(early_stopping_rounds=patience)
            model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
            return int(model.best_iteration) + 1
        model.fit(X, y, verbose=False)
        return None
    if name == "catboost":
        kw = dict(eval_set=(_cb(Xv, cats), yv), early_stopping_rounds=patience) if es else {}
        model.fit(_cb(X, cats), y, **kw)
        return int(model.get_best_iteration()) + 1 if es else None
    model.fit(X, y)
    return None


def predict(model, name, X, cats):
    return model.predict_proba(_cb(X, cats) if name == "catboost" else X)[:, 1]


def seed_param(name, seed):
    return {"catboost": {"random_seed": seed}}.get(name, {"random_state": seed})
