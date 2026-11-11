"""Baseline: stratified K-fold LightGBM, ROC-AUC, writes OOF + submission."""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import SEED, N_FOLDS, SUBS, MODELS
from .data import load, prepare


def main():
    train, test, sample, id_col, target = load()
    X, y, X_test, _ = prepare(train, test, id_col, target)
    print(f"train {X.shape}, test {X_test.shape}, positive rate {y.mean():.4f}")

    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    params = dict(n_estimators=5000, learning_rate=0.03, num_leaves=63,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  random_state=SEED, verbose=-1)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
              eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"fold {fold}: AUC {roc_auc_score(y.iloc[va], oof[va]):.5f}")

    cv = roc_auc_score(y, oof)
    print(f"CV AUC {cv:.5f}")
    np.save(MODELS / "lgb_oof.npy", oof)
    sample[target] = pred
    out = SUBS / f"lgb_cv{cv:.5f}.csv"
    sample.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
