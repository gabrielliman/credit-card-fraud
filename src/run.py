"""Config-driven experiment runner.

Usage: conda run -n fraud python -m src.run experiments/<exp_dir>

Reads <exp_dir>/config.json and writes to the same folder:
  metrics.json   per-fold + overall CV ROC-AUC, runtime
  oof.npy        out-of-fold predictions (same fixed folds for all experiments, D003)
  test_pred.npy  fold-averaged test predictions
  submission.csv Kaggle submission
  command.txt    command to reproduce
"""
import json, sys, time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import SEED, N_FOLDS, ROOT
from .data import load, prepare
from .models import build


def main(exp_dir):
    exp = Path(exp_dir)
    cfg = json.loads((exp / "config.json").read_text())
    train, test, sample, id_col, target = load()
    X, y, X_test, cats = prepare(train, test, id_col, target)

    oof, pred, folds = np.zeros(len(X)), np.zeros(len(X_test)), []
    t0 = time.time()
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, va) in enumerate(skf.split(X, y)):
        model = build(cfg["model"], cfg.get("params", {}), cats, X.columns)
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        folds.append(roc_auc_score(y.iloc[va], oof[va]))
        print(f"fold {k}: AUC {folds[-1]:.5f}", flush=True)

    metrics = dict(cv_auc=roc_auc_score(y, oof), fold_auc=folds,
                   fold_std=float(np.std(folds)), runtime_s=round(time.time() - t0, 1))
    print(json.dumps(metrics, indent=1))
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.save(exp / "oof.npy", oof)
    np.save(exp / "test_pred.npy", pred)
    sample[target] = pred
    sample.to_csv(exp / "submission.csv", index=False)
    (exp / "command.txt").write_text(f"conda run -n fraud python -m src.run {exp.relative_to(ROOT) if exp.is_absolute() else exp}\n")


if __name__ == "__main__":
    main(sys.argv[1])
