"""Config-driven experiment runner.

Usage: conda run -n fraud python -m src.run experiments/<exp_dir>

Reads <exp_dir>/config.json and writes to the same folder:
  metrics.json   per-fold + overall CV ROC-AUC, tree counts, runtime
  oof.npy        out-of-fold predictions (same fixed folds for all experiments, D003)
  test_pred.npy  fold-averaged test predictions
  submission.csv Kaggle submission
  command.txt    command to reproduce

Config keys:
  model, params        see src/models.py
  early_stopping       patience. Enables the two-pass protocol (D009):
                       pass 1 early-stops on each validation fold to find best iterations;
                       pass 2 retrains every fold with n_estimators = median(best iterations), no peeking.
  resample             optional: "smote" | "smote_round" (training fold only, D010)
  features             optional list of steps from src/features.py
  seeds                optional list of random seeds to average in pass 2 (D015)
"""
import json, sys, time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import SEED, N_FOLDS, ROOT
from .data import load, prepare
from .models import build, fit, predict, seed_param
from .resample import resample
from .features import apply as apply_features


def cv_pass(cfg, params, X, y, X_test, cats, splits, early_stopping=None):
    oof, pred, folds, best = np.zeros(len(X)), np.zeros(len(X_test)), [], []
    for k, (tr, va) in enumerate(splits):
        Xtr, ytr = resample(cfg.get("resample"), X.iloc[tr], y.iloc[tr], cats)
        # D015 seed averaging: pass 2 averages models trained with different seeds (pass 1 uses the first seed)
        seeds = cfg.get("seeds", [SEED]) if not early_stopping else cfg.get("seeds", [SEED])[:1]
        for sd in seeds:
            model = build(cfg["model"], {**params, **seed_param(cfg["model"], sd)}, cats, X.columns)
            b = fit(model, cfg["model"], Xtr, ytr, cats, X.iloc[va], y.iloc[va], early_stopping)
            if b is not None:
                best.append(b)
            oof[va] += predict(model, cfg["model"], X.iloc[va], cats) / len(seeds)
            pred += predict(model, cfg["model"], X_test, cats) / len(splits) / len(seeds)
        folds.append(roc_auc_score(y.iloc[va], oof[va]))
        print(f"  fold {k}: AUC {folds[-1]:.5f}" + (f" (best iter {best[-1]})" if best else ""), flush=True)
    return oof, pred, folds, best


def main(exp_dir):
    exp = Path(exp_dir)
    cfg = json.loads((exp / "config.json").read_text())
    train, test, sample, id_col, target = load()
    X, y, X_test, cats = prepare(train, test, id_col, target)
    X, X_test = apply_features(cfg.get("features"), X, X_test)
    cats = X.select_dtypes(include="category").columns.tolist()
    print("features:", X.shape[1], "| categorical:", len(cats))
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    params = cfg.get("params", {})
    t0, extra = time.time(), {}

    if cfg.get("early_stopping"):
        print("pass 1: early stopping to choose tree count")
        _, _, es_folds, best = cv_pass(cfg, params, X, y, X_test, cats, splits, cfg["early_stopping"])
        n_trees = int(np.median(best))
        extra = dict(pass1_best_iters=best, pass1_fold_auc=es_folds, n_trees=n_trees)
        params = {**params, "n_estimators": n_trees}
        print(f"pass 2: retrain all folds with fixed n_estimators={n_trees}")
    oof, pred, folds, _ = cv_pass(cfg, params, X, y, X_test, cats, splits)

    metrics = dict(cv_auc=roc_auc_score(y, oof), fold_auc=folds, fold_std=float(np.std(folds)),
                   runtime_s=round(time.time() - t0, 1), **extra)
    print(json.dumps({k: metrics[k] for k in ("cv_auc", "fold_std", "runtime_s")}, indent=1))
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.save(exp / "oof.npy", oof)
    np.save(exp / "test_pred.npy", pred)
    sample[target] = pred
    sample.to_csv(exp / "submission.csv", index=False)
    rel = exp.resolve().relative_to(ROOT)
    (exp / "command.txt").write_text(f"conda run -n fraud python -m src.run {rel}\n")


if __name__ == "__main__":
    main(sys.argv[1])
