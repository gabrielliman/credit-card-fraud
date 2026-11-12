"""Optuna TPE tuning of LightGBM (D013).

Usage: conda run -n fraud python -m src.tune experiments/exp06_optuna [n_trials]

Protocol (D013):
  - each trial trains on folds 1-4 of split 0 and is scored on fold 0 (same StratifiedKFold as everything else)
  - learning rate 0.05 + early stopping (patience 300, AUC only) to keep trials fast
  - study stored in <exp>/study.db (resumable; used for history/importance plots)
  - best params are written to <exp>/config.json, which is then run with src.run at lr 0.02 (D009 protocol)
"""
import json, sys
from pathlib import Path

import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import SEED, N_FOLDS
from .data import load, prepare

SEARCH_LR, FINAL_LR = 0.05, 0.02


def space(trial):
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 1000, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
    )


def main(exp_dir, n_trials=100):
    exp = Path(exp_dir)
    train, test, _, id_col, target = load()
    X, y, _, _ = prepare(train, test, id_col, target)
    tr, va = next(iter(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)))

    def objective(trial):
        p = space(trial)
        m = lgb.LGBMClassifier(n_estimators=10000, learning_rate=SEARCH_LR, subsample_freq=1,
                               metric="auc", verbose=-1, random_state=SEED, **p)
        m.fit(X.iloc[tr], y.iloc[tr], eval_X=X.iloc[va], eval_y=y.iloc[va],
              callbacks=[lgb.early_stopping(300, verbose=False)])
        trial.set_user_attr("best_iter", int(m.best_iteration_))
        return roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="lgbm",
                                storage=f"sqlite:///{exp / 'study.db'}", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    done = len(study.trials)
    study.optimize(objective, n_trials=max(0, n_trials - done),
                   callbacks=[lambda s, t: print(f"trial {t.number:3d}: {t.value:.5f} (best {s.best_value:.5f})", flush=True)])

    best = study.best_params
    print("best fold-0 AUC:", round(study.best_value, 5), "params:", best)
    cfg = {"model": "lgbm", "early_stopping": 300,
           "params": {"n_estimators": 10000, "learning_rate": FINAL_LR, "subsample_freq": 1, **best},
           "note": f"Optuna TPE best of {len(study.trials)} trials on fold 0 at lr {SEARCH_LR} (D013); "
                   f"fold-0 AUC {study.best_value:.5f}. Report gain on folds 1-4 (fold 0 was used for tuning)."}
    (exp / "config.json").write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100)
