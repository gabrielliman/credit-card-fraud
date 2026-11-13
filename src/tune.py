"""Optuna TPE tuning (D013), any model in src/models.py.

Usage: conda run -n fraud python -m src.tune experiments/<exp> <model> [n_trials]

Protocol (D013): train on folds 1-4 of split 0, score on fold 0; lr 0.05 + early stopping
(patience 300, AUC only); study in <exp>/study.db; best params -> <exp>/config.json (lr 0.02),
then `src.run` evaluates on all 5 folds (D009). Report gains on folds 1-4.
"""
import json, sys
from pathlib import Path

import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import SEED, N_FOLDS
from .data import load, prepare
from .models import build, fit, predict

SEARCH_LR, FINAL_LR, PATIENCE = 0.05, 0.02, 300

FIXED = {  # non-searched settings
    "lgbm": {"n_estimators": 10000, "subsample_freq": 1},
    "xgb": {"n_estimators": 10000},
    "catboost": {"n_estimators": 10000},  # n_estimators (synonym of iterations) so src.run can override it
}


def space(model, t):
    if model == "lgbm":
        return dict(num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
                    min_child_samples=t.suggest_int("min_child_samples", 10, 1000, log=True),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
                    reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    min_split_gain=t.suggest_float("min_split_gain", 0.0, 1.0))
    if model == "xgb":  # same concepts as lgbm; depth-wise trees -> max_depth instead of num_leaves
        return dict(max_depth=t.suggest_int("max_depth", 3, 10),
                    min_child_weight=t.suggest_float("min_child_weight", 0.1, 100, log=True),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
                    reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    gamma=t.suggest_float("gamma", 0.0, 1.0))
    if model == "catboost":  # symmetric trees -> depth; l2 + random_strength + bagging as regularizers
        return dict(depth=t.suggest_int("depth", 4, 10),
                    l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.1, 30, log=True),
                    random_strength=t.suggest_float("random_strength", 1e-3, 10, log=True),
                    bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 1.0),
                    rsm=t.suggest_float("rsm", 0.3, 1.0),
                    min_data_in_leaf=t.suggest_int("min_data_in_leaf", 1, 1000, log=True))
    raise ValueError(model)


def main(exp_dir, model, n_trials=100):
    exp = Path(exp_dir)
    train, test, _, id_col, target = load()
    X, y, _, cats = prepare(train, test, id_col, target)
    tr, va = next(iter(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)))

    def objective(t):
        m = build(model, {**FIXED[model], "learning_rate": SEARCH_LR, **space(model, t)}, cats, X.columns)
        t.set_user_attr("best_iter", fit(m, model, X.iloc[tr], y.iloc[tr], cats, X.iloc[va], y.iloc[va], PATIENCE))
        return roc_auc_score(y.iloc[va], predict(m, model, X.iloc[va], cats))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name=model, storage=f"sqlite:///{exp / 'study.db'}",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=max(0, n_trials - len(study.trials)),
                   callbacks=[lambda s, t: print(f"trial {t.number:3d}: {t.value:.5f} (best {s.best_value:.5f})", flush=True)])
    print("best fold-0 AUC:", round(study.best_value, 5), "params:", study.best_params)
    cfg = {"model": model, "early_stopping": PATIENCE,
           "params": {**FIXED[model], "learning_rate": FINAL_LR, **study.best_params},
           "note": f"Optuna TPE best of {len(study.trials)} trials on fold 0 at lr {SEARCH_LR} (D013); "
                   f"fold-0 AUC {study.best_value:.5f}. Report gain on folds 1-4."}
    (exp / "config.json").write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 100)
