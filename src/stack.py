"""Stacking (D016): logistic regression on rank-transformed OOF predictions.

Usage: conda run -n fraud python -m src.stack experiments/<out_exp> experiments/expA experiments/expB ...

- Inputs: each base experiment's oof.npy / test_pred.npy (same fixed folds, D003).
- Features: predictions converted to normalized ranks in [0,1] (AUC depends only on ranks; puts
  models with differently scaled probabilities on the same footing).
- The meta-model is evaluated with the SAME 5 folds: fit on 4 folds' OOF, predict the 5th, so the stack's
  OOF score is never computed on rows used to fit its weights (Wolpert, 1992).
- Final meta-model is fit on all OOF rows and applied to the test ranks.
- Writes metrics.json / oof.npy / test_pred.npy / submission.csv like src.run, so src.compare works.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import SEED, N_FOLDS
from .data import load


def ranks(a):
    return rankdata(a) / len(a)


def main(out, bases):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    train, _, sample, _, target = load()
    y = train[target].to_numpy()
    names = [Path(b).name for b in bases]
    O = np.column_stack([ranks(np.load(Path(b) / "oof.npy")) for b in bases])
    T = np.column_stack([ranks(np.load(Path(b) / "test_pred.npy")) for b in bases])

    oof, folds, coefs = np.zeros(len(y)), [], []
    for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(O, y):
        m = LogisticRegression(C=1.0, max_iter=1000).fit(O[tr], y[tr])
        oof[va] = m.predict_proba(O[va])[:, 1]
        folds.append(roc_auc_score(y[va], oof[va])); coefs.append(m.coef_[0])
    final = LogisticRegression(C=1.0, max_iter=1000).fit(O, y)
    pred = final.predict_proba(T)[:, 1]

    w = pd.DataFrame(coefs, columns=names)
    print("meta-model coefficients per fold:"); print(w.round(3).to_string())
    print("final coefficients:", dict(zip(names, final.coef_[0].round(3))))
    if (final.coef_[0] < 0).any():
        print("WARNING: negative weight(s): the stack is using a model as a correction, not a vote (Breiman, 1996)")
    metrics = dict(cv_auc=roc_auc_score(y, oof), fold_auc=folds, fold_std=float(np.std(folds)),
                   bases=names, final_coef=dict(zip(names, final.coef_[0].tolist())))
    print(json.dumps({k: metrics[k] for k in ("cv_auc", "fold_std")}, indent=1))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.save(out / "oof.npy", oof); np.save(out / "test_pred.npy", pred)
    sample[target] = pred; sample.to_csv(out / "submission.csv", index=False)
    (out / "command.txt").write_text("conda run -n fraud python -m src.stack " + " ".join([str(out)] + list(bases)) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
