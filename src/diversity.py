"""Diversity report over experiments' out-of-fold predictions (D014 M4).

Usage: conda run -n fraud python -m src.diversity experiments/expA experiments/expB ...

Prints each model's CV AUC, the Spearman (rank) correlation matrix of OOF predictions (AUC only
depends on ranks), and the CV AUC of a simple rank-average of every pair, a first hint of
ensembling potential (Kuncheva & Whitaker, 2003).
"""
import sys, itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .data import load


def main(exps):
    train, _, _, _, target = load()
    y = train[target].to_numpy()
    names = [Path(e).name for e in exps]
    P = pd.DataFrame({n: np.load(Path(e) / "oof.npy") for n, e in zip(names, exps)})
    print("CV AUC:"); print(P.apply(lambda c: roc_auc_score(y, c)).round(5).to_string())
    print("\nSpearman correlation of OOF predictions:"); print(P.corr("spearman").round(4).to_string())
    print("\nRank-average of pairs (CV AUC):")
    R = P.apply(rankdata)
    for a, b in itertools.combinations(names, 2):
        best = max(roc_auc_score(y, P[a]), roc_auc_score(y, P[b]))
        avg = roc_auc_score(y, R[a] + R[b])
        print(f"  {a} + {b}: {avg:.5f} ({avg - best:+.5f} vs best single)")


if __name__ == "__main__":
    main(sys.argv[1:])
