"""Paired comparison of two experiments on the shared CV folds (D008).

Usage: conda run -n fraud python -m src.compare experiments/expA experiments/expB

Reports per-fold deltas d_k = AUC_B - AUC_A and the Nadeau & Bengio (2003)
corrected resampled t-test:  t = mean(d) / sqrt((1/k + n_test/n_train) * var(d)),
df = k-1.  For K-fold CV, n_test/n_train = 1/(K-1).
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy import stats


def compare(a, b):
    fa = np.array(json.loads((Path(a) / "metrics.json").read_text())["fold_auc"])
    fb = np.array(json.loads((Path(b) / "metrics.json").read_text())["fold_auc"])
    d, k = fb - fa, len(fa)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + 1 / (k - 1)) * var) if var > 0 else np.inf * np.sign(d.mean())
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    print(f"A = {a}\nB = {b}")
    print("fold deltas (B-A):", " ".join(f"{x:+.5f}" for x in d))
    print(f"mean delta {d.mean():+.5f} | improved on {(d > 0).sum()}/{k} folds | corrected t = {t:.2f}, p = {p:.4f}")
    print("verdict:", "significant (p<0.05)" if p < 0.05 else "not significant — treat as noise")


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
