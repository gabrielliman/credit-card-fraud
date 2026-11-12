# Experiment Log

CV = stratified 5-fold ROC-AUC (D003). LB = public leaderboard.

| # | date | change | hypothesis | CV AUC | LB | kept? | takeaway |
|---|------|--------|-----------|--------|----|-------|----------|
| 01 | 2026-11-11 | Logistic regression, one-hot + scaling | Linear floor | 0.94187 ± 0.0034 | 0.94245 | baseline | Strong for a linear model: most signal is roughly additive in log-odds |
| 02 | 2026-11-11 | LightGBM defaults, native categoricals | Non-linearity/interactions add signal | 0.94391 ± 0.0036 | 0.94557 | baseline | +0.0020 over 01, better on 5/5 folds (paired); gap is small → limited interaction signal at default capacity |
| 03 | 2026-11-11 | LightGBM lr 0.1→0.02, up to 10k trees, early stopping (patience 300) | exp02 was under-fitted (too little capacity) | 0.94594 ± 0.0034 | – | yes | +0.0020 over 02, 5/5 folds; best iterations 777–1195 → 100 trees was far too few. Most of the gap to logreg was model size, not missing interactions |
| 03b | 2026-11-12 | exp03 under the two-pass fixed-tree protocol (D009): 921 trees | Removes per-fold early-stopping peeking | 0.94590 ± 0.0034 | – | **reference** | Only −0.00004 vs exp03, so the early-stopping bias really was tiny |
| 04a | 2026-11-12 | exp03b + scale_pos_weight 58.9 | Class reweighting helps rare-class ranking | 0.94482 | – | no | −0.00106, worse on 5/5 folds, p = 0.015 (significantly worse) |
| 04b | 2026-11-12 | exp03b + scale_pos_weight 7.67 | Milder reweighting | 0.94557 | – | no | −0.00032, worse on 4/5 folds, p = 0.13 (noise at best) |
| 04c | 2026-11-12 | exp03b + SMOTENC 1:1 on training fold | Synthetic positives help | 0.89117 | – | no | −0.0547, worse on 5/5 folds: badly hurts. Stopped at ~120 trees, so it overfits synthetic points quickly |
