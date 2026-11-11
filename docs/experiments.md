# Experiment Log

CV = stratified 5-fold ROC-AUC (D003). LB = public leaderboard.

| # | date | change | hypothesis | CV AUC | LB | kept? | takeaway |
|---|------|--------|-----------|--------|----|-------|----------|
| 01 | 2026-11-11 | Logistic regression, one-hot + scaling | Linear floor | 0.94187 ± 0.0034 | 0.94245 | baseline | Strong for a linear model: most signal is roughly additive in log-odds |
| 02 | 2026-11-11 | LightGBM defaults, native categoricals | Non-linearity/interactions add signal | 0.94391 ± 0.0036 | 0.94557 | baseline | +0.0020 over 01, better on 5/5 folds (paired); gap is small → limited interaction signal at default capacity |
| 03 | 2026-11-11 | LightGBM lr 0.1→0.02, up to 10k trees, early stopping (patience 300) | exp02 was under-fitted (too little capacity) | 0.94594 ± 0.0034 | – | yes | +0.0020 over 02, 5/5 folds; best iterations 777–1195 → 100 trees was far too few. Most of the gap to logreg was model size, not missing interactions |
