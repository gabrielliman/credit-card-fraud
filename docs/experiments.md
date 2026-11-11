# Experiment Log

CV = stratified 5-fold ROC-AUC (D003). LB = public leaderboard.

| # | date | change | hypothesis | CV AUC | LB | kept? | takeaway |
|---|------|--------|-----------|--------|----|-------|----------|
| 01 | 2026-11-11 | Logistic regression, one-hot + scaling | Linear floor | 0.94187 ± 0.0034 | – | baseline | Strong for a linear model: most signal is roughly additive in log-odds |
| 02 | 2026-11-11 | LightGBM defaults, native categoricals | Non-linearity/interactions add signal | 0.94391 ± 0.0036 | – | baseline | +0.0020 over 01, better on 5/5 folds (paired); gap is small → limited interaction signal at default capacity |
