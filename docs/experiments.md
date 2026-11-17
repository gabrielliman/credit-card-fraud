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
| 04d | 2026-11-12 | 04c + round integer columns after SMOTENC | 04c failed because of non-integer synthetic values | 0.89117 | – | no | Identical to 04c: SMOTENC already outputs integers, so **hypothesis refuted** (see D010 follow-up) |
| 05a | 2026-11-12 | + velocity residual (OLS on txn_count, log hours) + txn_per_hour | velocity hides extra signal | 0.94574 | – | no | −0.00016, 1/5 folds, p = 0.31 → noise |
| 05b | 2026-11-12 | + n_flags (count of 6 risk flags) | Trees find additive sums slowly | 0.94597 | – | no | +0.00008, 2/5 folds, p = 0.70 → noise |
| 05c1 | 2026-11-12 | + frequency encoding of the 5 categoricals | Level frequency carries signal | 0.94587 | – | no | −0.00003, p = 0.88 → noise |
| 05c2 | 2026-11-12 | 6 few-valued numerics treated as categorical | Unordered treatment fits better | 0.94483 | – | no | −0.00107, 0/5 folds, p = 0.026 → significantly **worse**: the order of values carries information |
| 05d | 2026-11-12 | drop the 8 features with univariate AUC ≈ 0.50 | Removing noise helps | 0.94604 | – | candidate | +0.00014, 4/5 folds, p = 0.12 → not significant, but same accuracy with 16 instead of 24 features |
| 06 | 2026-11-12 | Optuna TPE, 100 trials on fold 0 (D013): num_leaves 24, min_child 543, colsample 0.59, subsample 0.63, reg_alpha 0.85, min_split_gain 0.58; 1082 trees at lr 0.02 | Default capacity/regularization isn't optimal | 0.94685 ± 0.0031 | 0.94726 | **yes (new reference)** | All 5 folds: +0.00093, 5/5, p = 0.037. **Honest (folds 1–4): +0.00083, 4/4, p = 0.078**: consistent but not significant with only 4 folds. Best trial at #28, flat afterwards |
| 07 | 2026-11-12 | XGBoost, Optuna 100 trials (D013/D014), GPU: max_depth 3, colsample 0.57, subsample 0.82, gamma 0.92; 2940 trees | Another GBM implementation with different trees fits better | **0.94748** ± 0.0032 | 0.94787 | **yes (best single)** | vs exp06 on folds 1–4: +0.00068, 4/4, p = 0.022 (significantly better). Best = **depth-3 trees**, even simpler than LightGBM's 24 leaves |
| 08 | 2026-11-12 | CatBoost, Optuna 100 trials, CPU: depth 4, l2 4.2, rsm 0.63, min_data_in_leaf 310; 6107 trees | Ordered target stats + symmetric trees | 0.94674 ± 0.0030 | – | for ensemble | vs exp06 on folds 1–4: −0.00008, 1/4, p = 0.57 (equal). Most different from the other GBMs (see diversity) |
| 09a | 2026-11-13 | exp06 (LightGBM) averaged over 5 seeds | Seed averaging reduces variance | 0.94706 | – | yes | folds 1–4: +0.00023, 4/4, p = 0.036 (significantly better) |
| 09b | 2026-11-13 | exp07 (XGBoost) averaged over 5 seeds | same | **0.94757** | 0.94776 | yes (best single) | folds 1–4: +0.00009, 4/4, p = 0.088 (consistent but small) |
| 09c | 2026-11-13 | exp08 (CatBoost) averaged over 5 seeds | same | 0.94691 | – | yes | folds 1–4: +0.00019, 4/4, p = 0.058 (consistent but small) |
| 10 | 2026-11-13 | Stack: logistic regression on log-odds of exp01 + 09a + 09b + 09c (D016) | A learned combination beats the best single model | 0.94755 ± 0.0031 | 0.94794 | tie | vs 09b on folds 1–4: +0.00002, 2/4, p = 0.83 (noise). Weights: LogReg −0.001, LGBM 0.29, XGB 0.41, CatBoost 0.32 |
| 11a | 2026-11-13 | 09a (LightGBM 5 seeds) minus the 8 no-signal features | 05d retest on final models: removing noise helps | 0.94735 | – | candidate | folds 1–4: +0.00024, 4/4, p = 0.027 (significantly better) |
| 11b | 2026-11-13 | 09b (XGBoost 5 seeds) minus the 8 features | same | **0.94767** | 0.94747 | candidate | folds 1–4: +0.00009, 3/4, p = 0.33 |
| 11c | 2026-11-13 | 09c (CatBoost 5 seeds) minus the 8 features | same | 0.94702 | – | candidate | folds 1–4: +0.00012, 3/4, p = 0.34 |
| 12 | 2026-11-13 | Stack (log-odds) of exp01 + 11a + 11b + 11c | Better members → better stack | **0.94769** | 0.94774 | candidate (best) | vs exp10 on folds 1–4: +0.00010, 4/4, p = 0.042 (significantly better). Weights LogReg 0.06, LGBM 0.31, XGB 0.36, CatBoost 0.27 |
