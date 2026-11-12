# Decision Record

Each entry: context → options → decision → rationale/evidence (with references).

## D001 — Environment (2026-11-11)
- **Context:** need a reproducible Python setup.
- **Options:** (A) dedicated conda env; (B) base conda.
- **Decision:** A — conda env `fraud` (Python 3.11) + `requirements.txt`.
- **Rationale:** isolation and reproducibility; base env shared with other projects.

## D002 — Data sources (2026-11-11)
- **Context:** Playground data is synthetic, generated from an original real dataset; many competitors append the original data.
- **Options:** (A) competition data only; (B) also use the original dataset.
- **Decision:** A — competition data only.
- **Rationale:** fairness / self-contained solution; all gains must come from our own modelling.

## D003 — Validation scheme (2026-11-11)
- **Context:** 699,960 train rows, 1.67% positives, metric ROC-AUC, no timestamp.
- **Options:** (A) stratified 5-fold; (B) repeated stratified K-fold; (C) time-based split (not possible — no time column).
- **Decision:** A — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, same folds for every experiment.
- **Rationale:** stratification keeps the positive rate per fold constant, reducing variance of the estimate (Kohavi, 1995, *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection*, IJCAI). 5 folds ≈ 11.7k positives per validation fold, enough for a stable AUC at half the cost of 10-fold. Fixed folds make experiments comparable and enable stacking later.
- **To verify:** CV ↔ public LB correlation over the first submissions.

## D004 — Baselines (2026-11-11)
- **Options:** (A) logistic regression + default LightGBM; (B) LightGBM only.
- **Decision:** A.
- **Rationale:** a linear floor quantifies what non-linearity buys. Trees are the expected state of the art on tabular data (Grinsztajn et al., 2022, *Why do tree-based models still outperform deep learning on tabular data?*, NeurIPS; Ke et al., 2017, *LightGBM*, NeurIPS).
- **Result:** exp01 0.94187 vs exp02 0.94391.

## D005 — Categorical handling for LightGBM (2026-11-11)
- **Options:** (A) native categorical splits; (B) one-hot; (C) target encoding (Micci-Barreca, 2001).
- **Decision:** A for now; C kept as a later experiment.
- **Rationale:** few levels per category (≤12), so native splits are simple and have no leakage risk.

## D006 — Experiment tracking (2026-11-11)
- **Decision:** each experiment lives in `experiments/expNN_name/` with `config.json`, `metrics.json` (per-fold AUC), `oof.npy`, `test_pred.npy`, `submission.csv` and `command.txt`; it runs via `make run EXP=...`. The `.npy` and CSV files are gitignored and can be regenerated.
- **Rationale:** reproducibility; out-of-fold predictions enable stacking (Wolpert, 1992, *Stacked Generalization*) and per-fold paired comparisons.

## D003 check — CV vs public LB (2026-11-11)
- exp01: CV 0.94187 → LB 0.94245; exp02: CV 0.94391 → LB 0.94557. Same ranking, LB about 0.001 higher. CV tracks the LB, as the adversarial validation suggested. The public LB covers only part of the test set, so small differences there are noise.

## D007 — Model sizing for LightGBM (2026-11-11)
- **Decision:** learning rate 0.02 with early stopping (patience 300 rounds, monitoring AUC) instead of a fixed 100 trees.
- **Rationale:** shrinkage plus more boosting iterations generalizes better (Friedman, 2001, *Greedy Function Approximation: A Gradient Boosting Machine*, Ann. Statist.).
- **Caveat:** early stopping watches the same validation fold that we score, so CV is slightly optimistic (the stopping point is picked using that fold). With 140k validation rows and stopping points that range widely (777–1195), the bias should be tiny. Alternative if needed: a separate inner validation split, or retraining with a fixed number of trees.

## D008 — Deciding whether a change helped (2026-11-11)
- **Options:** (A) paired fold deltas + corrected resampled t-test; (B) "better on overall CV and on ≥ 4/5 folds"; (C) overall CV only.
- **Decision:** A, implemented as `make compare A=... B=...`.
- **Rationale:** experiments share the same folds, so per-fold differences cancel fold difficulty. A plain t-test on CV folds gives too many false positives because training sets overlap (Dietterich, 1998, *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*, Neural Computation). Nadeau & Bengio (2003, *Inference for the Generalization Error*, Machine Learning) correct the variance by a factor of (1/k + n_test/n_train).
- **Results so far:** exp01→02 +0.00208, 5/5 folds, p = 0.011; exp02→03 +0.00203, 5/5 folds, p = 0.005.

## D007 follow-up — Early-stopping patience (2026-11-11)
- **Diagnostic:** trained 4000 trees with no early stopping on folds 0 and 1 and recorded validation AUC at every iteration.
- **Result:** the best iteration overall (1195 and 926) is exactly where patience = 300 stopped. After that AUC steadily falls (fold 0: 0.94409 → 0.94180 at 4000 trees), which is overfitting. **Patience 300 is enough, so there's no reason to increase it.**

## D009 — Tree-count protocol (2026-11-12)
- **Options:** (A) early-stop on the scored fold; (B) inner validation split; (C) two passes: early-stop to find the best iterations, then retrain every fold with n = median(best iterations).
- **Decision:** C for all experiments (each run takes ~2 minutes, so the extra pass is cheap).
- **Rationale:** no fold's score depends on early stopping run on that same fold (Cawley & Talbot, 2010, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR).
- **Evidence:** exp03 (A) 0.94594 vs exp03b (C) 0.94590, so the bias in A was about 0.00004.

## Bug found — early stopping on the wrong metric (2026-11-12)
- **Symptom:** exp04a/b stopped after 2 and 14 trees.
- **Cause:** by default LightGBM also tracks binary_logloss, and early stopping fires when *any* tracked metric stops improving. Class weights inflate the predicted probabilities, so validation log loss got worse from tree 2 (0.076 → 0.281) while AUC kept rising (0.896 → 0.942).
- **Fix:** `metric="auc"` in the LightGBM factory. exp03/03b weren't affected (their stopping points match the AUC peak in the D007 diagnostic), and all were rerun after the fix.

## D010 — Class imbalance handling (2026-11-12)
- **Options tested:** none (exp03b), scale_pos_weight = 58.9 (04a), scale_pos_weight = 7.67 (04b), SMOTENC to 1:1 on the training fold (04c; Chawla et al., 2002).
- **Decision:** **no imbalance correction**.
- **Evidence:** 04a significantly worse (−0.0011, p = 0.015); 04b no different (−0.0003, p = 0.13); 04c much worse (−0.055).
- **Interpretation:** AUC depends only on ranking, and reweighting mostly shifts scores up rather than changing their order (Elkan, 2001, *The Foundations of Cost-Sensitive Learning*). It also makes each positive count more, which raises variance. This matches van den Goorbergh et al. (2022, *The harm of class imbalance corrections for risk prediction models*, JAMIA): no gain in AUC and worse calibration. With 9.3k positives per training fold, the minority class is rare but not *scarce*, which is the situation where resampling usually doesn't help (He & Garcia, 2009, *Learning from Imbalanced Data*, IEEE TKDE).
- **SMOTE hypothesis (untested):** SMOTENC interpolates discrete numeric features (cvv_retry_count, txn_count_last_24h, customer_age, hour…) and produces non-integer values that never appear in real data. Trees can learn "non-integer value ⇒ fraud", a shortcut that never fires on real validation rows.

## D010 follow-up — why SMOTE failed (2026-11-12)
- **Hypothesis "non-integer synthetic values" was refuted:** 0% of synthetic rows have non-integer values (SMOTENC keeps integer dtypes); exp04d (rounding) = exp04c exactly.
- **Real cause:** a classifier separates synthetic from real positives with AUC 0.907. SMOTENC sets each categorical/flag value by the **majority vote of the k=5 nearest neighbours** (Chawla et al., 2002). Rare flags lose that vote almost every time: among real frauds is_foreign = 13.3%, ip_country_mismatch = 16.1%, used_vpn = 13.6%; among synthetic frauds 1.4%, 2.2%, 1.7%. Majority vote also over-represents the dominant level (No Authentication 37% → 48%).
- **Effect:** ~540k synthetic "frauds" without the strongest fraud signals swamp 9.3k real ones, teaching the model a distorted picture of fraud, hence −0.055 AUC.
- **Lesson:** oversampling methods have assumptions that break on mixed data with rare binary indicators. Check that synthetic data matches the real distribution (a real-vs-synthetic classifier) before trusting it.

## D011 — Feature engineering round 1 (2026-11-12)
- **Tested (each alone on exp03b):** velocity decomposition (05a), flag count (05b), frequency encoding (05c1), few-valued numerics as categorical (05c2), dropping 8 no-signal features (05d).
- **Results:** none significantly better; 05c2 significantly worse (p = 0.026).
- **Interpretation:** at ~900 trees, LightGBM already learns these transformations (sums, ratios, frequencies) on its own; Heaton (2016) found ratio/count features help neural nets and linear models far more than tree ensembles. Treating ordered values as unordered categories throws away their order and adds split candidates (more variance), so it hurts.
- **Decision:** keep all features; exp03b remains the reference. 05d (drop 8 features) is **not** adopted for now: not significant (p = 0.12), and other model families (CatBoost/XGBoost) might use those features through interactions. Re-test on the final models.

## D012 — Roadmap after FE round 1 (2026-11-12)
- **Options:** (a) hyperparameter tuning; (b) model diversity (XGBoost, CatBoost); (c) ensembling; (d) synthetic-artifact search.
- **Decision:** a → b → c. **No (d)**: it would exploit the data generator rather than model fraud, so the result is less general.

## D013 — Hyperparameter tuning protocol (2026-11-12)
- **Search method:** Optuna TPE (Bergstra et al., 2011, *Algorithms for Hyper-Parameter Optimization*, NeurIPS; Akiba et al., 2019, *Optuna*, KDD). Alternatives: random search (Bergstra & Bengio, 2012, JMLR) or grid search.
- **Trial scoring:** train on folds 1–4, score on fold 0, learning rate 0.05 with early stopping. The best params are rerun on all 5 folds at lr 0.02 with the D009 protocol. **Fold 0 is "seen", so the honest gain is reported on folds 1–4** (Cawley & Talbot, 2010). Rejected: 3-fold CV per trial (3× slower), nested CV (Varma & Simon, 2006; ~5× the cost).
- **Search space:** num_leaves 15–255 (log), min_child_samples 10–1000 (log), reg_lambda and reg_alpha 1e-3–10 (log), colsample_bytree 0.3–1, subsample 0.5–1, min_split_gain 0–1.
- **Budget:** 100 trials. The study is saved in `study.db` for history and fANOVA importance plots (Hutter et al., 2014).
- **Result (exp06):** CV 0.94685. Gain over exp03b on the unseen folds 1–4 is +0.00083, improving on 4/4 folds, p = 0.078 (+0.00093, p = 0.037 including fold 0). The sign is consistent, but with 4 folds the test has little power.
- **What tuning learned** (fANOVA importance): colsample_bytree 0.54, reg_alpha 0.13, min_child_samples 0.09, others ≤ 0.07. The top-5 trials all use **small trees (15–24 leaves), large leaves (194–543 samples) and ~50% of features per tree**, i.e. *simpler and more regularized than the defaults*. That fits the EDA: the signal is mostly additive (logistic regression already reaches 0.942), so deep interactions mostly fit noise. The search levelled off at trial 28 (0.94579), so 100 trials were enough.
