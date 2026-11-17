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

## D014 — Model diversity (2026-11-12)
- **M1 models:** XGBoost (Chen & Guestrin, 2016, KDD) + CatBoost (Prokhorenkova et al., 2018, *CatBoost: unbiased boosting with categorical features*, NeurIPS). An MLP (Gorishniy et al., 2021) was not chosen.
- **M2 tuning:** the same D013 protocol and the **same budget as LightGBM (100 trials)**, so the comparison is fair. Search spaces (`src/tune.py`) mirror the LightGBM concepts (tree size, minimum leaf size, L1/L2, row/column sampling, split penalty), adapted to each library (XGBoost max_depth/gamma; CatBoost depth, random_strength, bagging_temperature, rsm).
- **M3 XGBoost categoricals:** native `enable_categorical` (consistent with D005). CatBoost uses its ordered target statistics.
- **M4 measurement:** each model's CV, plus the Spearman correlation of out-of-fold predictions and the AUC of pairwise rank averages (`src.diversity`), the inputs for ensembling (Kuncheva & Whitaker, 2003, *Measures of Diversity in Classifier Ensembles*, Machine Learning).
- **Compute (updated):** execution time is not a constraint, only results. XGBoost runs on GPU (`device="cuda"`, same histogram algorithm, same search space). CatBoost stays on CPU because CatBoost-GPU does not support `rsm` (feature subsampling) for this loss, and dropping it would shrink the search space.
- **Results (exp07/exp08):** XGBoost 0.94748 (significantly better than LightGBM on the unseen folds, p = 0.022); CatBoost 0.94674 (no different, p = 0.57). All three tuned GBMs chose **shallow trees** (XGBoost depth 3, CatBoost depth 4, LightGBM 24 leaves) with strong regularization, three independent confirmations that the signal is mostly low-order (D013 interpretation).
- **Diversity (Spearman of OOF):** LightGBM–XGBoost 0.990, LightGBM–CatBoost 0.984, XGBoost–CatBoost 0.984, logistic regression vs the GBMs 0.93–0.95. Rank-averaging pairs: LightGBM+CatBoost +0.00040 over its best member; any pair with XGBoost ≈ 0 (−0.00001 to −0.00003); logistic regression pairs −0.0009 to −0.0011 (too weak to help an equal-weight average).
- **Bug found:** the runner sets the tree count via `n_estimators` while the CatBoost config used `iterations`, and CatBoost rejects both at once. Fixed by using the `n_estimators` synonym; only exp08's 5-fold run was repeated (its tuning was unaffected).

## D015 — Seed averaging (2026-11-13)
- **Decision:** each tuned GBM is retrained in pass 2 with 5 seeds (42, 1, 2, 3, 4) and the predictions averaged (exp09a/b/c). Pass 1 (tree count) uses seed 42 only.
- **Rationale:** with row/column subsampling, each seed is a slightly different model, and averaging them reduces variance (Breiman, 1996, *Bagging Predictors*, Machine Learning). Compute is not a constraint.

## D016 — Stacking (2026-11-13)
- **Decision (E2):** stacking with a logistic-regression meta-model on the base models' out-of-fold predictions, evaluated on the same 5 folds (Wolpert, 1992). E4 (more varied models, e.g. an MLP) skipped for now.
- **Finding on the meta-model's input (diagnostic on exp01/06/07/08):** on raw **ranks**, the meta-model gives all 4 models nearly equal weights (4.2–4.5), and the stack (0.94713) is *worse* than XGBoost alone (0.94748). Fraud sits in the top ~2% of ranks, but on a linear rank scale the fit is driven by the 98% of non-fraud rows. On **log-odds of the probabilities**, the weights become 0.02 / 0.15 / 0.55 / 0.29 (logistic regression ≈ 0, XGBoost largest), CV 0.94752. −log(1−rank) behaves similarly (0.94747).
- **Interpretation:** a learned combination only matches the best single model (+0.00004), because the base models are 0.98–0.99 rank-correlated. Ting & Witten (1999, *Issues in Stacked Generalization*, JAIR) recommend class probabilities over hard outputs as meta-features. Here, that means a scale that preserves the confidence differences at the top.
- **Decision:** log-odds of probabilities (option A).
- **Result (exp10):** CV 0.94755 vs XGBoost 5-seed 0.94757; folds 1–4: +0.00002, p = 0.83. **Stacking doesn't beat the best single model.** The meta-model drops logistic regression (−0.001) and spreads weight across the three GBMs (0.29 / 0.41 / 0.32). Rank correlations of 0.98–0.99 leave nothing for a combination to exploit. This is what ensemble theory predicts: the gain from combining grows with diversity among comparably accurate members (Krogh & Vedelsby, 1995, *Neural Network Ensembles, Cross Validation, and Active Learning*, NeurIPS, the "ambiguity decomposition").
- **D015 result:** seed averaging improved all three GBMs on 4/4 unseen folds (LightGBM +0.00023, p = 0.036; XGBoost +0.00009, p = 0.088; CatBoost +0.00019, p = 0.058). XGBoost gains least, consistent with its weaker subsampling (subsample 0.82) giving less seed-to-seed variance.

## D017 — Submissions and final selection (2026-11-13)
- **Submitted:** exp09b (XGBoost 5-seed) and exp10 (stack).
- **Final selection plan (2 allowed):** exp09b + exp10 (or their exp11/exp12 versions if the feature drop holds). They tie on CV, so we hold the simplest model and the most diversified one, which hedges against the private test set behaving differently.

## D018 — Wrap-up plan (2026-11-13)
- (b) Retest dropping the 8 no-signal features on the final seed-averaged models (exp11a–c, stack exp12).
- (c) Presentation materials: score-evolution chart (CV vs LB), Optuna history and importance, diversity heatmap, updated README, and a narrative "story" doc linking the decisions.
- (a) Neural-net diversity: not pursued.

## D011 revisited — Dropping the 8 no-signal features on the final models (2026-11-13)
- **Result:** dropping them improved 10 of 12 (model, fold) pairs on the unseen folds 1–4: LightGBM +0.00024 (p = 0.027), XGBoost +0.00009 (p = 0.33), CatBoost +0.00012 (p = 0.34). Stack exp12 0.94769 vs exp10 0.94755, +0.00010, 4/4, p = 0.042.
- **Interpretation:** consistent with 05d (+0.00014, 4/4). Each tree samples ~50–60% of the columns (colsample/rsm, the most important hyperparameter, D013). With 8 useless columns in the pool, many trees get fewer useful features to choose from, so removing them makes column sampling more efficient. The earlier worry (D011) that other model families would use them through interactions is not supported.
- **Caveat (selection bias):** the 8 features were chosen from EDA univariate AUCs computed on the *full* training set, i.e. including each validation fold. That is a (mild) form of the selection bias described by Ambroise & McLachlan (2002, *Selection bias in gene extraction on the basis of microarray gene-expression data*, PNAS). The risk is small here: 700k rows, the 8 features have univariate AUC ≤ 0.514, and the decision is "remove noise", not "pick the best few", but it should be stated.
- **Decision:** adopt the reduced feature set (16 features) for the final models.

## D017 update — Final selection (2026-11-13)
- **Final 2 selections:** **exp11b** (XGBoost 5 seeds, reduced features: the simplest strong model) + **exp12** (log-odds stack on reduced features: the most diversified). Same reasoning as D017.
- **Leaderboard check (2026-11-13):** the public LB moved the opposite way from CV. exp11b 0.94747 vs exp09b 0.94776 (−0.0003); exp12 0.94774 vs exp10 0.94794 (−0.0002).
- **Why we still trust CV:** the public LB scores ~20% of the test set (~60k rows, ~1,000 frauds). The standard error of AUC at that size is ~0.003 (Hanley & McNeil, 1982, *The meaning and use of the area under a ROC curve*, Radiology). Even allowing for the two submissions being highly correlated, a 0.0002–0.0003 difference is well within public-LB noise. The CV evidence uses all 700k rows, paired over folds (exp12 vs exp10: 4/4 folds, p = 0.042). Following D003, decisions stay CV-based; switching on public-LB noise is how teams overfit the public LB and drop on the private one. The selection stays exp11b + exp12.
