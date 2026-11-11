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
