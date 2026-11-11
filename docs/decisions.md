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
