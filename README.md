# Kaggle Playground S6E11 — Predicting Credit Card Fraud

https://www.kaggle.com/competitions/playground-series-s6e11 · target `is_fraud` · metric ROC-AUC


| Document | What's in it |
|---|---|
| `docs/eda.md` | EDA findings and the hypotheses they raised |
| `docs/decisions.md` | Decision record (D001–D018): context → options → decision → evidence and papers |
| `docs/experiments.md` | One row per experiment: change, hypothesis, CV, LB, kept?, takeaway |
| `docs/story.md` | Narrative of the whole project (for the presentation) |
| `reports/figures/` | Presentation figures (`make figures`) |

## Setup
```bash
conda create -n fraud python=3.11 && conda activate fraud   # never use base conda
make install
# Kaggle API token in ~/.kaggle/, competition rules accepted
make data
```

## Workflow
Each experiment is a folder `experiments/expNN_name/` holding a `config.json`; running it writes `metrics.json` (per-fold AUC), `oof.npy`, `test_pred.npy`, `submission.csv` and `command.txt`.

```bash
make eda                                                   # EDA profile
make run EXP=experiments/exp06_optuna                      # 5-fold CV, two-pass fixed-tree protocol (D009)
make compare A=experiments/<old> B=experiments/<new> X=0   # paired fold deltas + Nadeau-Bengio corrected t-test (D008); X = folds to exclude
make tune EXP=experiments/exp07_xgb_optuna M=xgb N=100     # Optuna TPE on fold 0 (D013)
conda run -n fraud python -m src.diversity experiments/A experiments/B ...   # OOF correlation and pair averages
conda run -n fraud python -m src.stack experiments/OUT experiments/A ...     # logistic stack on log-odds (D016)
make figures                                               # regenerate presentation figures
make submit EXP=experiments/<exp>
```

## Code
| File | Role |
|---|---|
| `src/data.py` | load and minimal model-agnostic preparation |
| `src/features.py` | feature-engineering steps (selected per config) |
| `src/resample.py` | SMOTE variants (training fold only) |
| `src/models.py` | model factory (logreg, LightGBM, XGBoost, CatBoost) + uniform fit/early stopping |
| `src/run.py` | experiment runner |
| `src/tune.py` | Optuna tuning |
| `src/compare.py`, `src/diversity.py`, `src/stack.py` | evaluation, diversity, stacking |
| `src/eda_profile.py`, `src/figures.py` | EDA and figures |
