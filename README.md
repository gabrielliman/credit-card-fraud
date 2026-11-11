# Kaggle Playground S6E11 — Predicting Credit Card Fraud

https://www.kaggle.com/competitions/playground-series-s6e11

## Setup
```bash
make install                      # deps
# put your API token at ~/.kaggle/kaggle.json (chmod 600) and accept the competition rules on Kaggle
make data                         # downloads into data/
```

## Workflow
- `notebooks/01_eda.ipynb` — quick EDA
- `make train` — 5-fold LightGBM baseline; prints CV ROC-AUC, writes `submissions/lgb_cv*.csv`
- `make submit FILE=submissions/<file>.csv MSG="baseline"`

ID/target columns are inferred from `sample_submission.csv`. Metric assumed to be ROC-AUC —
check the Evaluation tab and adjust `src/train.py` if different.
