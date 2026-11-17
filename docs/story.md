# The story of this project

A narrative of *why* each step happened, written for the presentation. Details and evidence: `decisions.md` (Dxxx) and `experiments.md` (expNN). Figures: `reports/figures/`.

## 1. Rules of the game
- Kaggle Playground S6E11: predict `is_fraud` (1.67% positives) from 25 synthetic features; metric ROC-AUC.
- Self-imposed constraints: competition data only (D002, fairness); every design decision is a documented choice between options, backed by literature.

## 2. Understand the data before modelling (`eda.md`, fig 00)
- Clean data: no missing values, no duplicates.
- **Train and test are indistinguishable**: an adversarial classifier gets AUC 0.500. → We can trust cross-validation (D003).
- Signal comes from merchant risk, velocity, CVV retries, authentication method and a few rare red-flag indicators. 8 features show no univariate signal.

## 3. Build a trustworthy measuring stick (D003, D008, D009)
- Stratified 5-fold CV with fixed folds, so every experiment is comparable fold by fold.
- **Is a gain real?** Paired fold deltas plus the Nadeau & Bengio (2003) corrected t-test: CV folds overlap, so the naive t-test is too optimistic (Dietterich, 1998).
- **No peeking:** tree counts are chosen in a first pass and then fixed (two-pass protocol). It cost only 0.00004 AUC, which is evidence the cleaner protocol is free.
- Validation: CV and the public leaderboard rank every submission the same way (fig 02).

## 4. Baselines: how far does simple go? (exp01–03)
- Logistic regression 0.9419 vs default LightGBM 0.9439.
- First surprise: the gap was **model size, not missing interactions**. Default LightGBM stops at 100 trees, while early stopping wanted ~900 (exp03: 0.9459).

## 5. Testing conventional wisdom: class imbalance (exp04, D010)
- Class weights hurt (−0.0011, significant); milder weights didn't help; SMOTE destroyed performance (−0.055).
- Why: AUC only depends on ranking, and reweighting shifts scores without reordering them (Elkan, 2001; van den Goorbergh et al., 2022).
- **The SMOTE investigation:** my first hypothesis (non-integer synthetic values) was *refuted* by testing it. The real cause: SMOTENC fills categories and flags by **majority vote of 5 neighbours**, so rare red flags (13–16% among real frauds) almost never win. The synthetic frauds came out 10× less "flagged" than real ones (shown by a real-vs-synthetic classifier, AUC 0.907).
- Bonus: suspicious results (training stopping after 2 trees) exposed a bug: early stopping was also watching log loss.

## 6. Feature engineering that didn't work (exp05, D011)
- Velocity decomposition, flag counts and frequency encoding were all noise. Treating ordered values as categories was significantly *worse*.
- Lesson: tree ensembles already learn sums, ratios and frequencies on their own (Heaton, 2016). Negative results are results.

## 7. Tuning, and what it revealed (exp06–08, D013/D014, figs 03–04)
- Optuna TPE, 100 trials per model, tuned on fold 0 only; honest gains are reported on folds 1–4.
- **Three libraries, one answer:** LightGBM (24 leaves), XGBoost (depth 3) and CatBoost (depth 4) all chose **shallow, heavily regularized trees**, and column subsampling was the most important hyperparameter for all three (~0.54).
- That matches the baseline story: the fraud signal is mostly low-order. XGBoost became the best single model (0.9475).

## 8. Ensembling: why it barely helped (exp09–10, D015/D016, fig 05)
- Seed averaging: small but consistent gains (+0.0001 to +0.0002, 4/4 folds).
- Stacking on raw ranks *hurt* (it squashes the top 2% where fraud lives). On log-odds it matched XGBoost exactly.
- Why: the GBMs are 0.98–0.99 rank-correlated, so there's nothing to combine. Ensemble gains need diverse, comparably accurate members (Krogh & Vedelsby, 1995).

## 9. Revisiting a postponed decision (exp11–12)
- Dropping the 8 no-signal features was postponed in round 1 (not significant). Retested on the final models, it helped all three (LightGBM significantly), and the stack became the best result: **CV 0.94769** (exp12, significantly better than exp10).
- Plausible mechanism: with ~50% column sampling per tree, useless columns waste the sample.
- Caveat stated openly: the 8 were picked from EDA on the full training set, a mild selection bias (Ambroise & McLachlan, 2002).
- Then the public LB went slightly *down* (−0.0002 to −0.0003) while CV went up. We kept the CV-based decision: that difference is far below public-LB noise (~0.003 AUC standard error on ~1,000 frauds; Hanley & McNeil, 1982), and trusting the larger, paired CV evidence is the point of section 3.

## 10. Results and takeaways
| milestone | CV | public LB |
|---|---|---|
| LogReg baseline (exp01) | 0.94187 | 0.94245 |
| LightGBM tuned (exp06) | 0.94685 | 0.94726 |
| XGBoost tuned (exp07) | 0.94748 | 0.94787 |
| Stack (exp10) | 0.94755 | 0.94794 |
| Stack, −8 features (exp12) | 0.94769 | 0.94774 |

Takeaways:
1. Invest in the measuring stick first; every later conclusion depends on it.
2. Test conventional wisdom (imbalance, SMOTE, feature engineering): here, most of it didn't hold.
3. Refuted hypotheses and bugs found through odd results are part of the process, not failures.
4. When three different models agree that simple is better, believe the data.
