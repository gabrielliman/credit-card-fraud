"""Presentation figures (D018). Usage: conda run -n fraud python -m src.figures
Writes PNGs to reports/figures/. Colors: validated reference categorical palette (light mode),
fixed slot order: 1 blue = CV / primary, 2 orange = public LB / secondary, 3 aqua = third series."""
import json, re
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from .config import ROOT

OUT = ROOT / "reports" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left"})


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=200); plt.close(fig); print("wrote", OUT / name)


def experiment_log():
    rows = []
    for line in (ROOT / "docs" / "experiments.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8 or not re.match(r"^\d+\w*$", cells[0]):
            continue
        cv = re.search(r"0\.\d{5}", cells[4]); lb = re.search(r"0\.\d{5}", cells[5])
        rows.append(dict(exp=cells[0], change=cells[2], cv=float(cv.group()) if cv else np.nan,
                         lb=float(lb.group()) if lb else np.nan, kept=cells[6]))
    return pd.DataFrame(rows)


SHORT = {"01": "LogReg baseline", "02": "LightGBM defaults", "03": "+ lr 0.02, early stop", "03b": "two-pass protocol",
         "04a": "class weight 59", "04b": "class weight 7.7", "04c": "SMOTE", "04d": "SMOTE + rounding",
         "05a": "+ velocity features", "05b": "+ flag count", "05c1": "+ freq. encoding", "05c2": "low-card as cat.",
         "05d": "drop 8 features", "06": "LightGBM tuned", "07": "XGBoost tuned", "08": "CatBoost tuned",
         "09a": "LightGBM 5 seeds", "09b": "XGBoost 5 seeds", "09c": "CatBoost 5 seeds", "10": "Stack (log-odds)",
         "11a": "LGBM 5s, −8 feats", "11b": "XGB 5s, −8 feats", "11c": "CatBoost 5s, −8 feats", "12": "Stack, −8 feats"}


def fig_evolution(df):
    # Dot plot: one row per experiment, CV (blue) and public LB (orange) on one AUC axis.
    d = df[df.cv > 0.93].iloc[::-1].reset_index(drop=True)  # SMOTE (0.891) is off-scale: noted in caption
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.scatter(d.cv, y, s=46, color=BLUE, zorder=3, label="CV AUC (5-fold)", edgecolors=SURF, linewidths=2)
    ax.scatter(d.lb, y, s=46, color=ORANGE, marker="D", zorder=3, label="Public LB AUC", edgecolors=SURF, linewidths=2)
    ax.set_yticks(y, [f"{e}  {SHORT.get(e, e)}" for e in d.exp])
    ref = df.loc[df.exp == "03b", "cv"].item()
    ax.axvline(ref, color=MUTED, lw=1, ls="--"); ax.text(ref, len(d) - 0.4, " exp03b reference", color=MUTED, fontsize=9, va="bottom")
    ax.set_xlabel("ROC-AUC"); ax.grid(axis="y", visible=False)
    ax.set_title("Score evolution across experiments")
    ax.legend(loc="lower left", frameon=False)
    fig.text(0.01, 0.005, "SMOTE runs (04c/04d, CV 0.891) omitted: off-scale.", color=MUTED, fontsize=8)
    save(fig, "01_score_evolution.png")


def fig_cv_vs_lb(df):
    # Left: all submissions. Right: zoom on the top cluster so its labels stay readable.
    d = df.dropna(subset=["lb"])
    top = d[d.cv > 0.9474]
    fig, (ax, az) = plt.subplots(1, 2, figsize=(11, 5))
    for a_, sub, pad in ((ax, d, 0.0008), (az, top, 0.00015)):
        lo, hi = min(sub.cv.min(), sub.lb.min()) - pad, max(sub.cv.max(), sub.lb.max()) + pad
        a_.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        a_.text(hi, hi, "CV = LB ", ha="right", va="bottom", color=MUTED, fontsize=9)
        a_.scatter(sub.cv, sub.lb, s=60, color=BLUE, zorder=3, edgecolors=SURF, linewidths=2)
        a_.set_xlim(lo, hi); a_.set_ylim(lo, hi); a_.set_aspect("equal")
        a_.set_xlabel("CV AUC")
    for _, r in d[d.cv <= 0.9474].iterrows():
        ax.annotate(SHORT.get(r.exp, r.exp), (r.cv, r.lb), xytext=(6, -10), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.annotate("top cluster →\n(right panel)", (top.cv.mean(), top.lb.mean()), xytext=(-12, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5, color=MUTED)
    nudge = {"09b": 8, "12": -8}  # these two sit close together
    for _, r in top.iterrows():
        az.annotate(f"{r.exp} {SHORT.get(r.exp, r.exp)}", (r.cv, r.lb), xytext=(7, nudge.get(r.exp, 0)), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    az.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4)); az.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_ylabel("Public LB AUC"); ax.set_title("CV tracks the leaderboard"); az.set_title("Top submissions (zoom)")
    fig.text(0.01, 0.005, "Differences inside the zoom (≤ 0.0005) are below public-LB noise (SE ≈ 0.003, Hanley & McNeil 1982).", color=MUTED, fontsize=8)
    save(fig, "02_cv_vs_lb.png")


def fig_optuna():
    studies = [("LightGBM", "exp06_optuna", "lgbm", BLUE), ("XGBoost", "exp07_xgb_optuna", "xgb", ORANGE),
               ("CatBoost", "exp08_catboost_optuna", "catboost", AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    imps, lims = {}, []
    for ax, (name, exp, sname, col) in zip(axes, studies):
        s = optuna.load_study(study_name=sname, storage=f"sqlite:///{ROOT / 'experiments' / exp / 'study.db'}")
        v = np.array([t.value for t in s.trials if t.value is not None])
        ax.scatter(np.arange(len(v)), v, s=12, color=col, alpha=0.55, edgecolors="none")
        ax.plot(np.maximum.accumulate(v), color=INK, lw=2, label="best so far")
        ax.set_title(name); ax.set_xlabel("trial")
        lims.append((np.percentile(v, 5) - 0.0005, v.max() + 0.0003))
        imps[name] = optuna.importance.get_param_importances(s)
    axes[0].set_ylim(min(l for l, _ in lims), max(h for _, h in lims))
    axes[0].set_ylabel("fold-0 AUC"); axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Optuna TPE: 100 trials per model; gains level off after ~30 trials", x=0.01, ha="left", fontweight="bold")
    save(fig, "03_optuna_history.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (name, *_ , col) in zip(axes, studies):
        imp = pd.Series(imps[name]).sort_values()
        ax.barh(imp.index, imp.values, color=col, height=0.6)
        for i, v in enumerate(imp.values):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
        ax.set_title(name); ax.set_xlim(0, imp.max() * 1.25); ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("fANOVA importance (share of explained variance in AUC)")
    save(fig, "04_param_importance.png")


def fig_diversity():
    from scipy.stats import rankdata  # noqa
    names = {"exp01_logreg": "LogReg", "exp11a_lgbm_drop": "LightGBM", "exp11b_xgb_drop": "XGBoost", "exp11c_catboost_drop": "CatBoost"}
    P = pd.DataFrame({v: np.load(ROOT / "experiments" / k / "oof.npy") for k, v in names.items()})
    C = P.corr("spearman")
    fig, ax = plt.subplots(figsize=(6.4, 5))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#e8f1fc", "#9ec3f0", BLUE, "#123f78"])  # one hue, light -> dark
    im = ax.imshow(C.values, cmap=cmap, vmin=0.92, vmax=1.0)
    ax.set_xticks(range(len(C)), C.columns); ax.set_yticks(range(len(C)), C.index); ax.grid(False)
    for i in range(len(C)):
        for j in range(len(C)):
            v = C.values[i, j]; ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, color="white" if v > 0.975 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Spearman ρ of OOF predictions", color=INK2)
    ax.set_title("GBMs agree almost perfectly")
    fig.tight_layout(); fig.subplots_adjust(left=0.2)
    fig.savefig(OUT / "05_model_diversity.png", dpi=200); plt.close(fig); print("wrote", OUT / "05_model_diversity.png")


def fig_signal():
    from .data import load, prepare
    from sklearn.metrics import roc_auc_score
    tr, te, _, i, t = load(); X, y, _, cats = prepare(tr, te, i, t)
    auc = {}
    for c in X.columns:
        if c in cats:
            s = X[c].map(y.groupby(X[c], observed=True).mean()).astype(float)  # level fraud rate (descriptive)
        else:
            s = X[c].astype(float)
        a = roc_auc_score(y, s); auc[c] = max(a, 1 - a)
    s = pd.Series(auc).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    from .features import NO_SIGNAL
    cols = [MUTED if c in NO_SIGNAL else BLUE for c in s.index]
    ax.barh(s.index, s.values - 0.5, left=0.5, color=cols, height=0.6)
    for k, v in enumerate(s.values):
        ax.text(v + 0.003, k, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.axvline(0.5, color=MUTED, lw=1); ax.set_xlim(0.5, s.max() + 0.04); ax.grid(axis="y", visible=False)
    ax.set_xlabel("single-feature ROC-AUC (0.5 = no signal)")
    ax.set_title("Signal per feature (EDA)")
    fig.text(0.01, 0.005, "Gray: the 8 features tested for removal (05d, 11a–c).", color=MUTED, fontsize=8)
    save(fig, "00_feature_signal.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = experiment_log()
    fig_signal(); fig_evolution(df); fig_cv_vs_lb(df); fig_optuna(); fig_diversity()
