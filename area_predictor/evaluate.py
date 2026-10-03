"""Metrics, residual checks, plots and feature importance."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred)),
            "median_abs_error": float(np.median(np.abs(np.asarray(y) - np.asarray(pred))))}


def residual_summary(y, pred):
    r = np.asarray(y) - np.asarray(pred)
    return {"mean_residual (bias)": float(r.mean()), "std_residual": float(r.std()),
            "p05": float(np.percentile(r, 5)), "p95": float(np.percentile(r, 95))}


def save_plots(y, pred, out, unit):
    out.mkdir(parents=True, exist_ok=True)
    y, pred = np.asarray(y), np.asarray(pred)
    res = y - pred
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, pred, s=8, alpha=.5)
    m = max(y.max(), pred.max())
    ax.plot([0, m], [0, m], "r--")
    ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set(xlabel=f"Actual area ({unit})", ylabel=f"Predicted area ({unit})", title="Predicted vs actual (test)")
    fig.tight_layout(); fig.savefig(out / "pred_vs_actual.png", dpi=120); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(res, bins=40); axs[0].set(title="Residual distribution (test)", xlabel=f"Actual - predicted ({unit})")
    axs[1].scatter(pred, res, s=8, alpha=.5); axs[1].axhline(0, color="r")
    axs[1].set(title="Residuals vs predicted", xlabel=f"Predicted ({unit})", ylabel="Residual")
    axs[1].set_xscale("symlog")
    fig.tight_layout(); fig.savefig(out / "residuals.png", dpi=120); plt.close(fig)


def feature_importance(pipe, X, y, out, seed):
    r = permutation_importance(pipe, X, y, n_repeats=5, random_state=seed,
                               scoring="neg_root_mean_squared_error")
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    imp[::-1].plot.barh(ax=ax); ax.set(title="Permutation importance (validation)", xlabel="Increase in RMSE when shuffled")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=120); plt.close(fig)
    return {k: float(v) for k, v in imp.items()}


def write_report(path, cfg, rep, val_results, final_name, base_test, final_test, resid, imp, sizes):
    u = cfg["unit_symbol"]
    L = ["# Training report", ""]
    if cfg.get("is_demo_data"):
        L += ["> **DEMONSTRATION DATA (synthetic).** These numbers show the pipeline works; they say nothing about real-world accuracy.", ""]
    L += [f"Target: `{cfg['target']}` in {cfg['units']} ({u}). Split: {cfg['split_strategy']} "
          f"(train/val/test rows = {sizes}). Seed: {cfg['random_seed']}.", "", "## Data cleaning", "```", json.dumps(rep, indent=1), "```",
          "", "## Validation results (used for model selection)", "| model | best params | MAE | RMSE | R2 |", "|---|---|---|---|---|"]
    for n, (p, m) in val_results.items():
        L.append(f"| {n} | {p} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.3f} |")
    L += ["", f"Selected model: **{final_name}**", "", f"## Held-out test results (units: {u})", "| model | MAE | RMSE | R2 |", "|---|---|---|---|"]
    for n, m in (("baseline (median)", base_test), (final_name, final_test)):
        L.append(f"| {n} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.3f} |")
    L += ["", "MAE/RMSE are average errors in " + cfg["units"] + "; RMSE penalises large misses more. R2 is the share of variance explained versus predicting the mean.",
          "", "## Residuals (test)", "```", json.dumps(resid, indent=1), "```", "", "## Permutation importance (validation)"]
    L += [f"- {k}: {v:.3f}" for k, v in imp.items()]
    path.write_text("\n".join(L), encoding="utf-8")
