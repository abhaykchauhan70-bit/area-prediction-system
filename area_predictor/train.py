"""End-to-end training: load -> clean -> split -> baseline -> tune -> select -> test -> save."""
import json
import random

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid

from .config import load_config, resolve
from .data import clean, load_raw, model_columns, split_data
from .evaluate import feature_importance, metrics, residual_summary, save_plots, write_report
from .preprocess import make_pipeline


def candidates(seed):
    return {
        "ridge": (Ridge(), {"alpha": [0.1, 1, 10, 100]}),
        "random_forest": (RandomForestRegressor(random_state=seed, n_jobs=-1),
                          {"n_estimators": [200], "max_depth": [8, None], "min_samples_leaf": [2, 5]}),
        "hist_gradient_boosting": (HistGradientBoostingRegressor(random_state=seed),
                                   {"learning_rate": [0.05, 0.1], "max_iter": [200, 400], "max_leaf_nodes": [15, 31]}),
    }


def run(config_path=None):
    cfg = load_config(config_path)
    seed = cfg["random_seed"]
    random.seed(seed); np.random.seed(seed)
    out = resolve(cfg, "output_dir"); out.mkdir(parents=True, exist_ok=True)
    cols, tgt, unit = model_columns(cfg), cfg["target"], cfg["unit_symbol"]

    df, rep = clean(load_raw(cfg), cfg)
    if len(df) < 50:
        raise ValueError(f"Only {len(df)} usable rows after cleaning; need at least 50 to train.")
    tr, va, te = split_data(df, cfg)
    sizes = (len(tr), len(va), len(te))
    print("Cleaning report:", json.dumps(rep)); print("Split sizes:", sizes)
    Xtr, ytr, Xva, yva, Xte, yte = tr[cols], tr[tgt], va[cols], va[tgt], te[cols], te[tgt]

    base = make_pipeline(cfg, DummyRegressor(strategy="median"), log_target=False).fit(Xtr, ytr)
    base_test = metrics(yte, base.predict(Xte))

    val_results, best_pipes = {}, {}
    for name, (est, grid) in candidates(seed).items():
        best = None
        for params in ParameterGrid(grid):
            pipe = make_pipeline(cfg, clone(est).set_params(**params)).fit(Xtr, ytr)
            m = metrics(yva, pipe.predict(Xva))
            if best is None or m["RMSE"] < best[1]["RMSE"]:
                best = (params, m, pipe)
        val_results[name] = (best[0], best[1]); best_pipes[name] = best[2]
        print(f"{name}: val RMSE={best[1]['RMSE']:.2f} R2={best[1]['R2']:.3f} {best[0]}")

    final_name = min(val_results, key=lambda n: val_results[n][1]["RMSE"])
    final = best_pipes[final_name]
    pred_te = final.predict(Xte)
    final_test = metrics(yte, pred_te)
    ratio = (yva / np.maximum(final.predict(Xva), 1e-9)).to_numpy()
    q10, q90 = np.percentile(ratio, [10, 90])

    save_plots(yte, pred_te, out, unit)
    imp = feature_importance(final, Xva, yva, out, seed)
    resid = residual_summary(yte, pred_te)
    all_metrics = {"is_demo_data": cfg.get("is_demo_data", False), "units": cfg["units"], "selected_model": final_name,
                   "validation": {n: {"params": p, "metrics": m} for n, (p, m) in val_results.items()},
                   "test_baseline": base_test, "test_final": final_test, "test_residuals": resid,
                   "split_sizes": dict(zip(("train", "val", "test"), sizes)), "cleaning": rep}
    (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    write_report(out / "report.md", cfg, rep, val_results, final_name, base_test, final_test, resid, imp, sizes)

    categories = {c: sorted(df[c].dropna().unique().tolist()) for c in cfg["categorical_features"]}
    bundle = {"pipeline": final, "config": cfg, "categories": categories, "model_name": final_name,
              "range_ratio": [float(q10), float(q90)], "metrics": all_metrics}
    mp = resolve(cfg, "model_path"); mp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, mp)
    print(f"\nSelected {final_name}. TEST: baseline RMSE={base_test['RMSE']:.2f}, model RMSE={final_test['RMSE']:.2f}, "
          f"MAE={final_test['MAE']:.2f}, R2={final_test['R2']:.3f}\nSaved model -> {mp}\nSaved outputs -> {out}")
    return all_metrics
