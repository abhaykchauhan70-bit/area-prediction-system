"""Data loading, cleaning, feature derivation and splitting."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import resolve

DATE_FEATURES = ["year", "month", "dayofyear"]


def model_columns(cfg):
    return cfg["raw_numeric_features"] + DATE_FEATURES + cfg["categorical_features"]


def add_date_features(df, cfg):
    d = df[cfg["date_column"]]
    df["year"], df["month"], df["dayofyear"] = d.dt.year, d.dt.month, d.dt.dayofyear
    return df


def load_raw(cfg):
    """Load only the columns needed (leakage columns are never loaded)."""
    path = resolve(cfg, "data_path")
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run 'python generate_demo_data.py' for demo data "
            "or set 'data_path' in config.json to your CSV.")
    needed = set(cfg["raw_numeric_features"] + cfg["categorical_features"] +
                 [cfg["target"], cfg["date_column"]])
    if cfg["split_strategy"] == "group":
        needed.add(cfg["group_column"])
    needed -= set(cfg.get("leakage_columns", []))
    header = pd.read_csv(path, nrows=0).columns.str.strip()
    missing = needed - set(header)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = pd.read_csv(path, usecols=lambda c: c.strip() in needed, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def clean(df, cfg):
    """Clean the data; returns (clean_df, report). No learned statistics are used here."""
    rep = {"rows_loaded": len(df)}
    df = df.drop_duplicates().copy()
    rep["duplicates_removed"] = rep["rows_loaded"] - len(df)
    tgt, dcol = cfg["target"], cfg["date_column"]
    for c in cfg["raw_numeric_features"] + [tgt]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    for c in cfg["categorical_features"]:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA).astype(object)
        df[c] = df[c].where(df[c].notna(), np.nan)
    n = len(df)
    df = df[df[tgt].notna() & np.isfinite(df[tgt]) & (df[tgt] > 0) & df[dcol].notna()]
    rep["invalid_target_or_date_removed"] = n - len(df)
    n = len(df)
    for c in ("latitude", "longitude"):  # location is essential: drop invalid rows
        lo, hi = cfg["valid_ranges"][c]
        df = df[df[c].isna() | df[c].between(lo, hi)]
    rep["invalid_coordinates_removed"] = n - len(df)
    nulled = 0
    for c, (lo, hi) in cfg["valid_ranges"].items():  # impossible values -> missing (imputed later)
        bad = df[c].notna() & ~df[c].between(lo, hi)
        nulled += int(bad.sum())
        df.loc[bad, c] = np.nan
    rep["impossible_values_set_missing"] = nulled
    df = add_date_features(df.reset_index(drop=True), cfg)
    rep["rows_after_cleaning"] = len(df)
    rep["missing_per_feature"] = {c: int(v) for c, v in df[model_columns(cfg)].isna().sum().items() if v}
    return df, rep


def split_data(df, cfg):
    f = cfg["split_fractions"]
    seed, strat = cfg["random_seed"], cfg["split_strategy"]
    if strat == "time":
        df = df.sort_values(cfg["date_column"]).reset_index(drop=True)
        i1, i2 = int(len(df) * f[0]), int(len(df) * (f[0] + f[1]))
        return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    if strat == "group":
        g = df[cfg["group_column"]]
        tv, te = next(GroupShuffleSplit(1, test_size=f[2], random_state=seed).split(df, groups=g))
        tvdf = df.iloc[tv]
        tr, va = next(GroupShuffleSplit(1, test_size=f[1] / (f[0] + f[1]), random_state=seed)
                      .split(tvdf, groups=tvdf[cfg["group_column"]]))
        return tvdf.iloc[tr], tvdf.iloc[va], df.iloc[te]
    tv, te = train_test_split(df, test_size=f[2], random_state=seed)
    tr, va = train_test_split(tv, test_size=f[1] / (f[0] + f[1]), random_state=seed)
    return tr, va, te
