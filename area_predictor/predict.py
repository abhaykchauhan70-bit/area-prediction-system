"""Load a saved model and predict on new, validated records."""
import joblib
import numpy as np
import pandas as pd

from .config import load_config, resolve
from .data import add_date_features, model_columns


class PredictionError(ValueError):
    """Raised for missing or invalid input."""


def load_bundle(config_path=None):
    cfg = load_config(config_path)
    path = resolve(cfg, "model_path")
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run 'python train.py' first.")
    return joblib.load(path)


def validate_record(rec, cfg, categories):
    if not isinstance(rec, dict):
        raise PredictionError("Each record must be a JSON object of field: value pairs.")
    errors, warnings, clean = [], [], {}
    blank = lambda v: v is None or str(v).strip() == ""
    for f in cfg["required_fields"]:
        if blank(rec.get(f)):
            errors.append(f"Missing required field '{f}'")
    for f in cfg["raw_numeric_features"]:
        v = rec.get(f)
        if blank(v):
            clean[f] = np.nan
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            errors.append(f"'{f}' must be a number (got {v!r})"); continue
        lo, hi = cfg["valid_ranges"].get(f, (-np.inf, np.inf))
        if not np.isfinite(x) or not lo <= x <= hi:
            errors.append(f"'{f}' must be between {lo} and {hi} (got {v})"); continue
        clean[f] = x
    dcol = cfg["date_column"]
    if not blank(rec.get(dcol)):
        try:
            clean[dcol] = pd.to_datetime(str(rec[dcol]), errors="raise")
        except (ValueError, TypeError):
            errors.append(f"'{dcol}' must be a valid date such as 2024-06-30")
    for c in cfg["categorical_features"]:
        v = rec.get(c)
        if blank(v):
            clean[c] = np.nan
        else:
            clean[c] = str(v).strip()
            if clean[c] not in categories.get(c, []):
                warnings.append(f"'{c}' value '{clean[c]}' was not seen in training; the estimate may be less reliable")
    if errors:
        raise PredictionError("; ".join(errors))
    return clean, warnings


def predict_records(records, bundle):
    cfg = bundle["config"]
    cleaned, warns = [], []
    for i, r in enumerate(records):
        try:
            c, w = validate_record(r, cfg, bundle["categories"])
        except PredictionError as e:
            raise PredictionError(f"Record {i}: {e}" if len(records) > 1 else str(e))
        cleaned.append(c); warns.append(w)
    df = add_date_features(pd.DataFrame(cleaned), cfg)
    preds = bundle["pipeline"].predict(df[model_columns(cfg)])
    q10, q90 = bundle["range_ratio"]
    return [{"predicted_area": round(float(p), 2), "units": cfg["units"], "unit_symbol": cfg["unit_symbol"],
             "approx_80pct_range": [round(float(p) * q10, 2), round(float(p) * q90, 2)],
             "warnings": w,
             "note": "Estimate only. Range is based on validation-set errors and is not a guarantee."}
            for p, w in zip(preds, warns)]
