import argparse, json
from area_predictor.predict import PredictionError, load_bundle, predict_records

ap = argparse.ArgumentParser(description="Predict area for one record (JSON file) or a list of records")
ap.add_argument("--file", required=True, help="JSON file containing an object or a list of objects")
a = ap.parse_args()
try:
    with open(a.file, encoding="utf-8") as f:
        data = json.load(f)
    res = predict_records(data if isinstance(data, list) else [data], load_bundle())
    print(json.dumps(res, indent=2))
except (PredictionError, FileNotFoundError, json.JSONDecodeError) as e:
    raise SystemExit(f"ERROR: {e}")
