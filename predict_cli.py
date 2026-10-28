import argparse
import json
from area_predictor.predict import PredictionError, load_bundle, predict_records

ap = argparse.ArgumentParser(description="Predict area for one record or list of records")
ap.add_argument("--file", required=True, help="JSON file containing an object or a list of objects")
args = ap.parse_args()

try:
    with open(args.file, encoding="utf-8") as f:
        data = json.load(f)
    
    # Load model once
    bundle = load_bundle()
    
    # Support both single object and list
    records = data if isinstance(data, list) else [data]
    
    results = predict_records(records, bundle)
    
    print(json.dumps(results, indent=2))

except (PredictionError, FileNotFoundError, json.JSONDecodeError) as e:
    print(f"ERROR: {e}")