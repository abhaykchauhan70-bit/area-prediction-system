"""Local website: python app.py  ->  http://127.0.0.1:5000"""
from flask import Flask, jsonify, render_template, request
from area_predictor.predict import PredictionError, load_bundle, predict_records

app = Flask(__name__)
_bundle = None


def bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_bundle()
    return _bundle


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/schema")
def schema():
    try:
        b = bundle()
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503
    cfg = b["config"]
    fields = [{"name": cfg["date_column"], "type": "date", "required": True}]
    for f in cfg["raw_numeric_features"]:
        lo, hi = cfg["valid_ranges"].get(f, (None, None))
        fields.append({"name": f, "type": "number", "required": f in cfg["required_fields"], "min": lo, "max": hi})
    for c in cfg["categorical_features"]:
        fields.append({"name": c, "type": "select", "required": c in cfg["required_fields"], "options": b["categories"][c]})
    m = b["metrics"]
    return jsonify(fields=fields, units=cfg["units"], model=b["model_name"], demo=cfg.get("is_demo_data", False),
                   test_metrics=m["test_final"], baseline_metrics=m["test_baseline"])


@app.post("/api/predict")
def api_predict():
    rec = request.get_json(silent=True)
    if rec is None:
        return jsonify(error="Request body must be JSON."), 400
    try:
        return jsonify(predict_records([rec], bundle())[0])
    except PredictionError as e:
        return jsonify(error=str(e)), 422
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503


if __name__ == "__main__":
    app.run(debug=False, port=5000)
