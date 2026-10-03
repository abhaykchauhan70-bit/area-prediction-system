# Area Prediction System

Predicts the size of a land area / affected region (`area_ha`, units configurable) from tabular + geospatial features, with a local website for predictions.

> The bundled dataset is **synthetic demonstration data**. A model trained on it proves the pipeline works, not that it is accurate on real data. Replace it with your own data before relying on any prediction.

## Setup (VS Code)
1. Open this folder in VS Code (`File > Open Folder`).
2. Terminal: `python -m venv .venv`, then activate it (Windows: `.venv\Scripts\activate`, macOS/Linux: `source .venv/bin/activate`).
3. `pip install -r requirements.txt`
4. Select the `.venv` interpreter (`Ctrl+Shift+P` > *Python: Select Interpreter*).

## Run
```
python generate_demo_data.py   # only for demo data
python train.py                # trains, evaluates, saves models/ and outputs/
python app.py                  # website at http://127.0.0.1:5000
python predict_cli.py --file sample_input.json
```
`.vscode/launch.json` provides Run/Debug entries for training and the website.

## Structure
| Path | Purpose |
|---|---|
| `config.json` | data path, target, units, seed, split strategy, features, valid ranges, leakage columns |
| `area_predictor/data.py` | loading (only needed columns), cleaning, date features, splitting |
| `area_predictor/preprocess.py` | imputation, scaling, one-hot encoding, log-target (fit on train only) |
| `area_predictor/train.py` | baseline, tuning on validation, model selection, final test, saving |
| `area_predictor/evaluate.py` | MAE/RMSE/R², residual checks, plots, permutation importance |
| `area_predictor/predict.py` | model loading, input validation, predictions |
| `app.py`, `templates/index.html` | website and JSON API (`/api/schema`, `/api/predict`) |
| `outputs/` | `metrics.json`, `report.md`, PNG plots |

## Using your own data
Provide a CSV and edit `config.json`: `data_path`, `target`, `units`/`unit_symbol`, `date_column`, `raw_numeric_features`, `categorical_features`, `required_fields`, `valid_ranges`, and set `"is_demo_data": false`.
- Put columns only known after the outcome (e.g. post-event measurements) in `leakage_columns`; they are never loaded.
- `split_strategy`: `time` (chronological; default), `group` (needs `group_column`, e.g. event/location id), or `random`.
- Required target: positive numeric area in the stated units. Rows with missing/non-positive target or invalid date/coordinates are dropped; out-of-range feature values become missing and are imputed.

## Method
Baseline = median area. Candidates = Ridge, Random Forest, HistGradientBoosting (small grids). Models train on log1p(area) and predict back in original units. Hyperparameters and model choice use the validation set; the test set is used once for the final report. Final model is trained on the training split only. The "80% range" shown is the 10th–90th percentile of actual/predicted ratios on validation data.

## Limitations
- Estimates only; the range is empirical, not a guaranteed interval.
- Chronological splits mean performance can degrade if conditions drift beyond the training period.
- Unseen categories and extrapolation outside the training ranges are less reliable.
- Results in `outputs/report.md` come from synthetic data unless you replaced it.
