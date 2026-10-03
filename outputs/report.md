# Training report

> **DEMONSTRATION DATA (synthetic).** These numbers show the pipeline works; they say nothing about real-world accuracy.

Target: `area_ha` in hectares (ha). Split: time (train/val/test rows = (2093, 448, 449)). Seed: 42.

## Data cleaning
```
{
 "rows_loaded": 3030,
 "duplicates_removed": 30,
 "invalid_target_or_date_removed": 5,
 "invalid_coordinates_removed": 5,
 "impossible_values_set_missing": 0,
 "rows_after_cleaning": 2990,
 "missing_per_feature": {
  "elevation_m": 60,
  "rainfall_mm": 61,
  "temperature_c": 60
 }
}
```

## Validation results (used for model selection)
| model | best params | MAE | RMSE | R2 |
|---|---|---|---|---|
| ridge | {'alpha': 0.1} | 12.35 | 20.55 | 0.591 |
| random_forest | {'max_depth': None, 'min_samples_leaf': 2, 'n_estimators': 200} | 12.20 | 19.94 | 0.615 |
| hist_gradient_boosting | {'learning_rate': 0.1, 'max_iter': 200, 'max_leaf_nodes': 15} | 10.73 | 18.02 | 0.686 |

Selected model: **hist_gradient_boosting**

## Held-out test results (units: ha)
| model | MAE | RMSE | R2 |
|---|---|---|---|
| baseline (median) | 23.62 | 39.55 | -0.115 |
| hist_gradient_boosting | 12.72 | 21.91 | 0.658 |

MAE/RMSE are average errors in hectares; RMSE penalises large misses more. R2 is the share of variance explained versus predicting the mean.

## Residuals (test)
```
{
 "mean_residual (bias)": 2.768516739147497,
 "std_residual": 21.733034879330166,
 "p05": -24.563044683518555,
 "p95": 40.656141239820414
}
```

## Permutation importance (validation)
- prior_area_ha: 13.098
- land_cover: 7.801
- temperature_c: 6.778
- rainfall_mm: 4.412
- region: 2.991
- population_density: 2.771
- longitude: 0.193
- elevation_m: 0.054
- latitude: 0.033
- year: 0.000
- month: -0.014
- dayofyear: -0.028