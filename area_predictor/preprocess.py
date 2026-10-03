"""Preprocessing pipeline. All statistics are learned inside fit() on training data only."""
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

from .data import DATE_FEATURES


def build_preprocessor(cfg):
    num = cfg["raw_numeric_features"] + DATE_FEATURES
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, num), ("cat", categorical, cfg["categorical_features"])])


def make_pipeline(cfg, estimator, log_target=True):
    """Area is right-skewed and positive, so models are trained on log1p(area) and
    predictions are converted back to the original units."""
    model = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1) \
        if log_target else estimator
    return Pipeline([("prep", build_preprocessor(cfg)), ("model", model)])
