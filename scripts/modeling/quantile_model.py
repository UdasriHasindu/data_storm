"""
=========================================================================
Modeling — LightGBM Quantile Regression
=========================================================================
Estimates the 90th percentile of potential as an ML-based ceiling.
=========================================================================
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD_MODELING = os.path.join(PROJECT_ROOT, "data", "gold", "modeling")

MODEL_FEATURES = [
    "mean_vol", "median_vol", "p75_vol", "p90_vol", "cov",
    "flatline_score", "vol_trend", "holiday_uplift_ratio",
    "urban_density_proxy", "active_months", "cluster_id",
    "mean_sku_count", "Outlet_Size_Num", "avg_unit_price",
]

QUANTILE = 0.90


def fit_quantile_model(gold: pd.DataFrame = None) -> tuple:
    """Fit LightGBM quantile regression for potential ceiling."""
    print("\n── Modeling: Quantile Regression ──")

    if gold is None:
        gold = pd.read_parquet(os.path.join(GOLD_MODELING, "model_ready.parquet"))

    available = [c for c in MODEL_FEATURES if c in gold.columns]
    df = gold.copy().fillna(0)
    X = df[available]
    y = df["p95_vol"].fillna(df["max_vol"])

    params = {
        "objective": "quantile",
        "alpha": QUANTILE,
        "metric": "quantile",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "verbosity": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)

    gold["quantile_potential"] = model.predict(X).clip(min=0)

    # Feature importance
    importance = pd.DataFrame({
        "feature": available,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"  [QUANTILE] q{int(QUANTILE*100)} potential estimated. "
          f"Top features: {list(importance.head(5)['feature'])}")
    return gold, model


if __name__ == "__main__":
    gold, model = fit_quantile_model()
    gold.to_parquet(os.path.join(GOLD_MODELING, "model_ready.parquet"), index=False)
