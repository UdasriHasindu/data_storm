"""
=========================================================================
Modeling — Tobit Regression (Censored Demand Model)
=========================================================================
Implements a Tobit model from scratch using scipy.optimize.
Models observed sales as left-censored: y = max(y*, threshold)
where y* is the true latent demand.
=========================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD_MODELING = os.path.join(PROJECT_ROOT, "data", "gold", "modeling")

MODEL_FEATURES = [
    "mean_vol", "p90_vol", "cov", "flatline_score", "vol_trend",
    "holiday_uplift_ratio", "urban_density_proxy", "active_months",
    "jan_historical_avg", "mean_sku_count", "Outlet_Size_Num",
    "cluster_id",
]


def tobit_log_likelihood(params, X, y, censor_threshold):
    """Tobit log-likelihood for left-censoring."""
    beta = params[:-1]
    sigma = np.exp(params[-1])

    mu = X @ beta
    residual = (y - mu) / sigma

    uncensored = y > censor_threshold
    ll = np.where(
        uncensored,
        norm.logpdf(residual) - np.log(sigma),
        norm.logcdf((censor_threshold - mu) / sigma)
    )
    return -ll.sum()


def fit_tobit(gold: pd.DataFrame = None) -> pd.DataFrame:
    """Fit Tobit regression and predict uncensored latent demand."""
    print("\n── Modeling: Tobit Regression ──")

    if gold is None:
        gold = pd.read_parquet(os.path.join(GOLD_MODELING, "model_ready.parquet"))

    available = [c for c in MODEL_FEATURES if c in gold.columns]
    df = gold[available + ["p95_vol"]].dropna().copy()
    y = df["p95_vol"].values

    censor_thresh = np.percentile(y, 30)

    X = df[available].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.column_stack([X_scaled, np.ones(len(X_scaled))])

    init_params = np.zeros(X_scaled.shape[1] + 1)

    result = minimize(
        tobit_log_likelihood,
        x0=init_params,
        args=(X_scaled, y, censor_thresh),
        method="L-BFGS-B",
        options={"maxiter": 1000}
    )

    beta_hat = result.x[:-1]
    sigma_hat = np.exp(result.x[-1])

    mu_pred = X_scaled @ beta_hat

    # Map predictions back to full gold dataset
    gold.loc[df.index, "tobit_potential"] = mu_pred

    # Inverse Mills ratio for censored correction
    z = (censor_thresh - mu_pred) / sigma_hat
    phi = norm.pdf(z)
    Phi = norm.cdf(z).clip(1e-6)
    gold.loc[df.index, "tobit_uncensored"] = mu_pred + sigma_hat * (phi / Phi)

    print(f"  [TOBIT] Model fitted. sigma={sigma_hat:.2f}, "
          f"converged={result.success}")
    return gold


if __name__ == "__main__":
    gold = fit_tobit()
    gold.to_parquet(os.path.join(GOLD_MODELING, "model_ready.parquet"), index=False)
