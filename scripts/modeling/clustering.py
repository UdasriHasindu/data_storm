"""
=========================================================================
Modeling — Outlet Clustering (Peer Grouping)
=========================================================================
K-Means clustering to group outlets with similar profiles.
Auto-selects optimal k via silhouette score.
=========================================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD_MODELING = os.path.join(PROJECT_ROOT, "data", "gold", "modeling")

CLUSTER_FEATURES = [
    "mean_vol", "p90_vol", "cov", "flatline_score", "vol_trend",
    "holiday_uplift_ratio", "urban_density_proxy", "active_months",
    "mean_sku_count", "Outlet_Size_Num",
]


def cluster_outlets(gold: pd.DataFrame = None, k_range=(6, 20)) -> pd.DataFrame:
    """Cluster outlets and auto-select best k."""
    print("\n── Clustering: Peer Groups ──")

    if gold is None:
        gold = pd.read_parquet(os.path.join(GOLD_MODELING, "model_ready.parquet"))

    available = [c for c in CLUSTER_FEATURES if c in gold.columns]
    X = gold[available].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, best_score = k_range[0], -1
    for k in range(k_range[0], k_range[1]):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_scaled)))
        if score > best_score:
            best_k, best_score = k, score

    print(f"  [CLUSTER] Best k={best_k}, silhouette={best_score:.3f}")
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    gold["cluster_id"] = km_final.fit_predict(X_scaled)

    gold.to_parquet(os.path.join(GOLD_MODELING, "model_ready.parquet"), index=False)
    print(f"  [CLUSTER] Assigned {best_k} clusters to {len(gold):,} outlets")
    return gold


if __name__ == "__main__":
    cluster_outlets()
