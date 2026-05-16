"""
=========================================================================
Modeling — Constraint Detection & Peer Benchmarking & Final Ensemble
=========================================================================
Detects constrained outlets, computes peer ceilings, and produces
the final Maximum_Monthly_Liters prediction via ensemble.
=========================================================================
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD_MODELING = os.path.join(PROJECT_ROOT, "data", "gold", "modeling")

CONSTRAINT_THRESHOLD = 0.60


def detect_constrained_outlets(gold: pd.DataFrame) -> pd.DataFrame:
    """Tag outlets showing signs of artificial constraint."""
    print("\n── Constraint Detection ──")
    gold["is_constrained"] = (
        (gold.get("flatline_score", 0) > CONSTRAINT_THRESHOLD) |
        (gold.get("ghost_rate", 0) > 0.30)
    ).astype(int)

    pct = gold["is_constrained"].mean() * 100
    print(f"  [CONSTRAINT] {pct:.1f}% of outlets flagged as potentially constrained")
    return gold


def peer_benchmark_ceiling(gold: pd.DataFrame) -> pd.DataFrame:
    """Use cluster p90 of top performers as the peer ceiling."""
    print("\n── Peer Benchmarking ──")
    cluster_ceilings = (
        gold.groupby("cluster_id")["p95_vol"]
        .quantile(0.90)
        .rename("cluster_p90_ceiling")
        .reset_index()
    )
    gold = gold.merge(cluster_ceilings, on="cluster_id", how="left")

    gold["peer_gap"] = (gold["cluster_p90_ceiling"] - gold["p95_vol"]).clip(lower=0)

    gold["peer_potential"] = np.where(
        gold["is_constrained"] == 1,
        gold["cluster_p90_ceiling"],
        gold[["p95_vol", "cluster_p90_ceiling"]].max(axis=1) * 0.85
    )
    print(f"  [PEER] Peer ceilings computed for {gold['cluster_id'].nunique()} clusters")
    return gold


def compute_final_potential(gold: pd.DataFrame) -> pd.DataFrame:
    """Ensemble: Tobit + Peer + Quantile, adjusted for seasonality."""
    print("\n── Final Potential Ensemble ──")

    WEIGHTS = {"tobit": 0.40, "peer": 0.35, "quantile": 0.25}

    # Fallbacks
    gold["tobit_uncensored"] = gold.get("tobit_uncensored", gold["p95_vol"]).fillna(gold["p95_vol"])
    gold["quantile_potential"] = gold.get("quantile_potential", gold["p95_vol"]).fillna(gold["p95_vol"])
    gold["peer_potential"] = gold.get("peer_potential", gold["p95_vol"]).fillna(gold["p95_vol"])

    gold["raw_potential"] = (
        WEIGHTS["tobit"]    * gold["tobit_uncensored"] +
        WEIGHTS["peer"]     * gold["peer_potential"] +
        WEIGHTS["quantile"] * gold["quantile_potential"]
    )

    # Apply January seasonality
    gold["seasonality_index"] = gold.get("seasonality_index", pd.Series(1.0, index=gold.index)).fillna(1.0)
    gold["Maximum_Monthly_Liters"] = (
        gold["raw_potential"] * gold["seasonality_index"]
    ).round(2).clip(lower=0)

    # Sanity floor: must be >= outlet's observed mean
    gold["Maximum_Monthly_Liters"] = gold[["Maximum_Monthly_Liters", "mean_vol"]].max(axis=1)

    print(f"  [FINAL] Potential computed:")
    print(f"          Median: {gold['Maximum_Monthly_Liters'].median():.1f}L")
    print(f"          Mean:   {gold['Maximum_Monthly_Liters'].mean():.1f}L")
    print(f"          Max:    {gold['Maximum_Monthly_Liters'].max():.1f}L")
    return gold


if __name__ == "__main__":
    gold = pd.read_parquet(os.path.join(GOLD_MODELING, "model_ready.parquet"))
    gold = detect_constrained_outlets(gold)
    gold = peer_benchmark_ceiling(gold)
    gold = compute_final_potential(gold)
    gold.to_parquet(os.path.join(GOLD_MODELING, "model_ready.parquet"), index=False)
