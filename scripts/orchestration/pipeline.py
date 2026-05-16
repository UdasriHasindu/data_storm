"""
=========================================================================
Master Pipeline Orchestrator
=========================================================================
Data Storm v7.0 — Bronze → Silver → POI → Gold → Model → Output

Usage:
    python scripts/orchestration/pipeline.py
    python scripts/orchestration/pipeline.py --phase silver
    python scripts/orchestration/pipeline.py --phase poi --synthetic
=========================================================================
"""

import os
import sys
import argparse
import traceback
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

TEAM_NAME = "data_storm_team"  # ← CHANGE THIS


def run_bronze():
    from ingestion.bronze_ingest import ingest_bronze
    ingest_bronze()


def run_silver():
    from cleaning.silver_clean import run_silver
    run_silver()


def run_poi(synthetic=False):
    print("\n" + "=" * 60)
    print("  PHASE 3: POI ENRICHMENT")
    print("=" * 60)
    if synthetic:
        from enrichment.poi_scraper import generate_synthetic_poi
        generate_synthetic_poi()
    else:
        from enrichment.poi_scraper import scrape_all_outlets
        scrape_all_outlets()


def run_gold():
    from feature_engineering.gold_features import run_gold
    run_gold()


def run_model():
    from modeling.clustering import cluster_outlets
    from modeling.tobit_model import fit_tobit
    from modeling.quantile_model import fit_quantile_model
    from modeling.potential_ceiling import (
        detect_constrained_outlets,
        peer_benchmark_ceiling,
        compute_final_potential,
    )

    print("\n" + "=" * 60)
    print("  PHASE 5: Modeling — Latent Demand Estimation")
    print("=" * 60)

    gold = cluster_outlets()
    gold = detect_constrained_outlets(gold)
    gold = fit_tobit(gold)
    gold, _ = fit_quantile_model(gold)
    gold = peer_benchmark_ceiling(gold)
    gold = compute_final_potential(gold)

    model_path = os.path.join(PROJECT_ROOT, "data", "gold", "modeling", "model_ready.parquet")
    gold.to_parquet(model_path, index=False)
    return gold


def run_output(gold=None):
    print("\n" + "=" * 60)
    print("  PHASE 6: FINAL OUTPUT")
    print("=" * 60)

    if gold is None:
        gold = __import__("pandas").read_parquet(
            os.path.join(PROJECT_ROOT, "data", "gold", "modeling", "model_ready.parquet")
        )

    submission = gold[["Outlet_ID", "Maximum_Monthly_Liters"]].copy()
    submission.columns = ["Outlet_ID", "Maximum_Monthly_Liters"]

    # Validation
    assert submission["Outlet_ID"].nunique() == len(submission), \
        "ERROR: Duplicate Outlet_IDs!"
    assert submission["Maximum_Monthly_Liters"].isna().sum() == 0, \
        "ERROR: NaN predictions!"
    assert (submission["Maximum_Monthly_Liters"] >= 0).all(), \
        "ERROR: Negative predictions!"

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{TEAM_NAME}_predictions.csv")
    submission.to_csv(out_path, index=False)

    print(f"  [OUTPUT] ✓ Submission saved: {out_path}")
    print(f"           Outlets: {len(submission):,}")
    print(f"           Mean: {submission['Maximum_Monthly_Liters'].mean():.1f}L")
    print(f"           Median: {submission['Maximum_Monthly_Liters'].median():.1f}L")
    print(f"           Max: {submission['Maximum_Monthly_Liters'].max():.1f}L")
    return submission


def run_full_pipeline(synthetic_poi=False):
    start = datetime.now()
    print("\n" + "=" * 60)
    print("  DATA STORM v7.0 — FULL PIPELINE")
    print(f"  Started: {start.isoformat()}")
    print("=" * 60)

    try:
        run_bronze()
        run_silver()
        run_poi(synthetic=synthetic_poi)
        run_gold()
        gold = run_model()
        run_output(gold)
    except Exception as e:
        print(f"\n  [ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        return

    elapsed = (datetime.now() - start).total_seconds()
    print("\n" + "=" * 60)
    print(f"  ✓ PIPELINE COMPLETE ({elapsed:.1f}s)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Storm v7.0 Pipeline")
    parser.add_argument("--phase", default="all",
                        choices=["all", "bronze", "silver", "poi", "gold", "model", "output"])
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic POI data instead of scraping")
    args = parser.parse_args()

    if args.phase == "all":
        run_full_pipeline(synthetic_poi=args.synthetic)
    elif args.phase == "bronze":
        run_bronze()
    elif args.phase == "silver":
        run_silver()
    elif args.phase == "poi":
        run_poi(synthetic=args.synthetic)
    elif args.phase == "gold":
        run_gold()
    elif args.phase == "model":
        run_model()
    elif args.phase == "output":
        run_output()
