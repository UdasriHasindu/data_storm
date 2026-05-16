"""
=========================================================================
Gold Layer — Feature Engineering
=========================================================================
Builds model-ready features from Silver datasets + POI enrichment.
All features target January 2026 potential estimation.
=========================================================================
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SILVER_CLEAN = os.path.join(PROJECT_ROOT, "data", "silver", "cleaned")
GOLD_DIR = os.path.join(PROJECT_ROOT, "data", "gold")
GOLD_FEATURES = os.path.join(GOLD_DIR, "features")
GOLD_MODELING = os.path.join(GOLD_DIR, "modeling")

for d in [GOLD_FEATURES, GOLD_MODELING]:
    os.makedirs(d, exist_ok=True)

TARGET_MONTH = 1
TARGET_YEAR = 2026


def build_sales_features():
    """Compute historical volume statistics per outlet."""
    print("\n── Building: Sales Features ──")
    txn = pd.read_parquet(os.path.join(SILVER_CLEAN, "transactions_clean.parquet"))

    # Aggregate to outlet-month level (sum across SKUs)
    monthly = (
        txn.groupby(["Outlet_ID", "Year", "Month"])
        .agg(
            monthly_volume=("Volume_Liters", "sum"),
            monthly_bill=("Total_Bill_Value", "sum"),
            sku_count=("SKU_ID", "nunique"),
            transaction_count=("SKU_ID", "count"),
            ghost_count=("ghost_flag", "sum"),
        )
        .reset_index()
    )

    # Per-outlet aggregate features
    feats = monthly.groupby("Outlet_ID").agg(
        mean_vol=("monthly_volume", "mean"),
        median_vol=("monthly_volume", "median"),
        std_vol=("monthly_volume", "std"),
        p75_vol=("monthly_volume", lambda x: x.quantile(0.75)),
        p90_vol=("monthly_volume", lambda x: x.quantile(0.90)),
        p95_vol=("monthly_volume", lambda x: x.quantile(0.95)),
        max_vol=("monthly_volume", "max"),
        min_vol=("monthly_volume", "min"),
        active_months=("monthly_volume", "count"),
        mean_bill=("monthly_bill", "mean"),
        mean_sku_count=("sku_count", "mean"),
        total_ghost_count=("ghost_count", "sum"),
    ).reset_index()

    feats["cov"] = feats["std_vol"] / (feats["mean_vol"] + 1e-6)
    feats["vol_range"] = feats["max_vol"] - feats["min_vol"]
    feats["ghost_rate"] = feats["total_ghost_count"] / (feats["active_months"] + 1e-6)

    # ── Constraint Detection Features ──
    # 1. Flatline score: fraction of months near the mode
    def flatline_score(series):
        if len(series) < 3:
            return 0.0
        mode_val = series.mode().iloc[0] if not series.mode().empty else series.mean()
        near_mode = ((series - mode_val).abs() / (mode_val + 1e-6)) < 0.05
        return near_mode.mean()

    flatlines = monthly.groupby("Outlet_ID")["monthly_volume"].apply(flatline_score)
    feats = feats.merge(flatlines.rename("flatline_score").reset_index(), on="Outlet_ID", how="left")

    # 2. Trend: slope of monthly volume over time
    def volume_trend(group):
        if len(group) < 3:
            return 0.0
        x = np.arange(len(group))
        coeffs = np.polyfit(x, group["monthly_volume"].values, 1)
        return float(coeffs[0])

    trends = monthly.sort_values(["Outlet_ID", "Year", "Month"]).groupby("Outlet_ID").apply(volume_trend)
    feats = feats.merge(trends.rename("vol_trend").reset_index(), on="Outlet_ID", how="left")

    # 3. Holiday uplift ratio
    holidays = pd.read_parquet(os.path.join(SILVER_CLEAN, "holidays_clean.parquet"))
    holiday_months_set = set(zip(holidays["Year"], holidays["Month"]))
    monthly_h = monthly.copy()
    monthly_h["is_holiday_month"] = monthly_h.apply(
        lambda r: (r["Year"], r["Month"]) in holiday_months_set, axis=1
    )

    # Compute average volume for holiday vs non-holiday months per outlet
    hol_avg = monthly_h[monthly_h["is_holiday_month"]].groupby("Outlet_ID")["monthly_volume"].mean()
    non_hol_avg = monthly_h[~monthly_h["is_holiday_month"]].groupby("Outlet_ID")["monthly_volume"].mean()
    holiday_uplift = (hol_avg / (non_hol_avg + 1e-6)).rename("holiday_uplift_ratio").reset_index()
    feats = feats.merge(holiday_uplift, on="Outlet_ID", how="left")
    feats["holiday_uplift_ratio"] = feats["holiday_uplift_ratio"].fillna(1.0)

    # 4. January historical average
    jan_data = monthly[monthly["Month"] == TARGET_MONTH]
    jan_avg = jan_data.groupby("Outlet_ID")["monthly_volume"].mean().rename("jan_historical_avg")
    feats = feats.merge(jan_avg.reset_index(), on="Outlet_ID", how="left")

    # 5. Recency: months since last transaction
    latest = monthly.groupby("Outlet_ID").apply(
        lambda g: (TARGET_YEAR - g["Year"].max()) * 12 + (TARGET_MONTH - g["Month"].max())
    ).rename("months_since_last")
    feats = feats.merge(latest.reset_index(), on="Outlet_ID", how="left")

    # 6. Unit price proxy
    feats["avg_unit_price"] = feats["mean_bill"] / (feats["mean_vol"] + 1e-6)

    feats.to_parquet(os.path.join(GOLD_FEATURES, "sales_features.parquet"), index=False)
    print(f"  [GOLD] Sales features: {feats.shape[1]} features × {len(feats):,} outlets")
    return feats


def build_outlet_features():
    """Encode outlet type, size, cooler count, distributor."""
    print("\n── Building: Outlet Features ──")
    outlets = pd.read_parquet(os.path.join(SILVER_CLEAN, "outlets_clean.parquet"))
    geo = pd.read_parquet(os.path.join(SILVER_CLEAN, "geo_clean.parquet"))

    # Merge outlet master with geo
    df = outlets.merge(geo, on="Outlet_ID", how="left")

    # Map distributor from transactions (outlet_master doesn't have it)
    txn = pd.read_parquet(os.path.join(SILVER_CLEAN, "transactions_clean.parquet"))
    outlet_dist = txn.groupby("Outlet_ID")["Distributor_ID"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    ).rename("Distributor_ID").reset_index()
    df = df.merge(outlet_dist, on="Outlet_ID", how="left")

    # Map province from distributor ID
    dist_province = {
        "DIST_W_01": "Western", "DIST_W_02": "Western", "DIST_W_03": "Western",
        "DIST_C_01": "Central", "DIST_C_02": "Central", "DIST_C_03": "Central",
        "DIST_NW_01": "North-Western", "DIST_NW_02": "North-Western",
        "DIST_S_01": "Southern", "DIST_S_02": "Southern",
    }
    df["Province"] = df["Distributor_ID"].map(dist_province)

    # Outlet_Size ordinal encoding
    size_map = {"Small": 1, "Medium": 2, "Large": 3, "Extra Large": 4}
    df["Outlet_Size_Num"] = df["Outlet_Size"].map(size_map).fillna(1)

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=["Outlet_Type", "Province"], drop_first=False, dtype=int)

    # Urban density proxy
    df["urban_density_proxy"] = df.get("coord_duplicate_flag", pd.Series(0, index=df.index)).fillna(0)

    df.to_parquet(os.path.join(GOLD_FEATURES, "outlet_features.parquet"), index=False)
    print(f"  [GOLD] Outlet features: {df.shape[1]} features × {len(df):,} outlets")
    return df


def build_seasonality_feature():
    """Extract January seasonality index per distributor."""
    print("\n── Building: Seasonality Features ──")
    season = pd.read_parquet(os.path.join(SILVER_CLEAN, "seasonality_clean.parquet"))
    # Get most recent January seasonality per distributor
    jan_season = season[season["Month"] == TARGET_MONTH].sort_values("Year", ascending=False)
    jan_season = jan_season.drop_duplicates("Distributor_ID", keep="first")
    result = jan_season[["Distributor_ID", "Seasonality_Numeric"]].rename(
        columns={"Seasonality_Numeric": "seasonality_index"}
    )
    print(f"  [GOLD] Seasonality: {len(result)} distributors")
    return result


def assemble_gold_dataset():
    """Join all feature tables into one model-ready dataset."""
    print("\n── Assembling: Gold Model-Ready Dataset ──")

    sales = pd.read_parquet(os.path.join(GOLD_FEATURES, "sales_features.parquet"))
    outlet = pd.read_parquet(os.path.join(GOLD_FEATURES, "outlet_features.parquet"))

    poi_path = os.path.join(GOLD_DIR, "poi", "poi_features.parquet")
    if os.path.exists(poi_path):
        poi = pd.read_parquet(poi_path)
    else:
        print("  [GOLD] ⚠ POI features not found — skipping POI merge")
        poi = pd.DataFrame({"Outlet_ID": sales["Outlet_ID"]})

    jan_idx = build_seasonality_feature()

    gold = (
        sales
        .merge(outlet, on="Outlet_ID", how="left")
        .merge(poi, on="Outlet_ID", how="left")
        .merge(
            outlet[["Outlet_ID", "Distributor_ID"]].drop_duplicates(),
            on="Outlet_ID", how="left", suffixes=("", "_dup")
        )
        .merge(jan_idx, on="Distributor_ID", how="left")
    )

    # Drop duplicate columns
    dup_cols = [c for c in gold.columns if c.endswith("_dup")]
    gold = gold.drop(columns=dup_cols, errors="ignore")

    # Fill POI NaN with median
    poi_cols = [c for c in gold.columns if "poi" in c.lower() or "catchment" in c.lower()]
    for col in poi_cols:
        gold[col] = pd.to_numeric(gold[col], errors="coerce")
    gold[poi_cols] = gold[poi_cols].fillna(gold[poi_cols].median())

    out_path = os.path.join(GOLD_MODELING, "model_ready.parquet")
    gold.to_parquet(out_path, index=False)
    print(f"  [GOLD] Model-ready: {gold.shape[1]} features × {len(gold):,} outlets → {out_path}")
    return gold


def run_gold():
    print("\n" + "=" * 60)
    print("  PHASE 4: GOLD LAYER — Feature Engineering")
    print("=" * 60)
    build_sales_features()
    build_outlet_features()
    assemble_gold_dataset()
    print("\n  [GOLD] ✓ All feature tables built.")


if __name__ == "__main__":
    run_gold()
