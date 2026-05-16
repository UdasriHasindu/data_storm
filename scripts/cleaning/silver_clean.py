"""
=========================================================================
Silver Layer — Data Forensics & Cleaning
=========================================================================
Applies reusable DQ checks to all bronze datasets.
Rejected records quarantined with documented failure reasons.
=========================================================================
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "validation"))

from dq_checks import (
    check_duplicates, check_nulls, check_value_range,
    check_format, check_referential_integrity, check_statistical_outlier,
    check_ghost_entries, check_cross_field_consistency,
    quarantine, write_dq_summary
)

BRONZE_DIR = os.path.join(PROJECT_ROOT, "data", "bronze")
SILVER_CLEAN = os.path.join(PROJECT_ROOT, "data", "silver", "cleaned")
SILVER_REJECTED = os.path.join(PROJECT_ROOT, "data", "silver", "rejected")
SILVER_VALIDATED = os.path.join(PROJECT_ROOT, "data", "silver", "validated")
SILVER_LOGS = os.path.join(PROJECT_ROOT, "data", "silver", "logs")

VALID_DISTRIBUTORS = [
    "DIST_W_01", "DIST_W_02", "DIST_W_03",
    "DIST_C_01", "DIST_C_02", "DIST_C_03",
    "DIST_NW_01", "DIST_NW_02",
    "DIST_S_01", "DIST_S_02"
]

for d in [SILVER_CLEAN, SILVER_REJECTED, SILVER_VALIDATED, SILVER_LOGS]:
    os.makedirs(d, exist_ok=True)


def _safe_concat_rejected(rejected_list):
    """Safely concatenate rejected DataFrames, handling empty case."""
    non_empty = [r for r in rejected_list if not r.empty]
    if non_empty:
        return pd.concat(non_empty, ignore_index=True)
    return pd.DataFrame()


def clean_transactions():
    """Clean transactions_history_final.csv"""
    print("\n── Cleaning: Transactions ──")
    df = pd.read_csv(os.path.join(BRONZE_DIR, "transactions_history_final.csv"))
    original_count = len(df)
    rejected = []
    checks_log = []

    # DQ1: Null mandatory fields
    mandatory = ["Outlet_ID", "Year", "Month", "Distributor_ID", "SKU_ID", "Volume_Liters"]
    null_mask = check_nulls(df, mandatory_cols=mandatory)
    rejected.append(quarantine(df[null_mask], reason="null_mandatory_field", check_name="null_check"))
    checks_log.append({"check": "null_check", "params": {"cols": mandatory}, "flagged_count": int(null_mask.sum())})
    df = df[~null_mask]

    # DQ2: Duplicates on composite key
    dupes = check_duplicates(df, key_cols=["Outlet_ID", "Year", "Month", "Distributor_ID", "SKU_ID"])
    rejected.append(quarantine(df[dupes], reason="duplicate_transaction", check_name="duplicate_check"))
    checks_log.append({"check": "duplicate_check", "params": {"key": "composite"}, "flagged_count": int(dupes.sum())})
    df = df[~dupes]

    # DQ3: Non-positive volume
    neg_mask = check_value_range(df, col="Volume_Liters", min_val=0.001)
    rejected.append(quarantine(df[neg_mask], reason="non_positive_volume", check_name="value_range_volume"))
    checks_log.append({"check": "value_range_volume_min", "params": {"min": 0.001}, "flagged_count": int(neg_mask.sum())})
    df = df[~neg_mask]

    # DQ4: Extreme volume outliers (>99.9th percentile × 3)
    upper_bound = df["Volume_Liters"].quantile(0.999) * 3
    extreme_mask = check_value_range(df, col="Volume_Liters", max_val=upper_bound)
    rejected.append(quarantine(df[extreme_mask], reason="extreme_volume_outlier", check_name="extreme_outlier"))
    checks_log.append({"check": "extreme_outlier", "params": {"max": float(upper_bound)}, "flagged_count": int(extreme_mask.sum())})
    df = df[~extreme_mask]

    # DQ5: Distributor ID referential integrity
    bad_dist = check_referential_integrity(df, fk_col="Distributor_ID", ref_values=VALID_DISTRIBUTORS)
    rejected.append(quarantine(df[bad_dist], reason="invalid_distributor_id", check_name="ref_integrity_dist"))
    checks_log.append({"check": "ref_integrity_distributor", "params": {"valid_count": len(VALID_DISTRIBUTORS)}, "flagged_count": int(bad_dist.sum())})
    df = df[~bad_dist]

    # DQ6: Outlet ID format
    bad_outlet = check_format(df, col="Outlet_ID", fmt="outlet_id")
    rejected.append(quarantine(df[bad_outlet], reason="invalid_outlet_id_format", check_name="format_outlet"))
    checks_log.append({"check": "format_outlet_id", "params": {"fmt": "OUT_XXXXX"}, "flagged_count": int(bad_outlet.sum())})
    df = df[~bad_outlet]

    # DQ7: Year/Month validity
    bad_year = check_value_range(df, col="Year", min_val=2020, max_val=2026)
    rejected.append(quarantine(df[bad_year], reason="invalid_year", check_name="range_year"))
    checks_log.append({"check": "range_year", "params": {"min": 2020, "max": 2026}, "flagged_count": int(bad_year.sum())})
    df = df[~bad_year]

    bad_month = check_value_range(df, col="Month", min_val=1, max_val=12)
    rejected.append(quarantine(df[bad_month], reason="invalid_month", check_name="range_month"))
    checks_log.append({"check": "range_month", "params": {"min": 1, "max": 12}, "flagged_count": int(bad_month.sum())})
    df = df[~bad_month]

    # DQ8: Cross-field — Total_Bill_Value should be positive when Volume > 0
    if "Total_Bill_Value" in df.columns:
        bill_check = check_cross_field_consistency(
            df, condition_expr="Total_Bill_Value > 0",
            description="bill_value_positive"
        )
        rejected.append(quarantine(df[bill_check], reason="non_positive_bill_value", check_name="cross_field_bill"))
        checks_log.append({"check": "cross_field_bill", "flagged_count": int(bill_check.sum())})
        df = df[~bill_check]

    # DQ9: Ghost entries — flag but keep (used as feature downstream)
    df_sorted = df.sort_values(["Outlet_ID", "Year", "Month", "SKU_ID"])
    ghost_mask = check_ghost_entries(
        df_sorted, group_col="Outlet_ID", value_col="Volume_Liters",
        sort_cols=["Outlet_ID", "Year", "Month"], consecutive_threshold=3
    )
    df_sorted["ghost_flag"] = ghost_mask.astype(int)
    checks_log.append({"check": "ghost_entry_flag", "flagged_count": int(ghost_mask.sum())})

    # Write outputs
    df_sorted.to_parquet(os.path.join(SILVER_CLEAN, "transactions_clean.parquet"), index=False)
    all_rejected = _safe_concat_rejected(rejected)
    if not all_rejected.empty:
        all_rejected.to_csv(os.path.join(SILVER_REJECTED, "transactions_rejected.csv"), index=False)

    write_dq_summary("transactions", original_count, len(df_sorted),
                      len(all_rejected), checks_log, SILVER_LOGS)
    print(f"  [SILVER] Transactions: {len(df_sorted):,} clean | {len(all_rejected):,} rejected")
    return df_sorted


def clean_outlets():
    """Clean outlet_master.csv — fix typos, standardize categories."""
    print("\n── Cleaning: Outlet Master ──")
    df = pd.read_csv(os.path.join(BRONZE_DIR, "outlet_master.csv"))
    original_count = len(df)
    rejected = []
    checks_log = []

    # DQ1: Null core fields
    mandatory = ["Outlet_ID", "Outlet_Type"]
    null_mask = check_nulls(df, mandatory_cols=mandatory)
    rejected.append(quarantine(df[null_mask], reason="null_outlet_core_field", check_name="null_check"))
    checks_log.append({"check": "null_check", "params": {"cols": mandatory}, "flagged_count": int(null_mask.sum())})
    df = df[~null_mask]

    # DQ2: Duplicate outlet IDs
    dupes = check_duplicates(df, key_cols=["Outlet_ID"])
    rejected.append(quarantine(df[dupes], reason="duplicate_outlet_id", check_name="duplicate_check"))
    checks_log.append({"check": "duplicate_check", "flagged_count": int(dupes.sum())})
    df = df[~dupes]

    # DQ3: Outlet ID format
    bad_oid = check_format(df, col="Outlet_ID", fmt="outlet_id")
    rejected.append(quarantine(df[bad_oid], reason="invalid_outlet_id_format", check_name="format_outlet"))
    checks_log.append({"check": "format_outlet_id", "flagged_count": int(bad_oid.sum())})
    df = df[~bad_oid]

    # ── Data Forensics: Fix known SFA system typos ──
    outlet_type_map = {
        "Grocry": "Grocery",
        "Bakry": "Bakery",
        " Eatery ": "Eatery",
    }
    df["Outlet_Type"] = df["Outlet_Type"].str.strip().replace(outlet_type_map)

    # Standardize Outlet_Size
    df["Outlet_Size"] = df["Outlet_Size"].str.strip().str.title()
    df["Outlet_Size"] = df["Outlet_Size"].replace({"": np.nan, "None": np.nan})

    # DQ4: Outlet_Type whitelist (after fixing typos)
    valid_types = ["Grocery", "Bakery", "Eatery", "Hotel", "Kiosk", "Pharmacy", "SMMT"]
    bad_type = check_referential_integrity(df, fk_col="Outlet_Type", ref_values=valid_types)
    rejected.append(quarantine(df[bad_type], reason="invalid_outlet_type", check_name="ref_integrity_type"))
    checks_log.append({"check": "ref_integrity_type", "flagged_count": int(bad_type.sum())})
    df = df[~bad_type]

    # DQ5: Cooler count range
    if "Cooler_Count" in df.columns:
        df["Cooler_Count"] = pd.to_numeric(df["Cooler_Count"], errors="coerce").fillna(0).astype(int)
        bad_cooler = check_value_range(df, col="Cooler_Count", min_val=0, max_val=20)
        rejected.append(quarantine(df[bad_cooler], reason="invalid_cooler_count", check_name="range_cooler"))
        checks_log.append({"check": "range_cooler", "flagged_count": int(bad_cooler.sum())})
        df = df[~bad_cooler]

    # Write
    df.to_parquet(os.path.join(SILVER_CLEAN, "outlets_clean.parquet"), index=False)
    all_rejected = _safe_concat_rejected(rejected)
    if not all_rejected.empty:
        all_rejected.to_csv(os.path.join(SILVER_REJECTED, "outlets_rejected.csv"), index=False)

    write_dq_summary("outlets", original_count, len(df), len(all_rejected), checks_log, SILVER_LOGS)
    print(f"  [SILVER] Outlets: {len(df):,} clean | {len(all_rejected):,} rejected")
    return df


def clean_geo():
    """Clean outlet_coordinates.csv — validate Sri Lanka bounding box."""
    print("\n── Cleaning: Outlet Coordinates ──")
    df = pd.read_csv(os.path.join(BRONZE_DIR, "outlet_coordinates.csv"))
    original_count = len(df)
    rejected = []
    checks_log = []

    # DQ1: Null coordinates
    null_mask = check_nulls(df, mandatory_cols=["Outlet_ID", "Latitude", "Longitude"])
    rejected.append(quarantine(df[null_mask], reason="null_coordinates", check_name="null_check"))
    checks_log.append({"check": "null_check", "flagged_count": int(null_mask.sum())})
    df = df[~null_mask]

    # DQ2: Duplicate outlet IDs
    dupes = check_duplicates(df, key_cols=["Outlet_ID"])
    rejected.append(quarantine(df[dupes], reason="duplicate_geo_outlet", check_name="duplicate_check"))
    checks_log.append({"check": "duplicate_check", "flagged_count": int(dupes.sum())})
    df = df[~dupes]

    # DQ3: Sri Lanka bounding box [lat: 5.9–9.9, lng: 79.5–81.9]
    bad_lat = check_value_range(df, col="Latitude", min_val=5.9, max_val=9.9)
    bad_lng = check_value_range(df, col="Longitude", min_val=79.5, max_val=81.9)
    out_of_bounds = bad_lat | bad_lng
    rejected.append(quarantine(df[out_of_bounds], reason="coordinates_outside_sri_lanka", check_name="geo_bounds"))
    checks_log.append({"check": "geo_bounds", "flagged_count": int(out_of_bounds.sum())})
    df = df[~out_of_bounds]

    # DQ4: Coordinate duplicates — flag for urban density proxy
    coord_dupes = df.duplicated(subset=["Latitude", "Longitude"], keep=False)
    df["coord_duplicate_flag"] = coord_dupes.astype(int)
    checks_log.append({"check": "coord_duplicate_flag", "flagged_count": int(coord_dupes.sum())})

    # DQ5: Format check
    bad_oid = check_format(df, col="Outlet_ID", fmt="outlet_id")
    rejected.append(quarantine(df[bad_oid], reason="invalid_outlet_id_format_geo", check_name="format_check"))
    checks_log.append({"check": "format_outlet_id", "flagged_count": int(bad_oid.sum())})
    df = df[~bad_oid]

    # Write
    df.to_parquet(os.path.join(SILVER_CLEAN, "geo_clean.parquet"), index=False)
    all_rejected = _safe_concat_rejected(rejected)
    if not all_rejected.empty:
        all_rejected.to_csv(os.path.join(SILVER_REJECTED, "geo_rejected.csv"), index=False)

    write_dq_summary("geo", original_count, len(df), len(all_rejected), checks_log, SILVER_LOGS)
    print(f"  [SILVER] Geo: {len(df):,} clean | {len(all_rejected):,} rejected")
    return df


def clean_seasonality():
    """Clean distributor_seasonality_details.csv — encode categorical seasonality."""
    print("\n── Cleaning: Seasonality ──")
    df = pd.read_csv(os.path.join(BRONZE_DIR, "distributor_seasonality_details.csv"))
    original_count = len(df)
    rejected = []
    checks_log = []

    # DQ1: Null check
    null_mask = check_nulls(df, mandatory_cols=["Distributor_ID", "Year", "Month", "Seasonality_Index"])
    rejected.append(quarantine(df[null_mask], reason="null_seasonality_field", check_name="null_check"))
    checks_log.append({"check": "null_check", "flagged_count": int(null_mask.sum())})
    df = df[~null_mask]

    # DQ2: Distributor ID referential integrity
    bad_dist = check_referential_integrity(df, fk_col="Distributor_ID", ref_values=VALID_DISTRIBUTORS)
    rejected.append(quarantine(df[bad_dist], reason="invalid_distributor_id", check_name="ref_integrity"))
    checks_log.append({"check": "ref_integrity", "flagged_count": int(bad_dist.sum())})
    df = df[~bad_dist]

    # DQ3: Seasonality_Index whitelist
    valid_seasonality = ["Favorable", "Moderate", "Un-Favorable"]
    bad_season = check_referential_integrity(df, fk_col="Seasonality_Index", ref_values=valid_seasonality)
    rejected.append(quarantine(df[bad_season], reason="invalid_seasonality_value", check_name="ref_seasonality"))
    checks_log.append({"check": "ref_seasonality", "flagged_count": int(bad_season.sum())})
    df = df[~bad_season]

    # Encode seasonality as numeric: Favorable=1.15, Moderate=1.0, Un-Favorable=0.85
    seasonality_map = {"Favorable": 1.15, "Moderate": 1.00, "Un-Favorable": 0.85}
    df["Seasonality_Numeric"] = df["Seasonality_Index"].map(seasonality_map)

    # Write
    df.to_parquet(os.path.join(SILVER_CLEAN, "seasonality_clean.parquet"), index=False)
    all_rejected = _safe_concat_rejected(rejected)
    if not all_rejected.empty:
        all_rejected.to_csv(os.path.join(SILVER_REJECTED, "seasonality_rejected.csv"), index=False)

    write_dq_summary("seasonality", original_count, len(df), len(all_rejected), checks_log, SILVER_LOGS)
    print(f"  [SILVER] Seasonality: {len(df):,} clean")
    return df


def clean_holidays():
    """Clean holiday_list.csv — parse ISO dates."""
    print("\n── Cleaning: Holidays ──")
    df = pd.read_csv(os.path.join(BRONZE_DIR, "holiday_list.csv"))
    original_count = len(df)
    rejected = []
    checks_log = []

    # DQ1: Null check
    null_mask = check_nulls(df, mandatory_cols=["Date", "Holiday_Name"])
    rejected.append(quarantine(df[null_mask], reason="null_holiday_field", check_name="null_check"))
    checks_log.append({"check": "null_check", "flagged_count": int(null_mask.sum())})
    df = df[~null_mask]

    # DQ2: Date format
    bad_date = check_format(df, col="Date", fmt="date")
    rejected.append(quarantine(df[bad_date], reason="invalid_date_format", check_name="format_date"))
    checks_log.append({"check": "format_date", "flagged_count": int(bad_date.sum())})
    df = df[~bad_date]

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # DQ3: Duplicate holidays
    dupes = check_duplicates(df, key_cols=["Date", "Holiday_Name"])
    rejected.append(quarantine(df[dupes], reason="duplicate_holiday", check_name="duplicate_check"))
    checks_log.append({"check": "duplicate_check", "flagged_count": int(dupes.sum())})
    df = df[~dupes]

    # Write
    df.to_parquet(os.path.join(SILVER_CLEAN, "holidays_clean.parquet"), index=False)
    all_rejected = _safe_concat_rejected(rejected)
    if not all_rejected.empty:
        all_rejected.to_csv(os.path.join(SILVER_REJECTED, "holidays_rejected.csv"), index=False)

    write_dq_summary("holidays", original_count, len(df), len(all_rejected), checks_log, SILVER_LOGS)
    print(f"  [SILVER] Holidays: {len(df):,} clean")
    return df


def run_silver():
    """Run all Silver layer cleaning."""
    print("\n" + "=" * 60)
    print("  PHASE 2: SILVER LAYER — Data Forensics & Cleaning")
    print("=" * 60)

    clean_transactions()
    clean_outlets()
    clean_geo()
    clean_seasonality()
    clean_holidays()

    print("\n  [SILVER] ✓ All datasets cleaned. Quarantine stores written.")


if __name__ == "__main__":
    run_silver()
