"""
=========================================================================
Reusable Data Quality (DQ) Check Framework
=========================================================================
Contains parameterizable functions for validating data integrity.
Used extensively in the Silver Layer.
=========================================================================
"""

import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime

def check_nulls(df: pd.DataFrame, mandatory_cols: list) -> pd.Series:
    """Returns a boolean mask of rows where ANY mandatory column is null."""
    return df[mandatory_cols].isnull().any(axis=1)

def check_duplicates(df: pd.DataFrame, key_cols: list) -> pd.Series:
    """Returns a boolean mask of rows that have duplicate keys."""
    return df.duplicated(subset=key_cols, keep=False)

def check_value_range(df: pd.DataFrame, col: str, min_val: float = None, max_val: float = None) -> pd.Series:
    """Returns a boolean mask of rows outside the specified range."""
    mask = pd.Series(False, index=df.index)
    if min_val is not None:
        mask |= (df[col] < min_val)
    if max_val is not None:
        mask |= (df[col] > max_val)
    return mask

def check_format(df: pd.DataFrame, col: str, fmt: str) -> pd.Series:
    """Returns a boolean mask of rows that fail the format regex check."""
    patterns = {
        "outlet_id": r"^OUT_\d+$",
        "date": r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}Z)?$"
    }
    regex = patterns.get(fmt, fmt)
    # Return true for rows that DO NOT match
    return ~df[col].astype(str).str.match(regex, na=False)

def check_referential_integrity(df: pd.DataFrame, fk_col: str, ref_values: list) -> pd.Series:
    """Returns a boolean mask of rows where fk_col is NOT in ref_values."""
    return ~df[fk_col].isin(ref_values)

def check_statistical_outlier(df: pd.DataFrame, col: str, std_devs: float = 3.0) -> pd.Series:
    """Returns mask of rows outside standard deviations."""
    mean = df[col].mean()
    std = df[col].std()
    return (df[col] < mean - std_devs * std) | (df[col] > mean + std_devs * std)

def check_cross_field_consistency(df: pd.DataFrame, condition_expr: str, description: str = "") -> pd.Series:
    """Returns boolean mask of rows that FAIL the condition expression."""
    # df.eval returns True where condition is met. We return the inverse (fails).
    return ~df.eval(condition_expr)

def check_ghost_entries(df: pd.DataFrame, group_col: str, value_col: str, sort_cols: list = None, consecutive_threshold: int = 3) -> pd.Series:
    """
    Detects repeated identical values which suggest automated/system default inputs.
    Returns a mask of rows that are part of a 'ghost' sequence.
    """
    if sort_cols:
        df = df.sort_values(sort_cols)
    
    # Calculate streaks of identical values within each group
    group = df.groupby(group_col)[value_col]
    
    # A change happens when current value != previous value
    change = group.diff().fillna(1) != 0
    
    # Cumulative sum creates a unique ID for each continuous streak
    streak_id = change.cumsum()
    
    # Count the size of each streak
    streak_sizes = df.groupby([group_col, streak_id])[value_col].transform("size")
    
    return streak_sizes >= consecutive_threshold

def quarantine(df_subset: pd.DataFrame, reason: str, check_name: str) -> pd.DataFrame:
    """Adds metadata to failed records for auditing."""
    if df_subset.empty:
        return pd.DataFrame()
    out = df_subset.copy()
    out["rejection_reason"] = reason
    out["dq_check_name"] = check_name
    out["rejected_at"] = datetime.utcnow().isoformat()
    return out

def write_dq_summary(dataset_name: str, original_count: int, clean_count: int, rejected_count: int, checks_log: list, output_dir: str):
    """Writes a JSON summary log of all DQ checks run on a dataset."""
    summary = {
        "dataset": dataset_name,
        "timestamp": datetime.utcnow().isoformat(),
        "original_row_count": original_count,
        "clean_row_count": clean_count,
        "rejected_row_count": rejected_count,
        "rejection_rate_pct": round((rejected_count / max(original_count, 1)) * 100, 2),
        "checks_applied": checks_log
    }
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"dq_log_{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    for check in checks_log:
        if check.get("flagged_count", 0) > 0:
            print(f"  [DQ] {check['check']}: {check['flagged_count']} rows flagged")
