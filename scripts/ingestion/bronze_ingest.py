"""
=========================================================================
Bronze Layer — Raw Data Ingestion
=========================================================================
Data Storm v7.0

Rule: ZERO transformations. Preserve every raw byte.
Copies raw CSVs from the source directory into the bronze layer with:
  - MD5 checksums for data lineage
  - Row/column manifest for auditing
=========================================================================
"""

import os
import sys
import shutil
import hashlib
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_md5(filepath: str) -> str:
    """Compute MD5 checksum of a file for data lineage tracking."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_bronze(source_dir: str = None) -> pd.DataFrame:
    """
    Ingest all raw CSV files into the bronze layer.
    """
    if source_dir is None:
        source_dir = os.path.join(PROJECT_ROOT, "data", "bronze")

    # Expected files based on competition spec
    EXPECTED_FILES = [
        "transactions_history_final.csv",
        "outlet_master.csv",
        "outlet_coordinates.csv",
        "distributor_seasonality_details.csv",
        "holiday_list.csv",
    ]

    # Create raw directory and copy originals there as backup
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    manifest = []
    print("\n" + "=" * 60)
    print("  PHASE 1: BRONZE LAYER — Raw Ingestion")
    print("=" * 60)

    for filename in EXPECTED_FILES:
        filepath = os.path.join(source_dir, filename)

        if not os.path.exists(filepath):
            print(f"  [BRONZE] ⚠ MISSING: {filename}")
            continue

        # Copy to raw/ for backup (preserving originals)
        raw_backup = os.path.join(raw_dir, filename)
        if not os.path.exists(raw_backup):
            shutil.copy2(filepath, raw_backup)

        # Compute checksum
        md5 = compute_md5(filepath)

        # Read to get schema info (without modifying)
        try:
            df = pd.read_csv(filepath, nrows=0)  # Just headers
            row_count = sum(1 for _ in open(filepath)) - 1  # Minus header
            columns = list(df.columns)
        except Exception as e:
            print(f"  [BRONZE] ⚠ Error reading {filename}: {e}")
            row_count = -1
            columns = []

        file_size = os.path.getsize(filepath)

        manifest.append({
            "file": filename,
            "path": os.path.relpath(filepath, PROJECT_ROOT).replace('\\', '/'),
            "rows": row_count,
            "columns": columns,
            "column_count": len(columns),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "md5_checksum": md5,
            "ingested_at": datetime.utcnow().isoformat(),
        })

        print(f"  [BRONZE] ✓ {filename}: {row_count:,} rows, "
              f"{len(columns)} cols, {file_size / (1024*1024):.1f} MB, "
              f"MD5={md5[:12]}...")

    # Write ingestion manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(source_dir, "ingestion_manifest.json")
    manifest_df.to_json(manifest_path, orient="records", indent=2)
    print(f"\n  [BRONZE] Manifest written: {manifest_path}")
    print(f"  [BRONZE] Total files: {len(manifest)}/{len(EXPECTED_FILES)}")
    print(f"  [BRONZE] Total rows: {sum(m['rows'] for m in manifest):,}")

    return manifest_df


if __name__ == "__main__":
    ingest_bronze()
