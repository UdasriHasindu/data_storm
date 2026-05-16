# Data Storm — Latent Maximum Monthly Purchase Potential

## Quick Start

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run full pipeline (with synthetic POI for dev)
python scripts/orchestration/pipeline.py --synthetic

# 3. Or run individual phases
python scripts/orchestration/pipeline.py --phase bronze
python scripts/orchestration/pipeline.py --phase silver
python scripts/orchestration/pipeline.py --phase poi --synthetic
python scripts/orchestration/pipeline.py --phase gold
python scripts/orchestration/pipeline.py --phase model
python scripts/orchestration/pipeline.py --phase output
```

## Project Structure

```
data_storm/
├── data/
│   ├── raw/                          # Backup of original files
│   ├── bronze/                       # Raw CSVs (untouched)
│   ├── silver/
│   │   ├── cleaned/                  # Validated parquet files
│   │   ├── validated/
│   │   ├── rejected/                 # Quarantined records with reasons
│   │   └── logs/                     # DQ check summaries (JSON)
│   └── gold/
│       ├── features/                 # Sales & outlet feature tables
│       ├── poi/                      # POI features & cache
│       ├── analytics/
│       └── modeling/                 # Model-ready dataset
├── notebooks/                        # EDA & analysis notebooks
├── outputs/
│   ├── predictions/                  # Final submission CSV
│   ├── visualizations/
│   └── reports/
├── scripts/
│   ├── ingestion/                    # Bronze layer
│   ├── validation/                   # Reusable DQ check functions
│   ├── cleaning/                     # Silver layer
│   ├── enrichment/                   # POI scraping
│   ├── feature_engineering/          # Gold layer
│   ├── modeling/                     # Clustering, Tobit, Quantile
│   └── orchestration/               # Master pipeline
├── tests/
├── requirements.txt
├── README.md
└── AGENT.md
```

## Pipeline Architecture

```
Bronze (Raw) → Silver (Cleaned) → Gold (Enriched) → Model → Output
                  ↓                                    ↑
            Rejected Store                        POI Enrichment
```

### Bronze Layer (`scripts/ingestion/bronze_ingest.py`)

- **Zero transformations** — preserves raw CSV data exactly as provided
- Computes **MD5 checksums** for each file (data lineage)
- Extracts **metadata**: row counts, column names, file sizes
- Outputs `ingestion_manifest.json` for audit trail

### Silver Layer (`scripts/cleaning/silver_clean.py`)

**8 reusable DQ checks** (from `scripts/validation/dq_checks.py`):

1. **Null check** — Missing mandatory fields
2. **Duplicate check** — Composite key validation
3. **Value range check** — Numeric bounds (e.g., Volume ≥ 0.001L)
4. **Format check** — Regex pattern matching (e.g., OUT_XXXX)
5. **Referential integrity** — Foreign key constraints
6. **Statistical outlier** — > 99.9th percentile bounds
7. **Cross-field consistency** — Multi-column logic
8. **Ghost entry detection** — Consecutive identical values (stagnation indicator)

**Outputs**:

- Cleaned parquets: `data/silver/cleaned/*.parquet`
- Rejected records: `data/silver/rejected/*.csv` (with rejection reasons)
- DQ logs: `data/silver/logs/dq_log_*.json` (audit trail)

### Gold Layer (`scripts/feature_engineering/gold_features.py`)

**Sales Features** (aggregated by outlet):

- `mean_vol`, `median_vol`, `p75_vol`, `p90_vol`, `p95_vol`, `max_vol` — distribution
- `cov` — coefficient of variation (volatility)
- `vol_trend` — linear growth/decline
- `flatline_score` — % of months at mode volume (constraint proxy)
- `ghost_rate` — frequency of automated-looking entries
- `holiday_uplift_ratio` — seasonal boost factor

**Outlet Features**:

- `Outlet_Type` — encoded (Grocery, Bakery, Eatery, Hotel, Kiosk, Pharmacy, SMMT)
- `Outlet_Size_Num` — numerical (Small=1, Medium=2, Large=3)
- `Cooler_Count` — number of coolers
- `Province` — mapped from distributor

**POI Features** (Points of Interest):

- Concentric ring counts (200m, 500m, 1km) for: schools, hospitals, restaurants, ATMs
- Urban density proxy from multi-outlet locations
- `catchment_score` — weighted proximity score

**Seasonality**:

- `seasonality_index` for January 2026 (Favorable=1.15, Moderate=1.0, Unfavorable=0.85)

### Modeling (`scripts/modeling/`)

**Step 1: Clustering** (`clustering.py`)

- **K-Means** auto-selects optimal k (6-20) via silhouette score
- Groups outlets into peer groups based on 11 features
- Output: `cluster_id` (0 to k-1)

**Step 2: Constraint Detection** (`potential_ceiling.py`)

- Flags outlets with flatline_score > 0.60 or ghost_rate > 0.30
- Marks as "constrained" (supply-limited, not demand-limited)

**Step 3: Tobit Regression** (`tobit_model.py`)

- **Problem**: Observed sales are left-censored (shelf-space floor)
- **Solution**: Estimates latent (unobserved) demand via censored regression
- **Outputs**:
  - `tobit_potential` — linear prediction
  - `tobit_uncensored` — latent demand with Heckman Mills ratio correction

**Step 4: Quantile Regression** (`quantile_model.py`)

- **LightGBM** trained on 90th percentile of historical volumes
- Captures non-linear relationships (e.g., outlet size → ceiling)
- **Output**: `quantile_potential` (ML-learned ceiling)

**Step 5: Peer Benchmarking** (`potential_ceiling.py`)

- Computes cluster p90 ceiling (top performers in each peer group)
- For constrained outlets: potential = cluster ceiling
- For unconstrained: potential = 85% of max(observed, cluster_ceiling)

**Step 6: Final Ensemble** (`potential_ceiling.py`)

- **Weights**: 40% Tobit + 35% Peer + 25% Quantile
- **Seasonality adjustment**: multiply by January seasonal index
- **Sanity floor**: must be ≥ outlet's historical mean

## Team
Data Storm Team — University of Moratuwa
