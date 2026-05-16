"""
=========================================================================
POI Enrichment — OpenStreetMap Overpass API Scraper
=========================================================================
Scrapes Point of Interest data for each outlet using concentric rings.
Implements caching for crash-resilience and rate limiting.
=========================================================================
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

POI_TYPES = {
    "schools":         '[amenity~"school|college|university"]',
    "bus_stands":      '[highway=bus_stop]',
    "hospitals":       '[amenity~"hospital|clinic|pharmacy"]',
    "hotels":          '[tourism~"hotel|guesthouse|hostel"]',
    "restaurants":     '[amenity~"restaurant|cafe|fast_food"]',
    "fuel_stations":   '[amenity=fuel]',
    "religious_sites": '[amenity~"place_of_worship"]',
    "markets":         '[amenity~"marketplace|supermarket"]',
    "atms_banks":      '[amenity~"bank|atm"]',
}

RADII_METERS = [250, 500, 1000]


def build_overpass_query(lat, lng, radius, osm_filter):
    return f"""
    [out:json][timeout:25];
    (
      node{osm_filter}(around:{radius},{lat},{lng});
      way{osm_filter}(around:{radius},{lat},{lng});
    );
    out count;
    """


def count_pois_for_outlet(lat, lng):
    result = {}
    for poi_name, osm_filter in POI_TYPES.items():
        for radius in RADII_METERS:
            key = f"{poi_name}_{radius}m"
            query = build_overpass_query(lat, lng, radius, osm_filter)
            try:
                resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
                data = resp.json()
                count = data.get("elements", [{}])[0].get("tags", {}).get("total", 0)
                result[key] = int(count)
            except Exception:
                result[key] = -1
            time.sleep(0.5)
    return result


def scrape_all_outlets(batch_size=50):
    """Scrape POI data for all outlets. Uses file-based cache for resume."""
    cache_dir = os.path.join(PROJECT_ROOT, "data", "gold", "poi")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "poi_raw_cache.json")

    geo = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "silver", "cleaned", "geo_clean.parquet"))
    outlets = geo[["Outlet_ID", "Latitude", "Longitude"]].drop_duplicates("Outlet_ID")

    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}

    print(f"\n  [POI] Starting scrape for {len(outlets):,} outlets "
          f"({len(cache)} already cached)")

    records = []
    for _, row in tqdm(outlets.iterrows(), total=len(outlets), desc="Scraping POIs"):
        oid = str(row["Outlet_ID"])
        if oid in cache:
            records.append({"Outlet_ID": oid, **cache[oid]})
            continue

        poi_counts = count_pois_for_outlet(row["Latitude"], row["Longitude"])
        cache[oid] = poi_counts
        records.append({"Outlet_ID": oid, **poi_counts})

        if len(records) % batch_size == 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f)

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    poi_df = pd.DataFrame(records)

    # Composite features
    for radius in RADII_METERS:
        cols = [f"{p}_{radius}m" for p in POI_TYPES if f"{p}_{radius}m" in poi_df.columns]
        poi_df[f"total_poi_{radius}m"] = poi_df[cols].clip(lower=0).sum(axis=1)

    # Catchment score: weighted sum (proximity = more weight)
    poi_df["catchment_score"] = (
        poi_df.get("total_poi_250m", 0) * 3.0 +
        poi_df.get("total_poi_500m", 0) * 1.5 +
        poi_df.get("total_poi_1000m", 0) * 0.5
    )

    out_path = os.path.join(PROJECT_ROOT, "data", "gold", "poi", "poi_features.parquet")
    poi_df.to_parquet(out_path, index=False)
    print(f"  [POI] Scraped {len(poi_df):,} outlets → {out_path}")
    return poi_df


def generate_synthetic_poi(seed=42):
    """
    Generate synthetic POI features based on coordinates for development/testing
    when Overpass API is unavailable or too slow for 20k outlets.
    Uses lat/lng density as a proxy for urbanization.
    """
    print("\n  [POI] Generating synthetic POI features from coordinate density...")
    geo = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "silver", "cleaned", "geo_clean.parquet"))

    np.random.seed(seed)

    # Urban density proxy from coordinate clustering
    from sklearn.neighbors import BallTree
    coords = np.radians(geo[["Latitude", "Longitude"]].values)
    tree = BallTree(coords, metric="haversine")

    poi_df = geo[["Outlet_ID", "Latitude", "Longitude"]].copy()

    for radius_m in RADII_METERS:
        radius_rad = radius_m / 6371000  # Earth radius in meters
        counts = tree.query_radius(coords, r=radius_rad, count_only=True)
        density = counts / max(counts.max(), 1)  # Normalize

        for poi_name in POI_TYPES:
            # Scale by density + random noise
            base = (density * np.random.uniform(2, 15)) + np.random.poisson(1, len(geo))
            poi_df[f"{poi_name}_{radius_m}m"] = base.astype(int).clip(0)

        cols = [f"{p}_{radius_m}m" for p in POI_TYPES]
        poi_df[f"total_poi_{radius_m}m"] = poi_df[cols].sum(axis=1)

    poi_df["catchment_score"] = (
        poi_df["total_poi_250m"] * 3.0 +
        poi_df["total_poi_500m"] * 1.5 +
        poi_df["total_poi_1000m"] * 0.5
    )

    out_path = os.path.join(PROJECT_ROOT, "data", "gold", "poi", "poi_features.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    poi_df.to_parquet(out_path, index=False)
    print(f"  [POI] Synthetic POI features for {len(poi_df):,} outlets → {out_path}")
    return poi_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic POI data instead of scraping")
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_poi()
    else:
        scrape_all_outlets()
