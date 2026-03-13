# playground/train_maps.py

import pandas as pd
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import os
from math import radians, cos

# Import Playground Modules
from data_lib.config import H_INSTALL, H_DECLINE, WEIGHT_DECLINE, WEIGHT_INSTALL, \
    HEX_GRID_SIZES, COMPETITION_SEARCH_RADIUS_DEG, HEX_TILING_RADIUS_KM
import data_lib.config as config
from data_lib.data_fetch.get_data import get_train_data
from data_lib.feature.spatial_weights import build_desirability_field_idw
from data_lib.geometry.hex import find_best_hexes, create_hex_grid, compute_hexes
from data_lib.geometry.find_boundary import run_find_boundary
from data_lib.test import get_overlap


def process_single_partner(partner_id, df_train, bad_se, mid_se):
    """
    Worker function to process one partner's hexagons.
    """
    sub_df = df_train[df_train["partner_id"] == partner_id].copy()

    if len(sub_df) < 5:  # Min samples
        return []

    center_lat = np.median(sub_df["latitude"])
    center_lon = np.median(sub_df["longitude"])

    # Determine best hex size using configured search sizes and radius
    best_size = find_best_hexes(
        center_lat, center_lon, 
        sub_df,
        hex_sizes=HEX_GRID_SIZES,
        radius_km=HEX_TILING_RADIUS_KM,
    )

    # Generate Grid using configured tiling radius
    hexes = create_hex_grid(
        center_lat,
        center_lon,
        radius_km=HEX_TILING_RADIUS_KM,
        hex_size_km=best_size,
    )

    # Compute Stats — NOW WITH reference_date for temporal buckets
    hex_stats = compute_hexes(
        hexes,
        center_lat,
        center_lon,
        sub_df,
        bad_se,
        mid_se,
        best_size,
        partner_id,
        reference_date=config.TRAIN_END_DATE,
    )

    return hex_stats


def main():
    print("--- STARTING MAP TRAINING (PLAYGROUND) ---")

    # Base directory resolution
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    ARTIFACTS_DIR = os.path.join(PARENT_DIR, "artifacts")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Get Data
    print(f"Train window: {config.TRAIN_START_DATE} -> {config.TRAIN_END_DATE}")
    df_train = get_train_data(config.TRAIN_START_DATE, config.TRAIN_END_DATE)
    if df_train.empty:
        print("No training data. Exiting.")
        return

    # 2. Build Desirability Fields
    print("Building Desirability Fields...")
    df_train = build_desirability_field_idw(
        df_train,
        radius_meters=max(H_INSTALL, H_DECLINE),
        decline_weight=WEIGHT_DECLINE,
        install_weight=WEIGHT_INSTALL,
    )

    # Set 'h' based on weight sign (Custom Logic for Split H)
    df_train["h"] = np.where(df_train["field_weight"] >= 0, H_INSTALL, H_DECLINE)

    # Save Train Data Artifact
    train_h5_path = os.path.join(ARTIFACTS_DIR, "train_data.h5")
    df_train.to_hdf(train_h5_path, mode="w", key="df")
    print(f"Saved {train_h5_path}")

    # 3. Hex Generation (Parallel)
    print("Generating Hexagons...")
    partners = df_train["partner_id"].unique().tolist()

    # Calculate SE thresholds
    df_train["is_installed"] = (df_train["final_decision"] == "INSTALLED").astype(int)
    df_train["is_declined"] = (df_train["final_decision"] == "DECLINED").astype(int)
    df_train["is_indeterminate"] = (
        df_train["final_decision"] == "INDETERMINATE"
    ).astype(int)
    df_train["is_hanging"] = (df_train["final_decision"] == "HANGING").astype(int)
    df_train = df_train[df_train["final_decision"].isin(["INSTALLED", "DECLINED"])].copy()

    dfp = df_train.groupby("partner_id").agg(
        total=("mobile", "count"), installed=("is_installed", "sum")
    ).reset_index()
    dfp["se"] = dfp["installed"] / dfp["total"]
    dfp["se_rng"] = pd.qcut(dfp["se"], q=10, labels=False, duplicates="drop") + 1

    bad_se_rng = config.BAD_SE
    mid_se_rng = config.MID_SE

    bad_se = dfp[dfp["se_rng"].isin(bad_se_rng)]["se"].max()
    mid_se = dfp[dfp["se_rng"].isin(mid_se_rng)]["se"].max()

    hexagons = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = {
            executor.submit(process_single_partner, pid, df_train, bad_se, mid_se): pid
            for pid in partners
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Partners"):
            hexagons.extend(future.result())

    # LET DICTS DEFINE SCHEMA — no hardcoded columns list
    df_hex = pd.DataFrame(hexagons)

    # Verify temporal columns made it through
    temporal_check = [f"se_{wd}d" for wd in config.TEMPORAL_WINDOWS]
    present = [c for c in temporal_check if c in df_hex.columns]
    missing = [c for c in temporal_check if c not in df_hex.columns]
    print(f"[HEX] Temporal columns present: {present}")
    if missing:
        print(f"[HEX] WARNING: Temporal columns missing: {missing}")

    # Temporary save for find_boundary
    temp_poly_path = "artifacts/poly_stats.h5"
    df_hex.to_hdf(temp_poly_path, mode="w", key="df")
    print(f"Saved intermediate {temp_poly_path} ({len(df_hex)} hexes, {len(df_hex.columns)} cols)")

    # 4. Find Boundaries
    print("Finding Boundaries...")
    run_find_boundary()

    # Move result to artifacts
    bound_path = "partner_cluster_boundaries.h5"
    artifact_bound_path = os.path.join(ARTIFACTS_DIR, "partner_cluster_boundaries.h5")

    if os.path.exists(bound_path):
        import shutil
        shutil.move(bound_path, artifact_bound_path)
        print(f"Moved {bound_path} to {artifact_bound_path}")

    # 5. Competition / Overlaps (Final Map)
    print("Computing Competition Overlaps...")

    if os.path.exists(artifact_bound_path):
        import shutil
        shutil.copy(artifact_bound_path, bound_path)

    get_overlap(
        search_radius_deg=COMPETITION_SEARCH_RADIUS_DEG
    )

    final_poly_path = "poly_stats_final.h5"
    artifact_final_poly_path = os.path.join(ARTIFACTS_DIR, "poly_stats_final.h5")

    if os.path.exists(final_poly_path):
        import shutil
        shutil.move(final_poly_path, artifact_final_poly_path)
        print(f"Saved {artifact_final_poly_path}")

    print("--- MAP TRAINING COMPLETE ---")


if __name__ == "__main__":
    main()