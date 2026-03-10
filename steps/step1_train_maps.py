# playground/train_maps.py

import pandas as pd
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import os
from math import radians, cos

# Import Playground Modules
from lib.config import H_INSTALL, H_DECLINE, WEIGHT_DECLINE, WEIGHT_INSTALL, \
    HEX_GRID_SIZES, COMPETITION_SEARCH_RADIUS_DEG, HEX_TILING_RADIUS_KM
import lib.config as config
from lib.data_fetch.get_data import get_train_data
from lib.feature.spatial_weights import build_desirability_field_idw
from lib.geometry.hex import find_best_hexes, create_hex_grid, compute_hexes
from lib.geometry.find_boundary import run_find_boundary
from lib.test import get_overlap


def process_single_partner(partner_id, df_train, bad_se, mid_se):
    """
    Worker function to process one partner's hexagons.
    """
    sub_df = df_train[df_train["partner_id"] == partner_id].copy()

    if len(sub_df) < 50:  # Min samples
        return []

    center_lat = np.median(sub_df["latitude"])
    center_lon = np.median(sub_df["longitude"])

    METERS_PER_DEG_LAT = 111320.0
    cos_lat = cos(radians(center_lat))

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

    # Compute Stats
    hex_stats = compute_hexes(
        hexes,
        center_lat,
        center_lon,
        sub_df,
        bad_se,
        mid_se,
        best_size,
        METERS_PER_DEG_LAT,
        cos_lat,
        partner_id,
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
    # Note: build_desirability_field_idw in lib_spatial_weights.py uses hardcoded params?
    # We should update lib_spatial_weights.py to take args if we want to experiment.
    # I will call it with the args I see in the file signature if available.
    # Looking at lib_spatial_weights.py signature:
    # def build_desirability_field_idw(df, radius_meters=100.0, decline_weight=-2.0, install_weight=1.0, power=2.0)

    # We use our config values
    df_train = build_desirability_field_idw(
        df_train,
        radius_meters=max(H_INSTALL, H_DECLINE),  # Use max radius
        decline_weight=WEIGHT_DECLINE,
        install_weight=WEIGHT_INSTALL,
    )

    """
    # Add Time Decay: THIS SHOULD BE DONE WHEN SCORING TEST LEADS, WRT THEIR DECISION TIME, NOT HERE.
    eval_time = pd.Timestamp.now()
    ages_days = (eval_time - df_train['decision_time']).dt.total_seconds() / 86400
    df_train['field_weight'] = df_train['field_weight'] * np.exp(-LAMBDA_DECAY * ages_days)
    """
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

    df_hex = pd.DataFrame(
        hexagons,
        columns=[
            "partner_id",
            "best_size",
            "poly_id",
            "poly",
            "se",
            "installs",
            "declines",
            "total",
            "color",
        ],
    )

    # Temporary save for find_boundary - use CWD for compatibility with existing script imports
    temp_poly_path = "poly_stats.h5"
    df_hex.to_hdf(temp_poly_path, mode="w", key="df")
    print("Saved intermediate poly_stats.h5")

    # 4. Find Boundaries
    print("Finding Boundaries...")

    # Ensure run_find_boundary can find the file.
    # run_find_boundary() uses 'poly_stats.h5' from current working directory.
    # Since we just saved it there, it should be fine.
    run_find_boundary()  # Generates partner_cluster_boundaries.h5 in CWD

    # Move result to artifacts
    bound_path = "partner_cluster_boundaries.h5"
    artifact_bound_path = os.path.join(ARTIFACTS_DIR, "partner_cluster_boundaries.h5")

    if os.path.exists(bound_path):
        import shutil

        shutil.move(bound_path, artifact_bound_path)
        print(f"Moved {bound_path} to {artifact_bound_path}")

    # 5. Competition / Overlaps (Final Map)
    print("Computing Competition Overlaps...")

    # test.get_overlap reads poly_stats.h5 and partner_cluster_boundaries.h5 from CWD
    # We need to ensure partner_cluster_boundaries.h5 is in CWD for it.
    if os.path.exists(artifact_bound_path):
        import shutil

        shutil.copy(artifact_bound_path, bound_path)

    get_overlap(
        search_radius_deg=COMPETITION_SEARCH_RADIUS_DEG
    )  # Generates poly_stats_final.h5 in CWD

    final_poly_path = "poly_stats_final.h5"
    artifact_final_poly_path = os.path.join(ARTIFACTS_DIR, "poly_stats_final.h5")

    if os.path.exists(final_poly_path):
        import shutil

        shutil.move(final_poly_path, artifact_final_poly_path)
        print(f"Saved {artifact_final_poly_path}")

    print("--- MAP TRAINING COMPLETE ---")


if __name__ == "__main__":
    main()
