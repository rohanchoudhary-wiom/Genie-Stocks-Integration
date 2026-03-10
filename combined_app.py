

# ===== FILE: ./combined_app.py =====



# ===== FILE: ./create_combined.py =====

import os

APP_DIR = "./"
OUTPUT_FILE = "combined_app.py"

EXCLUDE_DIRS = {"venv", ".venv", "__pycache__"}

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for root, dirs, files in os.walk(APP_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in sorted(files):
            if file.endswith(".py"):
                path = os.path.join(root, file)
                out.write(f"\n\n# ===== FILE: {path} =====\n\n")
                out.write(read_file(path))

print("Done ->", OUTPUT_FILE)

# ===== FILE: ./steps/step1_train_maps.py =====

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


# ===== FILE: ./steps/step2_score_test.py =====

# playground/score_test.py

import argparse
from typing import cast
import pandas as pd
import numpy as np
import os
import lib.config as config
import datetime as dt
from lib.data_fetch.get_data import get_test_data, get_g1_distance
from lib.compute import process
from lib.geometry.geometric_features import batch_compute_geometry, calculate_adaptive_h
from steps.step3_simpulate import run_declines_simulation


def evaluate_bucket(df, bucket_name):
    """
    Computes SE, Installs, and Volume for a specific slice of data.
    """
    total = len(df)
    if total == 0:
        return {"Bucket": bucket_name, "Vol": 0, "Installs": 0, "SE": 0.0}

    installs = df["installed_decision"].sum()
    se = installs / total
    return {
        "Bucket": bucket_name,
        "Vol": total,
        "Installs": int(installs),
        "SE": round(se, 4),
    }


def main(simulate=False):
    print("--- STARTING TEST SCORING (PLAYGROUND) ---")

    # Base directory resolution
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load Artifacts
    try:
        poly_path = os.path.join(ARTIFACTS_DIR, "poly_stats_final.h5")
        bound_path = os.path.join(ARTIFACTS_DIR, "partner_cluster_boundaries.h5")
        train_path = os.path.join(ARTIFACTS_DIR, "train_data.h5")

        df_poly = cast(pd.DataFrame, pd.read_hdf(poly_path, "df"))
        df_bound = cast(pd.DataFrame, pd.read_hdf(bound_path, "df"))
        df_train = cast(pd.DataFrame, pd.read_hdf(train_path, "df"))

        if config.DEFINITE_DECISIONS == 1:
            df_train = cast(
                pd.DataFrame,
                df_train[
                    df_train["final_decision"].isin(["DECLINED", "INSTALLED"])
                ].copy(),
            )

        # Initial default H
        df_train["h"] = np.where(
            df_train["field_weight"] >= 0, config.H_INSTALL, config.H_DECLINE
        )

        # Apply Adaptive H if enabled
        if config.USE_ADAPTIVE_H:
            print(
                f"Computing Adaptive H (k={config.ADAPTIVE_H_NEIGHBOR_K}, min={config.ADAPTIVE_H_MIN}m, max={config.ADAPTIVE_H_MAX}m)..."
            )
            df_train = calculate_adaptive_h(df_train)

        print("Loaded Maps & Training Data.")
    except FileNotFoundError:
        print("Artifacts not found. Run step1_train_maps.py first.")
        return

    # 2. Load Test Data
    print(f"Test window: {config.TEST_START_DATE} -> {config.TEST_END_DATE}")
    df_test = get_test_data(config.TEST_START_DATE, config.TEST_END_DATE)
    if df_test.empty:
        print("No test data.")
        return

    # Aggregate to mobile-level (one row per lead/location)
    before = len(df_test)
    df_test = df_test.groupby(["mobile", "latitude", "longitude"], as_index=False).agg(
        nmbr_partners=("partner_id", "count"),
        decision_time=("decision_time", "min"),
        installed_decision=("installed_decision", "max"),
        installed_time=("installed_time", "max"),
    )
    print(f"Aggregated test decisions: {before} rows -> {len(df_test)} unique mobiles")

    # 3. Geometric Features (New!)
    print("Computing Geometric 'Pattern-of-History' Features...")
    df_train_geom = df_train.copy()
    if config.GEOM_INSTALL_FILTER == 1:
        df_train_geom = df_train_geom[df_train_geom["installed_decision"] == 1].copy()

    geom_features = batch_compute_geometry(
        df_test, df_train_geom, radius_m=int(config.GEOM_SEARCH_RADIUS_M)
    )
    df_test = pd.concat([df_test.reset_index(drop=True), geom_features], axis=1)

    # Derive geometry regime labels from regime scores
    regimes = []
    dense_mask = df_test["dense_score"] >= config.GEOM_DENSE_THRESHOLD
    gully_mask = df_test["gully_score"] >= config.GEOM_GULLY_THRESHOLD
    sparse_mask = df_test["sparse_score"] >= config.GEOM_SPARSE_THRESHOLD

    def regime_row(d, g, s):
        hits = []
        if d:
            hits.append("dense")
        if g:
            hits.append("gully")
        if s:
            hits.append("sparse")
        return "+".join(hits) if hits else "mixed"

    df_test["geometry_regime"] = [
        regime_row(d, g, s) for d, g, s in zip(dense_mask, gully_mask, sparse_mask)
    ]

    # 4. Standard Scoring (Hex + Field + Boundaries)
    print("Running Production Scoring Engine...")
    # process() expects: df_source, df_target, df_poly, df_bound
    # It returns df_target with 'predicted_field_hex', 'parent_color', etc.
    scored_df = process(
        df_train,
        df_test,
        df_poly,
        df_bound,
        lambda_decay=config.LAMBDA_DECAY,
        max_radius_m=int(config.MIN_DIST_CUTOFF_M),
    )

    if scored_df is None or scored_df.empty:
        print("Scoring returned empty dataframe. Likely parent hex assignment failed.")
        print(
            "Check that poly_stats_final.h5 contains shapely polygons and expected columns."
        )
        return

    required_cols = [
        "parent_color",
        "parent_installs",
        "predicted_field_hex",
        "contested_field",
    ]
    missing = [c for c in required_cols if c not in scored_df.columns]
    if missing:
        print(f"Scoring output missing required columns: {missing}")
        print(f"Available columns: {scored_df.columns.tolist()}")
        return

    # Merge G1 distance logs (aligned to test window)
    test_start = dt.date.fromisoformat(config.TEST_START_DATE)
    g1_start = (
        test_start - dt.timedelta(days=config.G1_LOOKBACK_EXTRA_DAYS)
    ).isoformat()
    g1_end = config.TEST_END_DATE

    print(f"G1 logs window: {g1_start} -> {g1_end}")
    df_g1 = get_g1_distance(g1_start, g1_end)
    if df_g1.empty:
        print(
            "Warning: G1 distance data is empty for this window; min_dist buckets will be NaN"
        )
        df = scored_df.copy()
    else:
        keep_cols = [
            "mobile",
            "min_dist",
            "g1_is_bdo_lead",
            "serviceable",
            "zone_alias",
        ]
        df = scored_df.merge(df_g1[keep_cols], on="mobile", how="inner")

    # Create distance buckets (20 included in 0-20; 100+ bucket)
    bins = [-np.inf, 20, 40, 60, 100, np.inf]
    labels = ["0-20", "20-40", "40-60", "60-100", "100+"]
    df["min_dist_bucket"] = pd.cut(
        df["min_dist"], bins=bins, labels=labels, right=True, include_lowest=True
    )

    # Apply Final Decision Logic (production logic, WITHOUT distance gating)
    indeterminate_mask = (df["parent_color"] == "lightgreen") & (
        df["parent_installs"] <= config.INDETERMINATE_INSTALLS_CUTOFF
    )
    df["parent_color"] = np.where(
        indeterminate_mask, "indeterminate", df["parent_color"]
    )
    df["parent_color_super"] = np.where(
        df["parent_color"] == "indeterminate", "lightgreen", df["parent_color"]
    )



    df["gravity_score"] = np.where(
        df["predicted_field_hex"].isna(),
        -99,
        np.where((df["predicted_field_hex"] > config.FIELD_THRESHOLD) & (df['parent_total']>config.PARENT_TOTAL_THRESHOLD), 2, 
            np.where(df["predicted_field_hex"] > config.FIELD_THRESHOLD,1,0)
            )
    )
    
    """
    OLD 1.1:
    df["gravity_score"] = np.where(
        df["predicted_field_hex"].isna(),
        -99,
        np.where(df["predicted_field_hex"] > config.FIELD_THRESHOLD, 1, 0),
    )
    """

    df["comp_gravity_score"] = np.where(
        df["contested_field"].isna(),
        -99,
        np.where(df["contested_field"] > config.CONTEST_FIELD_THRESHOLD, 1, 0),
    )

    """
    OLD 1.2:

    bk_class_score_2 = (df["gravity_score"] == 1) & (df["comp_gravity_score"] == 1)
    bk_class_score_1 = df["gravity_score"] == 1

    df["bk_class_score"] = np.where(
        bk_class_score_2,
        "A. Comp+Field",
        np.where(bk_class_score_1, "B. Field", "C. Bad Field"),
    )
    
    booking_classifier_v1 = (df["parent_color_super"].isin(["lightgreen"])) & (
        df["bk_class_score"].isin(["A. Comp+Field", "B. Field"])
    )
    booking_classifier_v2 = (df["parent_color_super"].isin(["orange"])) & (
        df["bk_class_score"].isin(["A. Comp+Field"])
    )
    """
    
    conditions = [
        (df["gravity_score"]==2) & (df["comp_gravity_score"]==1),
        (df["gravity_score"]==2),
        (df["gravity_score"]==1) & (df["comp_gravity_score"]==1),
        (df["gravity_score"]==1),
        (df["comp_gravity_score"]==1),
    ]

    choices = [
        "A. STRONG Comp+ STRONG Field",
        "B. STRONG Field",
        "C. STRONG Comp + WEAK Field",
        "D. WEAK FIELD",
        "E. STRONG Comp"
    ]

    df["bk_class_score"] = np.select(conditions, choices, default="F. Bad Field")


    booking_classifier_v1 = (df["parent_color_super"].isin(["lightgreen"])) & (
        df["bk_class_score"].isin(["A. STRONG Comp+ STRONG Field", "B. STRONG Field", "C. STRONG Comp + WEAK Field", "E. STRONG Comp"])
    )
    booking_classifier_v2 = (df["parent_color_super"].isin(["orange"])) & (
        df["bk_class_score"].isin(["A. STRONG Comp+ STRONG Field", "B. STRONG Field"])
    )


    df["booking_classifier_new"] = np.where(
        booking_classifier_v1 | booking_classifier_v2, 1, 0
    )

    # NO distance gate: final_serviceability = booking_classifier_new
    df["final_serviceability"] = df["booking_classifier_new"]
    #df["final_serviceability"] = np.where((df["booking_classifier_new"] == 1) & (df['parent_total']>config.PARENT_TOTAL_THRESHOLD), 1, 0)
    

    # Geometric flags
    df["is_sparse"] = np.where(
        df["local_density"] < config.SPARSE_DENSITY_THRESHOLD, 1, 0
    )
    df["is_deep"] = np.where(
        df["depth_score_point_hex"] > config.DEPTH_SCORE_THRESHOLD, 1, 0
    )

    # 5. Evaluation / Bucketing
    print("\n--- RESULTS: SEPARATION ANALYSIS ---")

    results = []
    results.append(
        evaluate_bucket(df[df["parent_color_super"] == "lightgreen"], "Green Hex")
    )
    results.append(
        evaluate_bucket(df[df["parent_color_super"] == "orange"], "Orange Hex")
    )
    results.append(
        evaluate_bucket(df[df["parent_color_super"] == "crimson"], "Crimson Hex")
    )
    results.append(
        evaluate_bucket(
            df[
                (df["parent_color_super"] == "lightgreen")
                & (df["parent_overlap"] == True)
            ],
            "Green + Contested",
        )
    )
    results.append(
        evaluate_bucket(
            df[
                (df["parent_color_super"] == "lightgreen")
                & (df["parent_overlap"] == False)
            ],
            "Green + Solo",
        )
    )
    results.append(
        evaluate_bucket(
            df[(df["parent_color_super"] == "lightgreen") & (df["is_sparse"] == 0)],
            "Green + Dense History",
        )
    )
    results.append(
        evaluate_bucket(
            df[(df["parent_color_super"] == "lightgreen") & (df["is_sparse"] == 1)],
            "Green + Sparse History",
        )
    )

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    # --- New analysis: distance buckets vs regime ---

    # ---------------------------------------------------------
    # A. Discriminatory Power of Boundary Proximity (All Serviceability)
    # ---------------------------------------------------------

    # 1. Binning Boundary Distance
    # Default to 'INSIDE' for points within boundaries

    df["boundary_dist_bucket"] = "INSIDE"

    # Filter for points that have a distance to nearest boundary (meaning they are outside)
    outside_mask = df["nearest_boundary_dist_m"].notna()

    # Define bins for outside distances
    bins_dist = [0, 20, 40, 60, 100, np.inf]
    labels_dist = ["0-20", "20-40", "40-60", "60-100", "100+"]

    # Apply cuts only to outside points
    boundary_bins = pd.cut(
        df.loc[outside_mask, "nearest_boundary_dist_m"],
        bins=bins_dist,
        labels=labels_dist,
        right=True,
        include_lowest=True,
    )
    df.loc[outside_mask, "boundary_dist_bucket"] = pd.Series(
        boundary_bins, index=df.index[outside_mask]
    ).astype(str)

    # 2. Install Rate by Distance Bucket x Final Serviceability
    dist_fsb = (
        df.groupby(
            ["min_dist_bucket", "final_serviceability", "boundary_dist_bucket"],
            dropna=False,
        )
        .agg(total=("mobile", "count"), installs=("installed_decision", "sum"))
        .reset_index()
    )

    dist_fsb["install_rate"] = dist_fsb["installs"] / dist_fsb["total"]
    dist_fsb_path = os.path.join(
        REPORTS_DIR,
        "install_rate_by_min_dist_bucket_final_serviceability_distance_from_boundary.csv",
    )
    dist_fsb.to_csv(dist_fsb_path, index=False)

    # 3. Install Rate by Boundary Count x Final Serviceability
    dist_count = (
        df.groupby(
            ["min_dist_bucket", "final_serviceability", "nmbr_boundaries_within_100m"],
            dropna=False,
        )
        .agg(total=("mobile", "count"), installs=("installed_decision", "sum"))
        .reset_index()
    )
    dist_count["install_rate"] = dist_count["installs"] / dist_count["total"]

    path_boundary_count = os.path.join(
        REPORTS_DIR,
        "install_rate_by_min_dist_bucket_final_serviceability_boundary_count.csv",
    )
    dist_count.to_csv(path_boundary_count, index=False)

    # ---------------------------------------------------------
    # B. Existing Analysis (Regime & General Serviceability)
    # ---------------------------------------------------------

    # A.b) Install rate by distance bucket x final_serviceability
    dist_fs = (
        df.groupby(["min_dist_bucket", "final_serviceability"], dropna=False)
        .agg(total=("mobile", "count"), installs=("installed_decision", "sum"))
        .reset_index()
    )
    dist_fs["install_rate"] = dist_fs["installs"] / dist_fs["total"]
    dist_fs_path = os.path.join(
        REPORTS_DIR, "install_rate_by_min_dist_bucket_and_final_serviceability.csv"
    )
    dist_fs.to_csv(dist_fs_path, index=False)

    # B) Regime-level discrimination within each distance + final serviceability
    dist_regime = (
        df.groupby(
            ["min_dist_bucket", "geometry_regime", "final_serviceability"], dropna=False
        )
        .agg(total=("mobile", "count"), installs=("installed_decision", "sum"))
        .reset_index()
    )
    dist_regime["install_rate"] = dist_regime["installs"] / dist_regime["total"]
    dist_regime_path = os.path.join(
        REPORTS_DIR, "install_rate_by_min_dist_bucket_and_geometry_regime.csv"
    )
    dist_regime.to_csv(dist_regime_path, index=False)

    if simulate:
        run_declines_simulation(
            df_train, df_poly, df_bound, g1_start, g1_end, REPORTS_DIR
        )

    # Save detailed results for analysis
    report_path = os.path.join(REPORTS_DIR, "test_scores.csv")
    report_path_h5 = os.path.join(REPORTS_DIR, "test_scored.h5")
    df.to_csv(report_path, index=False)
    category_cols = df.select_dtypes(include=["category"]).columns
    if len(category_cols) > 0:
        df[category_cols] = df[category_cols].astype(str)
    df.to_hdf(report_path_h5, mode="w", key="df")

    print(f"\nDetailed scores saved to {report_path}")
    print(f"Distance x FS summary saved to {dist_fs_path}")
    print(f"Distance x FS x regime summary saved to {dist_regime_path}")
    # from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # y_true = df["installed_decision"]
    # y_pred = df["final_serviceability"]

    # acc = accuracy_score(y_true, y_pred)
    # cm = confusion_matrix(y_true, y_pred)
    # report = classification_report(y_true, y_pred)

    # print("Accuracy:", acc)
    # print("Confusion Matrix:\n", cm)
    # print("Classification Report:\n", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test scoring (playground)")
    parser.add_argument(
        "--simulate", action="store_true", help="Run declines simulation"
    )
    args = parser.parse_args()
    main(simulate=args.simulate)


# ===== FILE: ./steps/step3_simpulate.py =====

import os
import datetime as dt
from typing import cast

import numpy as np
import pandas as pd

import lib.config as config
from lib.data_fetch.get_data import get_g1_distance
from lib.compute import process as compute_process
from lib.geometry.geometric_features import calculate_adaptive_h


def run_declines_simulation(df_train, df_poly, df_bound, g1_start, g1_end, reports_dir):
    print("\n--- SIMULATION: DECLINES (G1) ---")
    df_geeone = get_g1_distance(g1_start, g1_end)
    if df_geeone.empty:
        print("Warning: G1 distance data is empty for this window; simulation skipped")
        return None

    df_declines = df_geeone[df_geeone["serviceable"] == 0].copy()
    if df_declines.empty:
        print("No declines found in G1 window; simulation skipped")
        return None

    dfus = compute_process(
        df_train,
        df_declines,
        df_poly,
        df_bound,
        lambda_decay=config.LAMBDA_DECAY,
        max_radius_m=int(config.MIN_DIST_CUTOFF_M),
    )
    if dfus is None or dfus.empty:
        print("Simulation scoring returned empty dataframe; skipping")
        return None

    bins = [-np.inf, 20, 40, 60, 100, np.inf]
    labels = ["0-20", "20-40", "40-60", "60-100", "100+"]
    dfus["min_dist_bucket"] = pd.cut(
        dfus["min_dist"], bins=bins, labels=labels, right=True, include_lowest=True
    )

    indeterminate_mask = (dfus["parent_color"] == "lightgreen") & (
        dfus["parent_installs"] <= config.INDETERMINATE_INSTALLS_CUTOFF
    )
    dfus["parent_color"] = np.where(
        indeterminate_mask, "indeterminate", dfus["parent_color"]
    )
    dfus["parent_color_super"] = np.where(
        dfus["parent_color"] == "indeterminate", "lightgreen", dfus["parent_color"]
    )


    dfus["gravity_score"] = np.where(
        dfus["predicted_field_hex"].isna(),
        -99,
        np.where((dfus["predicted_field_hex"] > config.FIELD_THRESHOLD) & (dfus['parent_total']>config.PARENT_TOTAL_THRESHOLD), 2,
            np.where(dfus["predicted_field_hex"] > config.FIELD_THRESHOLD,1,0)
            )
    )


    """
    OLD 1.1:
    dfus["gravity_score"] = np.where(
        dfus["predicted_field_hex"].isna(),
        -99,
        np.where(dfus["predicted_field_hex"] > config.FIELD_THRESHOLD, 1, 0),
    )
    
    """


    dfus["comp_gravity_score"] = np.where(
        dfus["contested_field"].isna(),
        -99,
        np.where(dfus["contested_field"] > config.CONTEST_FIELD_THRESHOLD, 1, 0),
    )

    """
    OLD 1.2:
    us_class_score_2 = (dfus["gravity_score"] == 1) & (dfus["comp_gravity_score"] == 1)
    us_class_score_1 = dfus["gravity_score"] == 1
    dfus["us_class_score"] = np.where(
        us_class_score_2,
        "A. Comp+Field",
        np.where(us_class_score_1, "B. Field", "C. Bad Field"),
    )

    declines_classifier_v1 = (dfus["parent_color_super"].isin(["lightgreen"])) & (
        dfus["us_class_score"].isin(["A. Comp+Field", "B. Field"])
    )
    declines_classifier_v2 = (dfus["parent_color_super"].isin(["orange"])) & (
        dfus["us_class_score"].isin(["A. Comp+Field"])
    )
    """


    us_class_score_2 = (dfus["gravity_score"] == 2) & (dfus["comp_gravity_score"] == 1)
    us_class_score_1 = dfus["gravity_score"] == 2
    us_class_score_01 = (dfus["gravity_score"]==1) & (dfus["comp_gravity_score"]==1)
    us_class_score_02 = dfus["gravity_score"]==1
    us_class_score_03 = dfus["comp_gravity_score"]==1

    dfus["us_class_score"] = np.where(
        us_class_score_2,
        "A. STRONG Comp+ STRONG Field",
        np.where(us_class_score_1, "B. STRONG Field",
            np.where(us_class_score_01, "C. STRONG Comp + WEAK Field",
                np.where(us_class_score_02, "D. WEAK FIELD",
                    np.where(us_class_score_03, "E. STRONG Comp", "F. Bad Field")
                    )
                )
            )
        )


    declines_classifier_v1 = (dfus["parent_color_super"].isin(["lightgreen"])) & (
        dfus["us_class_score"].isin(["A. STRONG Comp+ STRONG Field", "B. STRONG Field", "C. STRONG Comp + WEAK Field", "E. STRONG Comp"])
    )
    declines_classifier_v2 = (dfus["parent_color_super"].isin(["orange"])) & (
        dfus["us_class_score"].isin(["A. STRONG Comp+ STRONG Field", "B. STRONG Field"])
    )

    

    dfus["declines_classifier_new"] = np.where(
        declines_classifier_v1 | declines_classifier_v2, 1, 0
    )

    dfus["declines_serviceability"] = dfus["declines_classifier_new"]
    #dfus["declines_serviceability"] = np.where((dfus["declines_classifier_new"] == 1) & (dfus['parent_total']>config.PARENT_TOTAL_THRESHOLD), 1, 0)

    df_potential = (
        dfus.groupby(["min_dist_bucket", "declines_serviceability"])
        .agg(total=("mobile", "count"))
        .reset_index()
    )
    df_potential["potential_bookings"] = df_potential["total"] * 0.8

    potential_path = os.path.join(
        reports_dir, "potential_by_min_dist_bucket_and_final_declines.csv"
    )
    df_potential.to_csv(potential_path, index=False)
    print(f"Potential declines summary saved to {potential_path}")
    return potential_path


def main():
    print("--- STARTING DECLINES SIMULATION (PLAYGROUND) ---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, "artifacts")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    try:
        poly_path = os.path.join(artifacts_dir, "poly_stats_final.h5")
        bound_path = os.path.join(artifacts_dir, "partner_cluster_boundaries.h5")
        train_path = os.path.join(artifacts_dir, "train_data.h5")

        df_poly = cast(pd.DataFrame, pd.read_hdf(poly_path, "df"))
        df_bound = cast(pd.DataFrame, pd.read_hdf(bound_path, "df"))
        df_train = cast(pd.DataFrame, pd.read_hdf(train_path, "df"))

        if config.DEFINITE_DECISIONS == 1:
            df_train = cast(
                pd.DataFrame,
                df_train[
                    df_train["final_decision"].isin(["DECLINED", "INSTALLED"])
                ].copy(),
            )

        df_train["h"] = np.where(
            df_train["field_weight"] >= 0, config.H_INSTALL, config.H_DECLINE
        )

        if config.USE_ADAPTIVE_H:
            print(
                f"Computing Adaptive H (k={config.ADAPTIVE_H_NEIGHBOR_K}, min={config.ADAPTIVE_H_MIN}m, max={config.ADAPTIVE_H_MAX}m)..."
            )
            df_train = calculate_adaptive_h(cast(pd.DataFrame, df_train))

        print("Loaded Maps & Training Data.")
    except FileNotFoundError:
        print("Artifacts not found. Run step1_train_maps.py first.")
        return

    test_start = dt.date.fromisoformat(config.TEST_START_DATE)
    g1_start = (
        test_start - dt.timedelta(days=config.G1_LOOKBACK_EXTRA_DAYS)
    ).isoformat()
    g1_end = config.TEST_END_DATE

    run_declines_simulation(df_train, df_poly, df_bound, g1_start, g1_end, reports_dir)


if __name__ == "__main__":
    main()


# ===== FILE: ./lib/compute.py =====

from tqdm.auto import tqdm

from sklearn.neighbors import BallTree
from lib.geometry.distance import equiv_radius_m, haversine_m
import numpy as np
import pandas as pd

from lib.feature.hop_features import compute_hop_features
from pyproj import Transformer

import geopandas as gpd
from shapely.geometry import Point
from shapely import contains_xy

from functools import reduce
import operator

import lib.config as config
import lib.params as params

# GLOBAL CONFIG — NEVER CHANGE PER PARTNER
DEFAULT_LAMBDA = 0.005
best_w_neg = -2.0
min_decisions = 250


min_samples = 50
EARTH_RADIUS_M = 6371000  # metres (standard for most geospatial work)

COLOR_NUMERIC_MAP = {
    "lightgreen": 3,
    "orange": 2,
    "crimson": 1,
    "indeterminate": 2,
}



def compute_adaptive_gaussian_field(
    df_target: pd.DataFrame,
    df_source: pd.DataFrame,
    lambda_decay: float = DEFAULT_LAMBDA,
    max_radius_m: float = 15000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adaptive Gaussian KDE with exponential time decay, causal.
    Uses BallTree radius query → O(1) memory.
    Returns `kernel_sum` — the denominator Σ(k_i) — as per-target confidence weight.
    """
    if "decision_time" in df_target.columns:
        eval_times = df_target["decision_time"].values.astype("datetime64[ns]")
    else:
        eval_times = np.full(len(df_target), np.datetime64("now"))

    required_source = ["latitude", "longitude", "field_weight", "h", "decision_time"]
    df_source = df_source[required_source].copy()

    if len(df_source) == 0:
        df_target["predicted_field"] = np.nan
        df_target["kernel_sum"] = 0.0
        return df_target

    source_coords_rad = np.radians(df_source[["latitude", "longitude"]].values)
    tree = BallTree(source_coords_rad, metric="haversine")

    source_times = df_source["decision_time"].values.astype("datetime64[ns]")
    source_h = df_source["h"].values
    source_weights = df_source["field_weight"].values

    target_coords_rad = np.radians(df_target[["latitude", "longitude"]].values)
    radius_rad = max_radius_m / EARTH_RADIUS_M

    predicted = np.zeros(len(df_target))
    kernel_sums = np.zeros(len(df_target))

    iterator = (
        tqdm(
            enumerate(zip(target_coords_rad, eval_times)),
            total=len(df_target),
            desc=f"Gaussian field ({len(df_target)} targets)",
        )
        if verbose
        else enumerate(zip(target_coords_rad, eval_times))
    )

    for i, (target_coord, eval_time) in iterator:
        indices, dist_rad = tree.query_radius(
            target_coord.reshape(1, -1),
            r=radius_rad,
            return_distance=True,
            sort_results=False,
        )
        indices = indices[0]
        dist_m = dist_rad[0] * EARTH_RADIUS_M

        if len(indices) == 0:
            predicted[i] = np.nan
            kernel_sums[i] = 0.0
            continue

        neigh_times = source_times[indices]
        neigh_h = source_h[indices]
        neigh_w = source_weights[indices]

        ages_days = (eval_time - neigh_times) / np.timedelta64(1, "D")
        valid_mask = ~np.isnan(ages_days)
        past_mask = valid_mask & (ages_days >= 0)
        time_weights = np.zeros_like(ages_days, dtype=float)
        time_weights[past_mask] = np.exp(-lambda_decay * ages_days[past_mask])

        effective_weights = neigh_w * time_weights
        kernel = np.exp(-0.5 * (dist_m**2) / (neigh_h**2))

        weighted_sum = np.sum(kernel * effective_weights)
        norm = np.sum(kernel)

        predicted[i] = weighted_sum / (norm + 1e-12)
        kernel_sums[i] = norm

    df_target["predicted_field"] = predicted
    df_target["kernel_sum"] = kernel_sums
    return df_target





def compute_contested_metrics_engine(
    joined_gdf_4326: gpd.GeoDataFrame,
    df_target: pd.DataFrame,
    gdf_source_4326: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Computes contested overlap geometry (cached) + per-mobile field.

    REQUIRED COLUMNS
    ----------------
    joined_gdf_4326 : mobile, boundary_poly_keep (Polygon, EPSG:4326)
    df_target       : mobile, latitude, longitude, decision_time
    gdf_source_4326 : latitude, longitude, decision_time, field_weight, h, installed_decision

    RETURNS
    -------
    DataFrame indexed by mobile with contested metrics.
    """
    sidx = gdf_source_4326.sindex

    to_4326 = Transformer.from_crs("EPSG:7755", "EPSG:4326", always_xy=True)

    poly_cache = {}
    zonal_cache = {}
    results = []

    for mobile, grp in tqdm(
        joined_gdf_4326.groupby("mobile", sort=False),
        total=joined_gdf_4326["mobile"].nunique(),
    ):
        polys = [p for p in grp["boundary_poly_keep"].tolist() if p is not None]

        base = {
            "mobile": mobile,
            "n_overlapping_partners": len(polys),
            "contested_area_km2": np.nan,
            "contested_radius_m": np.nan,
            "contested_centroid_lat": np.nan,
            "contested_centroid_lon": np.nan,
            "contested_installs": np.nan,
            "contested_total": np.nan,
            "contested_se": np.nan,
            "contested_field": np.nan,
        }

        if len(polys) == 0:
            results.append(base)
            continue

        tr = df_target.loc[df_target["mobile"] == mobile].iloc[0]
        lat0, lon0 = float(tr["latitude"]), float(tr["longitude"])
        t0 = pd.to_datetime(tr["decision_time"])
        pt = Point(lon0, lat0)

        sig = tuple(sorted(p.wkb for p in polys))

        if sig not in poly_cache:
            cp = polys[0] if len(polys) == 1 else reduce(operator.and_, polys)

            if cp is None or cp.is_empty or cp.geom_type == "GeometryCollection":
                poly_cache[sig] = None
                zonal_cache[sig] = None
                results.append(base)
                continue
            else:
                if cp.geom_type == "MultiPolygon":
                    candidates = [g for g in cp.geoms if g.contains(pt) or g.intersects(pt)]
                    cp = max(candidates, key=lambda g: g.area) if candidates else max(cp.geoms, key=lambda g: g.area)

                poly_cache[sig] = cp

                cp_proj = gpd.GeoSeries([cp], crs="EPSG:4326").to_crs("EPSG:7755").iloc[0]
                area_m2 = float(cp_proj.area)
                area_km2 = area_m2 / 1e6
                radius_m = equiv_radius_m(area_m2)

                rp = cp_proj.representative_point()
                rp_lon, rp_lat = to_4326.transform(rp.x, rp.y)

                minx, miny, maxx, maxy = cp.bounds
                cand_idx = list(sidx.intersection((minx, miny, maxx, maxy)))

                if len(cand_idx) == 0:
                    inside = gdf_source_4326.iloc[[]]
                else:
                    cand = gdf_source_4326.iloc[cand_idx]
                    mask = contains_xy(cp, cand["longitude"].to_numpy(), cand["latitude"].to_numpy())
                    inside = cand.loc[mask]

                zonal_cache[sig] = {
                    "area_km2": area_km2,
                    "radius_m": radius_m,
                    "centroid_lat": rp_lat,
                    "centroid_lon": rp_lon,
                    "inside": inside,
                }

        if poly_cache[sig] is None:
            results.append(base)
            continue

        zonal = zonal_cache[sig]
        inside = zonal["inside"]

        n_total = len(inside)
        n_installs = int(inside["installed_decision"].sum()) if n_total > 0 else 0
        se = (n_installs / n_total) if n_total > 0 else np.nan

        base.update(
            {
                "contested_area_km2": round(zonal["area_km2"], 4),
                "contested_radius_m": round(zonal["radius_m"], 1),
                "contested_centroid_lat": zonal["centroid_lat"],
                "contested_centroid_lon": zonal["centroid_lon"],
                "contested_installs": n_installs,
                "contested_total": n_total,
                "contested_se": round(se, 4) if not np.isnan(se) else np.nan,
            }
        )

        if n_total > 0:
            lat = inside["latitude"].to_numpy()
            lon = inside["longitude"].to_numpy()

            dlat = np.radians(lat - lat0)
            dlon = np.radians(lon - lon0)
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(np.radians(lat0)) * np.cos(np.radians(lat)) * np.sin(dlon / 2) ** 2
            )
            d = 2 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            h = inside["h"].to_numpy(dtype=float)
            w = inside["field_weight"].to_numpy(dtype=float)

            dt_days = (
                t0 - pd.to_datetime(inside["decision_time"])
            ).dt.total_seconds().to_numpy() / 86400.0
            valid_mask = ~np.isnan(dt_days)
            past_mask = valid_mask & (dt_days >= 0)
            time_decay = np.zeros_like(dt_days, dtype=float)
            time_decay[past_mask] = np.exp(-config.LAMBDA_DECAY * dt_days[past_mask])

            eff_w = w * time_decay
            k = np.exp(-0.5 * (d**2) / (h**2 + 1e-12))
            base["contested_field"] = float(np.sum(k * eff_w) / (np.sum(k) + 1e-12))

        results.append(base)

    return pd.DataFrame(results)


def add_boundary_details_precise(df_target, df_bound, df_source):
    geometry_points = [
        Point(lon, lat)
        for lon, lat in zip(df_target["longitude"], df_target["latitude"])
    ]

    gdf_target = gpd.GeoDataFrame(df_target.copy(), geometry=geometry_points, crs="EPSG:4326")
    gdf_bound = gpd.GeoDataFrame(df_bound.copy(), geometry="boundary_poly", crs="EPSG:4326")
    gdf_bound["boundary_poly_keep"] = gdf_bound["boundary_poly"]

    joined_gdf_4326 = gpd.sjoin(
        gdf_target, gdf_bound, how="left", predicate="within"
    ).reset_index(drop=True)

    joined_gdf = joined_gdf_4326.to_crs(7755)

    poly_keep_proj = gpd.GeoSeries(joined_gdf["boundary_poly_keep"], crs="EPSG:4326").to_crs(7755)

    joined_gdf["dist_to_boundary_edge_m"] = poly_keep_proj.boundary.distance(joined_gdf.geometry)
    joined_gdf["dist_to_boundary_edge_m"] = joined_gdf["dist_to_boundary_edge_m"].where(
        joined_gdf["boundary_poly_keep"].notna(), np.nan
    )

    outside_mask = joined_gdf["boundary_poly_keep"].isna()

    joined_gdf["dist_to_nearest_boundary_m"] = np.nan
    joined_gdf["nmbr_boundaries_within_100m"] = np.nan

    if outside_mask.any():
        joined_gdf.loc[outside_mask, "nmbr_boundaries_within_100m"] = 0

        gdf_outside = joined_gdf.loc[outside_mask].copy()
        gdf_bound_proj = gdf_bound.to_crs(7755)

        try:
            bound_geom_col = gdf_bound_proj.geometry.name
            nearest = gpd.sjoin_nearest(
                gdf_outside[["mobile", "geometry"]],
                gdf_bound_proj[[bound_geom_col]],
                how="left",
                distance_col="dist_temp",
            )
            dist_map = nearest.groupby(level=0)["dist_temp"].min()
            joined_gdf.loc[dist_map.index, "dist_to_nearest_boundary_m"] = dist_map
        except AttributeError:
            pass

        buf = gdf_outside[["mobile", "geometry"]].copy()
        buf["geometry"] = buf.geometry.buffer(100)

        bound_geom_col = gdf_bound_proj.geometry.name
        within_100 = gpd.sjoin(buf, gdf_bound_proj[[bound_geom_col]], predicate="intersects")

        if not within_100.empty:
            count_map = within_100.groupby("mobile").size()
            joined_gdf.loc[outside_mask, "nmbr_boundaries_within_100m"] += (
                joined_gdf.loc[outside_mask, "mobile"].map(count_map).fillna(0)
            )

    df_joined = pd.DataFrame(joined_gdf.drop(columns="geometry"))

    df_joined["dist_to_cluster_center_m"] = haversine_m(
        df_joined["latitude"], df_joined["longitude"],
        df_joined["center_lat"], df_joined["center_lon"],
    )

    df = df_joined[params.one_to_many_cols].copy()

    df["near_edge"] = np.where(df["dist_to_boundary_edge_m"] < 0.7 * df["dist_to_cluster_center_m"], 1, 0)
    df["depth_score"] = df["dist_to_cluster_center_m"] - df["dist_to_boundary_edge_m"]
    df["is_solo_cluster"] = np.where(df["cluster_type"] == "p90_single_cluster", 1, 0)

    df_summary = (
        df.groupby(["mobile"])
        .agg(
            nmbr_overlap_clusters=("cluster_id", "count"),
            mean_dist_to_edge_m=("dist_to_boundary_edge_m", "mean"),
            near_edge_instances=("near_edge", "sum"),
            mean_dist_to_center_m=("dist_to_cluster_center_m", "mean"),
            total_area_boundaries=("area_km2", "sum"),
            n_hexes=("n_hexes", "sum"),
            nmbr_partners=("partner_id", "nunique"),
            worst_depth_score=("depth_score", "max"),
            any_near_edge=("near_edge", "max"),
            is_solo_cluster=("is_solo_cluster", "max"),
            nearest_boundary_dist_m=("dist_to_nearest_boundary_m", "min"),
            nmbr_boundaries_within_100m=("nmbr_boundaries_within_100m", "max"),
        )
        .reset_index()
    )

    if df_source is not None and len(df_source) > 0:
        gdf_source = gpd.GeoDataFrame(
            df_source.copy(),
            geometry=gpd.points_from_xy(df_source.longitude, df_source.latitude),
            crs="EPSG:4326",
        )

        if "installed_decision" not in gdf_source.columns:
            gdf_source["installed_decision"] = (gdf_source["final_decision"] == "INSTALLED").astype(int)

        joined_gdf_for_contested = joined_gdf_4326[["mobile", "boundary_poly_keep"]].copy()

        df_contested = compute_contested_metrics_engine(
            joined_gdf_4326=joined_gdf_for_contested,
            df_target=df_target,
            gdf_source_4326=gdf_source,
        )

        df_summary = df_summary.merge(df_contested, on="mobile", how="left")

    return df_summary


def get_parent_hexagon(df_target: pd.DataFrame, df_poly: pd.DataFrame) -> pd.DataFrame:
    """
    CHANGED: No longer picks the single hex with highest SE.
    Keeps ALL covering hex rows, applies indeterminate check per partner,
    encodes color numerically, shrinks SE, then computes evidence-weighted consensus
    across all covering hexes per mobile.

    Geometric features still come from a single hex picked by max-total (most evidence).
    """
    df_poly["center_lat"] = pd.to_numeric(
        df_poly["poly"].apply(lambda p: p.centroid.y if p is not None else np.nan),
        errors="coerce",
    )
    df_poly["center_lon"] = pd.to_numeric(
        df_poly["poly"].apply(lambda p: p.centroid.x if p is not None else np.nan),
        errors="coerce",
    )

    geometry_points = [Point(lon, lat) for lon, lat in zip(df_target["longitude"], df_target["latitude"])]
    gdf_target = gpd.GeoDataFrame(df_target, geometry=geometry_points, crs="EPSG:4326")
    gdf_hex = gpd.GeoDataFrame(df_poly, geometry="poly", crs="EPSG:4326")
    gdf_hex["poly_keep"] = gdf_hex["poly"]

    joined = gpd.sjoin(gdf_target, gdf_hex, how="left", predicate="within")

    joined["dist_to_cluster_center_point_hex"] = haversine_m(
        joined["latitude"], joined["longitude"],
        joined["center_lat"], joined["center_lon"],
    )

    # A.1: Indeterminate check per partner
    joined["color_adj"] = joined["color"].copy()
    indeterminate_mask = (joined["color"] == "lightgreen") & (
        joined["installs"] <= config.INDETERMINATE_INSTALLS_CUTOFF
    )
    joined.loc[indeterminate_mask, "color_adj"] = "indeterminate"

    # A.2: Encode color numerically
    joined["color_numeric"] = joined["color_adj"].map(COLOR_NUMERIC_MAP).fillna(0)

    # A.3: Asymmetric credibility weight
    ratio = np.maximum(config.MIN_SHRINKAGE_RATIO, joined["total"] / (joined["total"] + config.SHRINKAGE_K))
    joined["se_shrunk"] = np.where(
        joined["se"] >= 0,
        joined["se"] * ratio,
        joined["se"] / ratio,
    )

    # A.4: Evidence-weighted consensus
    joined["_w_color"] = joined["color_numeric"] * joined["total"]

    consensus = (
        joined.groupby("mobile")
        .agg(
            parent_se=("se_shrunk", "sum"),
            _sum_w_color=("_w_color", "sum"),
            parent_total=("total", "sum"),
            parent_installs=("installs", "sum"),
            parent_declines=("declines", "sum"),
            n_covering_partners=("partner_id", "nunique"),
        )
        .reset_index()
    )

    consensus["parent_color_numeric"] = (
        consensus["_sum_w_color"] / consensus["parent_total"].replace(0, np.nan)
    )
    consensus.drop(columns=["_sum_w_color"], inplace=True)

    def numeric_to_color(v):
        if pd.isna(v): return np.nan
        if v >= 2.5: return "lightgreen"
        if v >= 1.5: return "orange"
        return "crimson"

    consensus["parent_color"] = consensus["parent_color_numeric"].apply(numeric_to_color)

    # Geometric features: pick hex with max total (most evidence)
    joined_valid = joined.dropna(subset=["partner_id"]).copy()
    if len(joined_valid) > 0:
        joined_valid = joined_valid.sort_values(["mobile", "total"], ascending=[True, False])
        best = joined_valid.drop_duplicates("mobile", keep="first")
    else:
        best = joined.drop_duplicates("mobile", keep="first")

    best = best.to_crs(7755)
    poly_keep_proj = gpd.GeoSeries(best["poly_keep"], crs="EPSG:4326").to_crs(7755)

    best["dist_to_boundary_edge_point_hex"] = poly_keep_proj.boundary.distance(best.geometry)
    best["dist_to_boundary_edge_point_hex"] = best["dist_to_boundary_edge_point_hex"].where(
        best["poly_keep"].notna(), np.nan
    )
    best["near_edge_point_hex"] = np.where(
        best["dist_to_boundary_edge_point_hex"] < 0.7 * best["dist_to_cluster_center_point_hex"], 1, 0
    )
    best["depth_score_point_hex"] = (
        best["dist_to_cluster_center_point_hex"] - best["dist_to_boundary_edge_point_hex"]
    )

    cols_to_use = [col for col in params.hex_cols if col in best.columns] + ["mobile"]
    parent_data = pd.DataFrame(best[cols_to_use])

    parent_data = parent_data.rename(columns={
        "partner_id": "partner_id",
        "poly_id": "poly_id",
        "is_overlap": "parent_overlap",
        "distance_from_boundary_m": "parent_hex_dist_to_foreign_m",
        "distance_to_own_boundary_m": "parent_hex_dist_to_own_m",
        "rank": "parent_rank",
    })
    parent_data = pd.merge(parent_data, consensus, on="mobile", how="left")

    return pd.merge(df_target, parent_data, how="left", on="mobile")


def process(
    df_source,
    df_target,
    df_poly,
    df_bound,
    lambda_decay=DEFAULT_LAMBDA,
    max_radius_m=500,
):
    try:
        df_poly["centroid_lat"] = pd.to_numeric(
            df_poly["poly"].apply(lambda p: p.centroid.y if p else np.nan), errors="coerce"
        )
        df_poly["centroid_lon"] = pd.to_numeric(
            df_poly["poly"].apply(lambda p: p.centroid.x if p else np.nan), errors="coerce"
        )

        df_target_boundary = add_boundary_details_precise(df_target, df_bound, df_source)
        df_target_new = pd.merge(df_target, df_target_boundary, how="left", on="mobile")

        _df_poly_snapshot = df_poly.copy(deep=True)

        df_target_with_hex = get_parent_hexagon(df_target_new, df_poly)

        print("[HOP FEATURES] Computing 3-hop neighbor SE aggregates...")
        df_hop = compute_hop_features(_df_poly_snapshot, n_hops=3)
        print(f"[HOP FEATURES] {len(df_hop)} hex rows with hop features")

        df_target_with_hex = df_target_with_hex.merge(
            df_hop, on=["partner_id", "poly_id"], how="left"
        )
        print(f"[HOP FEATURES] Merged. Columns added: {[c for c in df_hop.columns if c not in ('partner_id','poly_id')]}")

        unique_polys_y = (
            df_target_with_hex.groupby(["partner_id", "poly_id"])["mobile"].nunique().reset_index()
        )

        df_poly_sub_y = pd.merge(df_poly, unique_polys_y, how="inner", on=["partner_id", "poly_id"])

        gdf_poly_parents = gpd.GeoDataFrame(df_poly_sub_y, geometry="poly", crs="EPSG:4326")
        gdf_poly_parents.rename(columns={"partner_id": "poly_partner_id"}, inplace=True)

        gdf_source_points = gpd.GeoDataFrame(
            df_source,
            geometry=gpd.points_from_xy(df_source.longitude, df_source.latitude),
            crs="EPSG:4326",
        )

        df_with_parent = df_target_with_hex.dropna(subset=["partner_id", "poly_id"]).copy()

        if len(df_with_parent) == 0:
            print("No targets have a parent hexagon → predicted_field_hex = NaN")
            return pd.DataFrame()

        joined_src = gpd.sjoin(gdf_source_points, gdf_poly_parents, how="inner", predicate="within")
        joined_src = joined_src[joined_src["partner_id"] == joined_src["poly_partner_id"]].copy()

        df_source_in_hex = joined_src.drop(columns=["geometry", "index_right", "poly_partner_id"])

        dup_check = df_source_in_hex.duplicated(subset=df_source_in_hex.index.name, keep=False).sum()
        if dup_check > 0:
            raise ValueError(
                f"Invariant broken: {dup_check // 2} source rows in multiple polys of same partner. Fix hex clustering."
            )

        results = []
        hex_groups = df_with_parent.groupby(["partner_id", "poly_id"])
        for (pid, polyid), target_group in tqdm(hex_groups, total=hex_groups.ngroups, desc="Hexagons processed"):
            src_local = df_source_in_hex[
                (df_source_in_hex["partner_id"] == pid) & (df_source_in_hex["poly_id"] == polyid)
            ]
            target_group = target_group[["mobile", "latitude", "longitude", "decision_time"]].copy()

            if len(src_local) == 0:
                target_group["predicted_field_hex"] = np.nan
            else:
                field_df = compute_adaptive_gaussian_field(
                    df_target=target_group,
                    df_source=src_local,
                    lambda_decay=lambda_decay,
                    max_radius_m=max_radius_m,
                    verbose=False,
                )
                target_group["predicted_field_hex"] = field_df["predicted_field"]

            results.append(target_group[["mobile", "predicted_field_hex"]])

        hex_results = pd.concat(results, ignore_index=True)

        # ==============================================================
        # COMBINED FIELD ACROSS ALL OVERLAPPING PARTNER HEXAGONS
        # ==============================================================
        agg_all_hex = None

        try:
            print("[ALL-HEX FIELD] Starting combined field computation across all overlapping hexagons...")
            print(f"[ALL-HEX FIELD] _df_poly_snapshot columns: {_df_poly_snapshot.columns.tolist()}")
            print(f"[ALL-HEX FIELD] _df_poly_snapshot shape: {_df_poly_snapshot.shape}")
            print(f"[ALL-HEX FIELD] 'partner_id' in snapshot: {'partner_id' in _df_poly_snapshot.columns}")
            print(f"[ALL-HEX FIELD] 'poly' in snapshot: {'poly' in _df_poly_snapshot.columns}")

            _df_poly_copy = _df_poly_snapshot.copy()
            _df_poly_copy.rename(columns={"partner_id": "all_poly_partner_id"}, inplace=True)

            gdf_all_polys = gpd.GeoDataFrame(_df_poly_copy, geometry="poly", crs="EPSG:4326")

            gdf_target_pts = gpd.GeoDataFrame(
                df_target[["mobile", "latitude", "longitude", "decision_time"]].copy(),
                geometry=gpd.points_from_xy(df_target.longitude, df_target.latitude),
                crs="EPSG:4326",
            )

            all_joined = gpd.sjoin(gdf_target_pts, gdf_all_polys, how="inner", predicate="within")
            print(f"[ALL-HEX FIELD] Targets × all hexagons sjoin: {len(all_joined)} rows")

            if len(all_joined) == 0:
                raise ValueError("Empty sjoin — skipping all-hex field")

            _df_poly_copy_src = _df_poly_snapshot.copy()
            _df_poly_copy_src.rename(columns={"partner_id": "all_poly_partner_id"}, inplace=True)

            gdf_all_polys_for_src = gpd.GeoDataFrame(_df_poly_copy_src, geometry="poly", crs="EPSG:4326")

            all_joined_src = gpd.sjoin(gdf_source_points, gdf_all_polys_for_src, how="inner", predicate="within")
            print(f"[ALL-HEX FIELD] Sources × all hexagons sjoin: {len(all_joined_src)} rows")

            all_joined_src = all_joined_src[
                all_joined_src["partner_id"] == all_joined_src["all_poly_partner_id"]
            ].copy()
            print(f"[ALL-HEX FIELD] After partner match filter: {len(all_joined_src)} source rows")

            df_all_source_in_hex = all_joined_src.drop(
                columns=["geometry", "index_right", "all_poly_partner_id"], errors="ignore"
            )

            all_hex_results = []
            all_hex_groups = all_joined.groupby(["all_poly_partner_id", "poly_id"])
            print(f"[ALL-HEX FIELD] Processing {all_hex_groups.ngroups} hex groups...")

            for (pid, polyid), tgt_grp in tqdm(all_hex_groups, total=all_hex_groups.ngroups, desc="All-hex combined field"):
                src_local = df_all_source_in_hex[
                    (df_all_source_in_hex["partner_id"] == pid)
                    & (df_all_source_in_hex["poly_id"] == polyid)
                ]
                tgt_sub = tgt_grp[["mobile", "latitude", "longitude", "decision_time"]].copy()

                if len(src_local) == 0:
                    tgt_sub["_field_all"] = np.nan
                    tgt_sub["_n_sources"] = 0
                    tgt_sub["_kernel_sum"] = 0.0
                else:
                    field_df = compute_adaptive_gaussian_field(
                        df_target=tgt_sub,
                        df_source=src_local,
                        lambda_decay=lambda_decay,
                        max_radius_m=max_radius_m,
                        verbose=False,
                    )
                    tgt_sub["_field_all"] = field_df["predicted_field"]
                    tgt_sub["_n_sources"] = len(src_local)
                    tgt_sub["_kernel_sum"] = field_df["kernel_sum"]

                tgt_sub["_hex_partner"] = pid
                tgt_sub["_hex_poly"] = polyid
                all_hex_results.append(
                    tgt_sub[["mobile", "_hex_partner", "_hex_poly", "_field_all", "_n_sources", "_kernel_sum"]]
                )

            if all_hex_results:
                df_all_hex = pd.concat(all_hex_results, ignore_index=True)
                print(f"[ALL-HEX FIELD] Concatenated {len(df_all_hex)} field rows across all hexes")

                df_valid = df_all_hex.dropna(subset=["_field_all"]).copy()
                df_valid["_weighted_field"] = df_valid["_field_all"] * df_valid["_n_sources"]
                df_valid["_ks_weighted_field"] = df_valid["_field_all"] * df_valid["_kernel_sum"]

                agg_all_hex = (
                    df_valid.groupby("mobile")
                    .agg(
                        _sum_weighted_field=("_weighted_field", "sum"),
                        _sum_sources=("_n_sources", "sum"),
                        _sum_ks_weighted_field=("_ks_weighted_field", "sum"),
                        _sum_kernel_sums=("_kernel_sum", "sum"),
                        predicted_field_hex_all_min=("_field_all", "min"),
                        predicted_field_hex_all_max=("_field_all", "max"),
                        predicted_field_hex_all_std=("_field_all", "std"),
                        n_overlapping_hexes_field=("_field_all", "count"),
                    )
                    .reset_index()
                )

                agg_all_hex["predicted_field_hex_all_wmean"] = (
                    agg_all_hex["_sum_weighted_field"] / agg_all_hex["_sum_sources"].replace(0, np.nan)
                )
                agg_all_hex["predicted_field_hex_all_kswmean"] = (
                    agg_all_hex["_sum_ks_weighted_field"] / agg_all_hex["_sum_kernel_sums"].replace(0, np.nan)
                )

                unweighted = (
                    df_valid.groupby("mobile")["_field_all"]
                    .mean()
                    .rename("predicted_field_hex_all_mean")
                    .reset_index()
                )
                agg_all_hex = agg_all_hex.merge(unweighted, on="mobile", how="left")
                agg_all_hex.rename(columns={"_sum_sources": "total_sources_all_hexes"}, inplace=True)
                agg_all_hex.drop(
                    columns=["_sum_weighted_field", "_sum_ks_weighted_field", "_sum_kernel_sums"],
                    inplace=True,
                )

                print(f"[ALL-HEX FIELD] SUCCESS — {len(agg_all_hex)} mobiles with combined field")
            else:
                print("[ALL-HEX FIELD] WARNING: no hex results produced")
                agg_all_hex = None

        except Exception as e_allhex:
            print(f"[ALL-HEX FIELD] ERROR (non-fatal, existing columns unaffected): {e_allhex}")
            import traceback
            traceback.print_exc()
            agg_all_hex = None

        df_target_final = pd.merge(df_target_with_hex, hex_results, on="mobile", how="left")

        if agg_all_hex is not None and len(agg_all_hex) > 0:
            df_target_final = pd.merge(df_target_final, agg_all_hex, on="mobile", how="left")
        else:
            for col in [
                "predicted_field_hex_all_wmean", "predicted_field_hex_all_kswmean",
                "predicted_field_hex_all_mean", "predicted_field_hex_all_min",
                "predicted_field_hex_all_max", "predicted_field_hex_all_std",
                "n_overlapping_hexes_field", "total_sources_all_hexes",
            ]:
                df_target_final[col] = np.nan
            print("[ALL-HEX FIELD] Columns added as NaN (computation failed or empty)")

        return df_target_final

    except Exception as e:
        print(f"ERROR IN PROCESSING OF HEXAGONS AND BOUNDARIES: {e}")
        return pd.DataFrame()

# ===== FILE: ./lib/config.py =====

# EXPERIMENT CONFIGURATION
# Modify these values to run different experiments

# 1. Hexagon Generation

# DECIDING BAD AND MID SE RANGES based on portfolio deciles:
BAD_SE = [1, 2, 3]
MID_SE = [4, 5, 6, 7]

# The list of sizes (in km) to test for finding the best separation
HEX_GRID_SIZES = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.25]
DEFAULT_HEX_SIZE = 0.25
# How far out to tile hexes around the partner center (km)
HEX_TILING_RADIUS_KM = 3.0

# 2. Desirability Field Physics
# Adaptive H (Influence Radius)
USE_ADAPTIVE_H = True
ADAPTIVE_H_MIN = 20.0
ADAPTIVE_H_MAX = 150.0
ADAPTIVE_H_NEIGHBOR_K = 3

# CONSTANT 'h' - The impact radius (meters) where influence drops to ~60%
H_INSTALL = 60.0  # Impact radius for positive decisions
H_DECLINE = 60.0  # Impact radius for negative decisions

# Weights for different decision types
WEIGHT_INSTALL = 1.0
WEIGHT_DECLINE = -1.5  # Stronger negative signal

# Time Decay (Lambda)
# w_t = w_0 * exp(-lambda * days_old)
LAMBDA_DECAY = 0.005

# 3. Geometric Features
# Radius to search for local geometric patterns (gullies, clusters) around a lead
GEOM_SEARCH_RADIUS_M = 100.0
# If 1, compute geometry using only installed historical points
GEOM_INSTALL_FILTER = 1
# Regime thresholds for geometry-based context
GEOM_DENSE_THRESHOLD = 0.60
GEOM_GULLY_THRESHOLD = 0.70
GEOM_SPARSE_THRESHOLD = 0.40

# 4. Boundary & Competition
# Search radius to find nearby clusters (degrees)
COMPETITION_SEARCH_RADIUS_DEG = 0.027  # ~3km

# Toggle boundary filtering in lib_find_boundary.py: use p30 filteration to exclude hexagons less than p30 installs
ENABLE_BOUNDARY_FILTER = 1

# 5. Scoring Logic
MIN_DIST_CUTOFF_M = 500.0


# 6. Scoring Thresholds
FIELD_THRESHOLD = -0.1
CONTEST_FIELD_THRESHOLD = -0.1
INDETERMINATE_INSTALLS_CUTOFF = 4
SPARSE_DENSITY_THRESHOLD = 0.5
DEPTH_SCORE_THRESHOLD = 200
PARENT_TOTAL_THRESHOLD = 10



# 7. Fixed Date Windows (inclusive)
# Playground uses fixed calendar windows for reproducibility.
TRAIN_START_DATE = "2024-10-20"
TRAIN_END_DATE = "2025-10-19"
# TRAIN_START_DATE = "2025-09-15"
# TRAIN_END_DATE = "2025-12-15"
TEST_START_DATE = "2025-10-20"
TEST_END_DATE = "2025-11-09"

# G1 logs are allowed an earlier lookback than test start.
G1_LOOKBACK_EXTRA_DAYS = 15

# 8. TOGGLING BETWEEN USING ONLY DEFINITE DECISIONS FOR SCORING OR NOT:
DEFINITE_DECISIONS = 0

SHRINKAGE_K = 4
MIN_SHRINKAGE_RATIO = 0.45

EARTH_RADIUS_KILOMETER = 6371
EARTH_RADIUS_METER = 6371000
METERS_PER_DEG_LAT = 111320.0


# ===== FILE: ./lib/params.py =====

one_to_many_cols = [
    "mobile", "partner_id", "cluster_id", "center_lat", "center_lon",
    "total_installs", "total_obs", "n_hexes", "area_km2",
    "dist_to_boundary_edge_m", "dist_to_cluster_center_m", "cluster_type",
    "dist_to_nearest_boundary_m", "nmbr_boundaries_within_100m",
]

hex_cols = [
    "partner_id", "poly_id", "local_field_x1000", "is_overlap",
    "distance_from_boundary_m", "distance_to_own_boundary_m", "rank", "best_size",
    "dist_to_boundary_edge_point_hex", "dist_to_cluster_center_point_hex",
    "near_edge_point_hex", "depth_score_point_hex",
]

# ===== FILE: ./lib/test.py =====

# ~/Apps/maanasbot/portfolio/genie_ml/matchmaking/self_learning/revamp/new_code/test.py
# FINAL, CORRECT & PRODUCTION-READY VERSION — December 2025

import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.strtree import STRtree


def get_overlap(search_radius_deg: float = 0.027):
    print("Loading data...")
    df_hex = pd.read_hdf("poly_stats.h5", "df")
    df_bound = pd.read_hdf("partner_cluster_boundaries.h5", "df")

    print(f"Loaded {len(df_hex):,} hexagons and {len(df_bound):,} cluster boundaries.")

    # =============================================================================
    # Part 1 + 2 COMBINED: Distance to FOREIGN and OWN territories (symmetric, clean, fast)
    # =============================================================================
    print("\nComputing distances to ALL cluster boundaries (own + foreign)...")

    bound_polys = df_bound["boundary_poly"].tolist()
    bound_pids = df_bound["partner_id"].tolist()
    tree = STRtree(bound_polys)

    # NEW: 3 km search radius in degrees (~0.027° at equator, safe for all India)
    SEARCH_RADIUS_DEG = search_radius_deg  # from config to allow tuning

    # We'll collect results for every hex
    results = []

    for _, row in tqdm(
        df_hex.iterrows(), total=len(df_hex), desc="Distance to nearest clusters"
    ):
        hex_poly = row["poly"]
        pid = row["partner_id"]
        lat = hex_poly.centroid.y

        # MODIFIED: Expand search area by 3 km so we never miss a nearby own cluster
        search_poly = hex_poly.buffer(SEARCH_RADIUS_DEG)
        candidate_idxs = tree.query(search_poly)

        # candidate_idxs = tree.query(hex_poly)
        if len(candidate_idxs) == 0:
            results.append(
                {
                    "is_overlap": False,
                    "distance_from_boundary_deg": np.nan,
                    "distance_from_boundary_m": np.nan,
                    "distance_to_own_boundary_deg": np.nan,
                    "distance_to_own_boundary_m": np.nan,
                }
            )
            continue

        own_dists = []
        foreign_dists = []

        for i in candidate_idxs:
            dist_deg = hex_poly.distance(bound_polys[i])
            if bound_pids[i] == pid:
                own_dists.append(dist_deg)
            else:
                foreign_dists.append(dist_deg)

        # Foreign: overlap + min distance
        foreign_min = min(foreign_dists) if foreign_dists else np.nan
        # Own: min distance to any of own clusters
        own_min = min(own_dists) if own_dists else np.nan

        # Convert to metres using hex latitude
        to_metres = (
            lambda d: d * 111320 * np.cos(np.radians(lat)) if not pd.isna(d) else np.nan
        )

        results.append(
            {
                "is_overlap": foreign_min == 0 if not pd.isna(foreign_min) else False,
                "distance_from_boundary_deg": foreign_min,
                "distance_from_boundary_m": round(to_metres(foreign_min), 1),
                "distance_to_own_boundary_deg": own_min,
                "distance_to_own_boundary_m": round(to_metres(own_min), 1)
                if own_min is not None
                else np.nan,
            }
        )

    # Attach all results
    result_df = pd.DataFrame(results)
    df_hex = pd.concat([df_hex.reset_index(drop=True), result_df], axis=1)

    # Save intermediate
    for col in df_hex.select_dtypes(include="string").columns:
        df_hex[col] = df_hex[col].astype("object")
    df_hex.to_hdf(
        "poly_stats_updated.h5", key="df", mode="w", complevel=9, complib="blosc"
    )
    print(f"Distance computation complete:")
    print(f"   → {df_hex['is_overlap'].sum():,} hexes overlap foreign territory")
    print(
        f"   → {df_hex['distance_to_own_boundary_m'].notna().sum():,} hexes have own territory nearby"
    )

    # =============================================================================
    # Part 3: Rank crimson ranking — only on hexes colored crimson
    # =============================================================================
    print("\nRanking crimson hexes by proximity to own territory...")
    crimson_hexes = df_hex[df_hex["color"] == "crimson"].copy()

    if len(crimson_hexes) == 0:
        print("No crimson hexes found.")
        df_crimson = pd.DataFrame()
    else:
        df_crimson = crimson_hexes[
            ["partner_id", "poly_id", "se", "distance_to_own_boundary_m"]
        ].copy()
        df_crimson["rank"] = df_crimson.groupby("partner_id")[
            "distance_to_own_boundary_m"
        ].rank(method="min", ascending=True)
        df_crimson = df_crimson.sort_values(["partner_id", "rank"])
        for col in df_crimson.select_dtypes(include="string").columns:
            df_crimson[col] = df_crimson[col].astype("object")
        df_crimson.to_hdf(
            "crimson_ranks.h5", key="df", mode="w", complevel=9, complib="blosc"
        )
        df_crimson.to_csv("crimson_ranks.csv", index=False)
        print(f"Ranked {len(df_crimson):,} crimson hexes.")

    # =============================================================================
    # Part 4: Final dataset — NO redundant conversion needed
    # =============================================================================
    print("\nFinalizing dataset...")

    crimson_merge = df_crimson[
        ["partner_id", "poly_id", "distance_to_own_boundary_m", "rank"]
    ].copy()

    # Final merge
    df_final = df_hex.merge(
        crimson_merge, on=["partner_id", "poly_id"], how="left", suffixes=("", "_dup")
    )

    # Remove any duplicate columns from bad merge
    df_final = df_final.loc[:, ~df_final.columns.str.endswith("_dup")]

    # Final column order
    final_columns = [
        "partner_id",
        "poly_id",
        "best_size",
        "poly",
        "se",
        "installs",
        "declines",
        "total",
        "color",
        "is_overlap",
        "distance_from_boundary_m",
        "distance_to_own_boundary_m",
        "rank",
    ]
    df_final = df_final[[c for c in final_columns if c in df_final.columns]]

    # Save
    for col in df_final.select_dtypes(include="string").columns:
        df_final[col] = df_final[col].astype("object")
    df_final.to_hdf(
        "poly_stats_final.h5", key="df", mode="w", complevel=9, complib="blosc"
    )

    return df_final


# get_overlap()


# ===== FILE: ./lib/data_fetch/get_data.py =====

from typing import Optional

import pandas as pd
import numpy as np

from lib.data_fetch.wiom_data import WiomData


_snowflake_client: Optional[WiomData] = None


def _get_snowflake_client() -> WiomData:
    global _snowflake_client
    if _snowflake_client is None:
        _snowflake_client = WiomData("snowflake")
    return _snowflake_client


def _query_snowflake_df(
    sql: str, *, cache_file: Optional[str] = None, cache_h: int = 1
) -> pd.DataFrame:
    client = _get_snowflake_client()
    if cache_file is None:
        return client.query(sql, cache_h=cache_h)
    return client.query(sql, cache_file=cache_file, cache_h=cache_h)


def standardise_decisions(df):
    # Safety check for empty df
    if df.empty:
        return df

    # Calculate time to decide in minutes
    df["time_to_decide"] = (
        df["last_event_time"] - df["first_event_time"]
    ).dt.total_seconds() / 60

    # Calculate quantile thresholds on the specific slice of Interest->Decline
    mask_interest_decline = (df["first_event"] == "INTERESTED") & (
        df["last_event"] == "DECLINED"
    )

    # If no data for quantiles, return original (or handle gracefully)
    if not mask_interest_decline.any():
        print(
            "Warning: No INTERESTED->DECLINED events found for standardization thresholds."
        )
        return df

    dec_quantiles = (
        df[mask_interest_decline]["time_to_decide"]
        .quantile([i / 10 for i in range(0, 11)])
        .round(2)
        .reset_index()
    )

    # p40 (40th percentile in the list; quantile=0.4) used as cutoff
    try:
        p_40 = dec_quantiles.loc[dec_quantiles["index"] == 0.4, "time_to_decide"].iloc[
            0
        ]
    except IndexError:
        p_40 = 0  # Fallback

    print(f"Standardization Cutoff (p40): {p_40} minutes")

    # Apply Logic
    # 1. INSTALLED remains INSTALLED
    # 2. Fast Decline (<= p40) -> DECLINED
    # 3. Slow Decline (> p40) -> INDETERMINATE
    # 4. HANGING (Interested/Called) -> HANGING

    df["final_decision"] = np.where(
        df["final_decision"] == "INSTALLED",
        df["final_decision"],
        np.where(
            mask_interest_decline & (df["time_to_decide"] <= p_40),
            "DECLINED",
            np.where(
                mask_interest_decline & (df["time_to_decide"] > p_40),
                "INDETERMINATE",
                np.where(
                    df["final_decision"].isin(["INTERESTED", "CALLED"]),
                    "HANGING",
                    df["final_decision"],
                ),
            ),
        ),
    )

    # Filter to only valid decision types (Removing RNR, etc from Training Data)
    valid_types = ["DECLINED", "INSTALLED", "INDETERMINATE", "HANGING"]
    df_clean = df[df["final_decision"].isin(valid_types)].copy()

    print("STANDARDISED DECISIONS (TRAIN):")
    print(df_clean["final_decision"].value_counts())

    return df_clean


def get_g1_distance(start_dt: str, end_dt: str) -> pd.DataFrame:
    """
    Pulls latest G1 serviceability log per mobile in the given date range.
    Returns columns: mobile, decision_time, min_dist, dist_rng, latitude, longitude,
    g1_is_bdo_lead, zone_alias, serviceable (as per log), plus typed fields.
    """
    print(f"GETTING G1 DECLINES FROM {start_dt} TO {end_dt}")

    query = f"""
        select mobile, created_at as decision_time, min_dist, CASE
        WHEN min_dist <= 5 THEN 'A. 0-5m'
        WHEN min_dist <= 10 THEN 'B. 5-10m'
        WHEN min_dist <= 15 THEN 'C. 10-15m'
        WHEN min_dist <= 20 THEN 'D. 15-20m'
        WHEN min_dist <= 25 THEN 'E. 20-25m'
        WHEN min_dist <= 30 THEN 'F. 25-30m'
        WHEN min_dist <= 35 THEN 'G. 30-35m'
        WHEN min_dist <= 40 THEN 'H. 35-40m'
        WHEN min_dist <= 45 THEN 'I. 40-45m'
        WHEN min_dist <= 50 THEN 'J. 45-50m'
        WHEN min_dist <= 55 THEN 'K. 50-55m'
        WHEN min_dist <= 60 THEN 'L. 55-60m'
        WHEN min_dist <= 65 THEN 'M. 60-65m'
        WHEN min_dist <= 70 THEN 'N. 65-70m'
        WHEN min_dist <= 75 THEN 'O. 70-75m'
        WHEN min_dist <= 80 THEN 'P. 75-80m'
        WHEN min_dist <= 85 THEN 'Q. 80-85m'
        WHEN min_dist <= 90 THEN 'R. 85-90m'
        WHEN min_dist <= 95 THEN 'S. 90-95m'
        WHEN min_dist <= 100 THEN 'T. 95-100m'
        ELSE 'U. >100m'
        END AS dist_rng, latitude, longitude, g1_is_bdo_lead, zone_alias, serviceable

        from
        (
            select *
            from
            (
                select *, row_number() over(partition by mobile order by created_at desc) as row_cnt
                from
                (
                    select *, parse_json(response):extra_data.nearest_coordinate.distance as min_dist,
                    PARSE_JSON(response):"serviceable"::boolean AS serviceable,
                    PARSE_JSON(response):"extra_data":"nearest_coordinate":"zone_alias"::string AS zone_alias,
                    PARSE_JSON(response):"genie_downstream_context":"g1_is_bdo_lead"::boolean AS g1_is_bdo_lead,
                    SPLIT(PARSE_JSON(response):genie_downstream_context.g1_lead_coords::string,',')[0]::float AS latitude,
                    SPLIT(PARSE_JSON(response):genie_downstream_context.g1_lead_coords::string,',')[1]::float AS longitude

                    from
                    (
                        select *
                        from prod_db.mysql_rds_genie_genie1.t_serviceability_logs where cast(created_at as date) between '{start_dt}' and '{end_dt}'
                    ) a
                ) a
            ) a
            where row_cnt = 1
        ) a
    """
    try:
        df = _query_snowflake_df(query)
        if df.empty:
            print("NO DATA FOUND OF G1 DECLINES")
            return pd.DataFrame()
        df.columns = df.columns.str.lower()
        df["decision_time"] = pd.to_datetime(df["decision_time"], errors="coerce")
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["mobile"] = df["mobile"].astype(str)
        df["min_dist"] = pd.to_numeric(df["min_dist"], errors="coerce")
        df["g1_is_bdo_lead"] = df["g1_is_bdo_lead"].astype(bool)
        return df
    except Exception as e:
        print(f"EXCEPTION IN PULLING G1 DECLINES: {e}")
        return pd.DataFrame()


def process_dataframe(df):
    if df.empty:
        return df

    df.columns = df.columns.str.lower()
    for col in [
        "first_notified_time",
        "installed_time",
        "first_event_time",
        "last_event_time",
    ]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Logic from prod_main.py
    df["decision_time"] = np.where(
        df["installed_decision"] == 1, df["installed_time"], df["first_event_time"]
    )

    for col in ["partner_id", "mobile"]:
        df[col] = df[col].astype(str)
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_train_data(start_dt: str, end_dt: str):
    """
    Fetches training data between inclusive dates.
    Used to build the maps (Hexes, Boundaries, Spatial Weights).

    Args:
        start_dt: 'YYYY-MM-DD'
        end_dt:   'YYYY-MM-DD'
    """
    print(f"Fetching TRAINING data ({start_dt} to {end_dt})...")

    # Snowflake query using dateadd with current_date()
    query = f"""
        select partner_id, mobile, first_notified_time, 
               case when lco_account_installed = partner_id then 1 else 0 end as installed_decision, 
               latitude, longitude, installed_time,
               case when lco_account_installed = partner_id then 'INSTALLED' else first_event end as final_decision, 
               active_base, partner_tenure, first_event_time, last_event, last_event_time, first_event
        from t_node_decisions_active 
        where first_event not in ('','None') 
        and cast(first_notified_time as date) between '{start_dt}' and '{end_dt}'
    """

    try:
        df = _query_snowflake_df(query)
        if df.empty:
            print("NO TRAINING DATA FOUND")
            return pd.DataFrame()

        df = process_dataframe(df)
        df = standardise_decisions(df)
        return df

    except Exception as e:
        print(f"ERROR FETCHING TRAIN DATA: {e}")
        return pd.DataFrame()


def get_test_data(start_dt: str, end_dt: str):
    """
    Fetches test data between inclusive dates.
    Used to simulate new leads and score them against the maps.

    Args:
        start_dt: 'YYYY-MM-DD'
        end_dt:   'YYYY-MM-DD'
    """
    print(f"Fetching TEST data ({start_dt} to {end_dt})...")

    query = f"""
        select partner_id, mobile, first_notified_time,
               case when lco_account_installed = partner_id then 1 else 0 end as installed_decision,
               latitude, longitude, installed_time,
               case when lco_account_installed = partner_id then 'INSTALLED' else first_event end as final_decision,
               active_base, partner_tenure, first_event_time, last_event, last_event_time, first_event
        from t_node_decisions_active
        where first_event not in ('','None')
        and cast(first_notified_time as date) between '{start_dt}' and '{end_dt}'
    """

    try:
        df = _query_snowflake_df(query)
        if df.empty:
            print("NO TEST DATA FOUND")
            return pd.DataFrame()

        df = process_dataframe(df)
        df = standardise_decisions(df)
        return df

    except Exception as e:
        print(f"ERROR FETCHING TEST DATA: {e}")
        return pd.DataFrame()


# ===== FILE: ./lib/data_fetch/wiom_data.py =====

import csv
import os
import tempfile
import uuid
from typing import Dict, Optional, cast

import pandas as pd
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization


class WiomData:
    """Snowflake helper that mirrors the trimmed transaction handler."""

    def __init__(self, db_name: str) -> None:
        if db_name != "snowflake":
            raise ValueError("WiomData only supports 'snowflake' in this playground")

        self._creds_snowflake: Dict[str, Optional[str]] = {
            "user": "MAANAS",
            "account": "BSNUFYZ-IP42416",
            "private_key": "-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIFHTBXBgkqhkiG9w0BBQ0wSjApBgkqhkiG9w0BBQwwHAQI/YGrNmgzhTUCAggAMAwGCCqGSIb3DQIJBQAwHQYJYIZIAWUDBAEqBBBQkB4AlGEOSOiVNdj7gdKWBIIEwJcs7cu/QZ9O1piIkJqHxGI7LrcrgwkPEqmC3B6G0H5Ou6ORELynphb/wgY56MQuVKEUgQnB/lUTG/A+u49IWUS5saXiMWxierIDnuWMUd+2HmEYBTgtAm1NUCX0+kws vMbZado7z7xXnDivE4oPyazlTiEPCruJ23EDHHiR2+PqYJKfUXYtTJyuUhQias8k PYLFbrEHHVXox9IfVCpPwAZN+WZIvup1s0X+wJoSiQApKeSI8M6NcoDq1uKzEadX 2WYWUFtktGGxF+cZNHVbEWnIfAyCpCAe5NxZvQbEH7zdPibJA7l4lP+TxUGEZhWI bgOk3wDlpwTrd6ub5nSTkzNHtO3XnSvoEio+HhFbeA/jmmro6TA4HI5HyhsxuNcA MHq1oglIi6Auu9awYoJi1//2xm3dwBOvRwxVoMs3p60nMdkSQvThLAAqSvlD/TPD MHRUT4yMpOOKuDUPsPxvualFwUax5yFSp7Wo+3ytaZEUtogho3KDz+/F9ZxIVXRU W40tDJ3YHRcN31SSd+9oTvIuZl/7p7noPtCIyAFCS/NAaVq/J62KM7m2A8a0pRAp 2y6OQtn5TRUkKJZ6RJRegNAwJ1cV0CXite5tCZ5QhaVMMTr5fjkGsquDdgm950Vm Ig+Tx9fmH46VBuhM2lZ9lwE25ZGACnX9ZsjND/Cf6OYs3KDcGabfW5FEtq7JYVm5 H2BWWtqnPqKY27AItzXdkHc+stSmW3bghsLYpQ6CmPFqxjE14uzjBMqbwJ3dW7Nc LYBC/rfJU2UEA4DTu08yV9qMNNc2iUlYXm05GHHH2Z1O1MSE9YiFnjLROUMz8Faj TxV1xOLqadnaLgzC2U0+i2RAq5VCS0KnnwFmMVs60huBrIswWNDilRWJMBvXdPQ7 4s0zqXWMwGUQgNiPCBFRBy49/vVuo3xN8iAcCJRE5aKYewShCE91TNm32wjENV58 KYnyi6T5o4/xXaYIhRtw71ED6qIbgdEvPEnp7i1Td7oERYmliWYnNqPD9VYeBR3Q lJ2KBhIkgFo9btv6O/gILfK26mavPOO8Ne5PZ/I5+aWPa4hTZ/1GdxLlQDCkB3mY xHqB7593ObZddz0a83KFtr+G3w4JxUcFctUIlpez4ieUrKyBW0CAIkrchL3OWjtL pCZUBwS2QQwgbKo2kjZ2fBW9VjdyDvQLyUriXnM7u2Zn4K/b7IrRIN/8BbIo3v8i giSzRMbxauS8XMJM1mFTN13hAH19aF2JJfb6jVoARH6DAxppGDy7WMEnQRUWpSn4 06UgCE/RYKzhmdTqs4/fNaGRIS4lYuxSIEL0nL5LvjQJ5sFtFVJE778Ab7q3f1YJ hsJIQ4LH2VPq2hnoJXrAlpfLuWyu3lKk1BJRVF5w7jQVfsLNohjYjDbh1ZuVX1y+ 3Vl108FwoOo8YgdtX1pbDFaAjOKmOWLdhhujYM9GsXtMcOJpu0IEwJi4oq2JksSS 8RWD86EQY+lnIzksmfttAI+HX4KJ8HT+2z5fpdluRjjWbUH8adnFQ6zBSf8DtLX0 DK4KW50RJguI1JRA2R7uvIC6/qSJIlvFMqxFk6WwycyRqJFE+YHjW0p9GtX8LFGo hNqVn/w47BziKFq2+Xwtlqs=\n-----END ENCRYPTED PRIVATE KEY-----",
            "private_key_password": "django123",
            "warehouse": "DS_MED_WH",
            "database": "PROD_DB",
            "schema": "DS_TABLES",
        }

        self._connection_params = self._build_connection_params(self._creds_snowflake)

    def _build_connection_params(self, creds: Dict[str, Optional[str]]) -> Dict[str, object]:
        params: Dict[str, object] = {
            "user": creds["user"],
            "account": creds["account"],
            "warehouse": creds["warehouse"],
            "database": creds["database"],
            "schema": creds["schema"],
        }

        private_key_pem = creds.get("private_key")
        if private_key_pem:
            pk_password = creds.get("private_key_password")
            password_bytes = pk_password.encode() if pk_password else None
            p_key = serialization.load_pem_private_key(
                private_key_pem.encode(),
                password=password_bytes,
                backend=default_backend(),
            )
            params["private_key"] = p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

        return params

    def _connect(self):
        return snowflake.connector.connect(**self._connection_params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def query(
        self, sql: str, cache_file: Optional[str] = None, cache_h: int = 1
    ) -> pd.DataFrame:
        sql_stripped = (sql or "").strip()
        if not sql_stripped:
            raise ValueError("Query cannot be empty")

        if cache_file and os.path.exists(cache_file):
            age_hours = (
                pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(cache_file), unit="s")
            ).total_seconds() / 3600.0
            if age_hours < cache_h:
                return cast(pd.DataFrame, pd.read_csv(cache_file))

        df = self.get_df(sql_stripped)

        if cache_file:
            df.to_csv(cache_file, index=False)

        return df

    def get_df(self, query: str) -> pd.DataFrame:
        print(f"[WiomData] snowflake_select_start: {query[:100]}")
        conn = self._connect()
        cur = conn.cursor()
        try:
            cur.execute(query)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            return pd.DataFrame(rows, columns=pd.Index(cols))
        finally:
            cur.close()
            conn.close()

    def execute(self, query: str, commit: bool = True) -> None:
        print(f"[WiomData] snowflake_execute_start: {query[:100]}")
        conn = self._connect()
        cur = conn.cursor()
        try:
            cur.execute(query)
            if commit:
                conn.commit()
        except Exception as exc:  # pragma: no cover - defensive logging path
            conn.rollback()
            print(f"[WiomData] snowflake_execute_failed: {exc}")
            raise
        finally:
            cur.close()
            conn.close()

    def sync_df_to_table(
        self,
        *,
        df: pd.DataFrame,
        table_name: str,
        schema_dict: Dict[str, str],
        temp_table_name: Optional[str] = None,
    ) -> None:
        if df is None or df.empty:
            print(f"[WiomData] snowflake_sync_skipped_empty_df: {table_name}")
            return

        df = df.copy()
        df.columns = [c.upper() for c in df.columns]
        schema_dict = {k.upper(): v for k, v in schema_dict.items()}

        expected_cols = list(schema_dict.keys())
        if not set(expected_cols).issubset(df.columns):
            missing = sorted(set(expected_cols) - set(df.columns))
            raise ValueError(f"Schema mismatch. Missing columns: {missing}")

        df = cast(pd.DataFrame, df.loc[:, expected_cols])

        actual_table = table_name.upper()
        temp_table = (temp_table_name or f"TEMP_{actual_table}").upper()

        conn = self._connect()
        cur = conn.cursor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name
            df.to_csv(csv_path, index=False)

        staged_name = os.path.basename(csv_path)

        try:
            print(
                f"[WiomData] snowflake_sync_start: table={actual_table} rows={len(df)}"
            )

            cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
            cols_def = ", ".join([f'"{col}" {dtype}' for col, dtype in schema_dict.items()])
            cur.execute(f"CREATE TABLE {temp_table} ({cols_def})")

            cur.execute(
                f"PUT file://{csv_path} @~/{staged_name} AUTO_COMPRESS=FALSE"
            )
            cur.execute(
                f"COPY INTO {temp_table} FROM @~/{staged_name} "
                "FILE_FORMAT=(TYPE=CSV SKIP_HEADER=1 FIELD_OPTIONALLY_ENCLOSED_BY='\"' "
                "ERROR_ON_COLUMN_COUNT_MISMATCH=FALSE)"
            )
            cur.execute(f"REMOVE @~/{staged_name}")

            cur.execute(f"DROP TABLE IF EXISTS {actual_table}")
            cur.execute(f"ALTER TABLE {temp_table} RENAME TO {actual_table}")

            conn.commit()
            print(f"[WiomData] snowflake_sync_complete: table={actual_table}")
        except Exception as exc:  # pragma: no cover - defensive logging path
            conn.rollback()
            print(f"[WiomData] snowflake_sync_failed: table={actual_table} error={exc}")
            raise
        finally:
            try:
                os.remove(csv_path)
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            cur.close()
            conn.close()

    def merge_df_to_table(
        self,
        *,
        df: pd.DataFrame,
        table_name: str,
        schema_dict: Dict[str, str],
        key_columns: list[str],
        update_columns: Optional[list[str]] = None,
        temp_table_name: Optional[str] = None,
    ) -> None:
        if df is None or df.empty:
            print(f"[WiomData] snowflake_merge_skipped_empty_df: {table_name}")
            return

        df = df.copy()
        df.columns = [c.upper() for c in df.columns]
        schema_dict = {k.upper(): v for k, v in schema_dict.items()}
        expected_cols = list(schema_dict.keys())

        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Schema mismatch. Missing columns: {sorted(missing)}")

        key_columns = [c.upper() for c in key_columns]
        update_columns = [c.upper() for c in (update_columns or [])]
        if not update_columns:
            update_columns = [col for col in expected_cols if col not in key_columns]

        df = cast(pd.DataFrame, df.loc[:, expected_cols])

        actual_table = table_name.upper()
        temp_table = (
            temp_table_name or f"TEMP_{actual_table}_{uuid.uuid4().hex[:8]}"
        ).upper()

        conn = self._connect()
        cur = conn.cursor()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            csv_path = f.name
            writer = csv.writer(f)
            writer.writerow(expected_cols)
            writer.writerows(df.itertuples(index=False, name=None))

        staged_name = os.path.basename(csv_path)

        try:
            print(
                f"[WiomData] snowflake_merge_stage_start: table={actual_table} rows={len(df)}"
            )

            cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
            cols_def = ", ".join([f'"{col}" {dtype}' for col, dtype in schema_dict.items()])
            cur.execute(f"CREATE TABLE {temp_table} ({cols_def})")

            cur.execute(
                f"PUT file://{csv_path} @~/{staged_name} AUTO_COMPRESS=FALSE"
            )
            cur.execute(
                f"COPY INTO {temp_table} FROM @~/{staged_name} "
                "FILE_FORMAT=(TYPE=CSV SKIP_HEADER=1 FIELD_OPTIONALLY_ENCLOSED_BY='\"')"
            )
            cur.execute(f"REMOVE @~/{staged_name}")

            on_clause = " AND ".join([f"t.{col} = s.{col}" for col in key_columns])
            update_clause = ", ".join([f"{col} = s.{col}" for col in update_columns])
            insert_columns = ", ".join(expected_cols)
            insert_values = ", ".join([f"s.{col}" for col in expected_cols])

            merge_sql = f"""
                MERGE INTO {actual_table} AS t
                USING {temp_table} AS s
                ON {on_clause}
                WHEN MATCHED THEN UPDATE SET {update_clause}
                WHEN NOT MATCHED THEN INSERT ({insert_columns}) VALUES ({insert_values})
            """
            cur.execute(merge_sql)
            cur.execute(f"DROP TABLE IF EXISTS {temp_table}")

            conn.commit()
            print(f"[WiomData] snowflake_merge_complete: table={actual_table}")
        except Exception as exc:  # pragma: no cover
            conn.rollback()
            print(
                f"[WiomData] snowflake_merge_failed: table={actual_table} error={exc}"
            )
            raise
        finally:
            try:
                os.remove(csv_path)
            except Exception:  # pragma: no cover
                pass
            cur.close()
            conn.close()

    def sync_query_to_table(
        self,
        *,
        select_query: str,
        table_name: str,
        temp_table_name: Optional[str] = None,
    ) -> None:
        actual_table = table_name.upper()
        temp_table = (
            temp_table_name or f"TEMP_{actual_table}_{uuid.uuid4().hex[:8]}"
        ).upper()
        select_query_clean = select_query.strip().rstrip(";")

        conn = self._connect()
        cur = conn.cursor()
        try:
            print(
                f"[WiomData] snowflake_sync_query_start: table={actual_table} temp={temp_table}"
            )

            cur.execute(
                f"CREATE OR REPLACE TABLE {temp_table} AS ({select_query_clean})"
            )

            cur.execute(
                """
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
                  AND TABLE_NAME = %s
                """,
                (actual_table,),
            )
            result = cur.fetchone()
            target_exists = bool(result and result[0] > 0)

            if target_exists:
                cur.execute(f"ALTER TABLE {temp_table} SWAP WITH {actual_table}")
                cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
            else:
                cur.execute(f"ALTER TABLE {temp_table} RENAME TO {actual_table}")

            conn.commit()
            print(f"[WiomData] snowflake_sync_query_complete: table={actual_table}")
        except Exception as exc:  # pragma: no cover
            conn.rollback()
            print(
                f"[WiomData] snowflake_sync_query_failed: table={actual_table} temp={temp_table} error={exc}"
            )
            raise
        finally:
            cur.close()
            conn.close()

    # ------------------------------------------------------------------
    # Backwards compatibility helpers
    # ------------------------------------------------------------------
    def sync_df_to_snowflake(
        self, table_name: str, df: pd.DataFrame, schema_dict: Dict[str, str]
    ) -> None:
        self.sync_df_to_table(df=df, table_name=table_name, schema_dict=schema_dict)


# ===== FILE: ./lib/feature/hop_features.py =====

"""
lib_hop_features.py
3-hop distance-weighted neighbor SE aggregation at the hex level.
No graph library. Just BallTree.

Produces 9 per-hex columns:
    hop1_se_wmean, hop1_se_std, hop1_count
    hop2_se_wmean, hop2_se_std, hop2_count
    hop3_se_wmean, hop3_se_std, hop3_count

Plus 3 cross-hop interaction columns:
    se_gradient_1to3    (hop1 - hop3 wmean; positive = local pocket, likely noisy)
    se_confirmed        (hop1 × hop3 wmean; high only when both rings are good)
    isolation_ratio     (hop1_count / hop3_count; low = fringe, high = dense core)
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from lib.geometry.distance import haversine_rad
from lib.config import EARTH_RADIUS_KILOMETER


def compute_hop_features(df_poly: pd.DataFrame, n_hops: int = 3) -> pd.DataFrame:
    """
    For every non-empty hex in df_poly, compute distance-weighted
    neighbor SE aggregates at hop 1, 2, 3.

    Parameters
    ----------
    df_poly : DataFrame
        Must contain: partner_id, poly_id, se, total, best_size, poly (Shapely).
        'poly' is used only for centroid extraction.
    n_hops : int
        Number of hop rings (default 3).

    Returns
    -------
    DataFrame with columns:
        partner_id, poly_id,
        hop1_se_wmean, hop1_se_std, hop1_count,
        hop2_se_wmean, hop2_se_std, hop2_count,
        hop3_se_wmean, hop3_se_std, hop3_count,
        se_gradient_1to3, se_confirmed, isolation_ratio
    """
    # Extract centroids once
    df = df_poly[["partner_id", "poly_id", "se", "total", "best_size", "poly"]].copy()
    df["clat"] = df["poly"].apply(lambda p: p.centroid.y if p else np.nan)
    df["clon"] = df["poly"].apply(lambda p: p.centroid.x if p else np.nan)
    df = df.dropna(subset=["clat", "clon", "se"])

    all_results = []

    for pid, grp in df.groupby("partner_id"):
        if len(grp) < 2:
            # Solo hex — all hop features NaN
            row = {"partner_id": pid, "poly_id": grp["poly_id"].iloc[0]}
            for h in range(1, n_hops + 1):
                row[f"hop{h}_se_wmean"] = np.nan
                row[f"hop{h}_se_std"] = np.nan
                row[f"hop{h}_count"] = 0
            row["se_gradient_1to3"] = np.nan
            row["se_confirmed"] = np.nan
            row["isolation_ratio"] = np.nan
            all_results.append(row)
            continue

        grp = grp.reset_index(drop=True)
        coords_rad = np.radians(grp[["clat", "clon"]].values)
        se_arr = grp["se"].values
        total_arr = grp["total"].values
        best_size_km = grp["best_size"].iloc[0]

        tree = BallTree(coords_rad, metric="haversine")

        # Hop radii: hex adjacency distance ≈ √3 × best_size for flat-top.
        # hop k ring: from (k-1)*step to k*step
        step_km = best_size_km * 2.0  # centroid-to-centroid of adjacent hexes

        for i in range(len(grp)):
            row = {
                "partner_id": pid,
                "poly_id": grp["poly_id"].iloc[i],
            }

            prev_indices = set()
            prev_indices.add(i)  # exclude self

            hop_wmeans = []

            for h in range(1, n_hops + 1):
                outer_r_km = step_km * h + step_km * 0.3  # small buffer for float
                outer_r_rad = outer_r_km / EARTH_RADIUS_KILOMETER

                idxs_outer = tree.query_radius(
                    coords_rad[i].reshape(1, -1), r=outer_r_rad
                )[0]

                # Ring = outer minus already-counted (inner hops + self)
                ring_idxs = np.array([j for j in idxs_outer if j not in prev_indices])

                if len(ring_idxs) == 0:
                    row[f"hop{h}_se_wmean"] = np.nan
                    row[f"hop{h}_se_std"] = np.nan
                    row[f"hop{h}_count"] = 0
                    hop_wmeans.append(np.nan)
                    continue

                # Distances in km for IDW
                ring_coords = coords_rad[ring_idxs]
                dists_rad = np.array([
                    haversine_rad(coords_rad[i], ring_coords[k])
                    for k in range(len(ring_idxs))
                ])
                dists_km = dists_rad * EARTH_RADIUS_KILOMETER
                dists_km = np.maximum(dists_km, 1e-6)  # avoid div/0

                # Weight = 1 / (hop × distance_km)
                weights = 1.0 / (h * dists_km)

                ring_se = se_arr[ring_idxs]

                wmean = np.average(ring_se, weights=weights)
                # Weighted std
                wvar = np.average((ring_se - wmean) ** 2, weights=weights)
                wstd = np.sqrt(wvar)

                row[f"hop{h}_se_wmean"] = round(wmean, 6)
                row[f"hop{h}_se_std"] = round(wstd, 6)
                row[f"hop{h}_count"] = len(ring_idxs)
                hop_wmeans.append(wmean)

                # Expand prev_indices for next hop ring
                prev_indices.update(ring_idxs.tolist())

            # Cross-hop interactions
            h1 = hop_wmeans[0] if len(hop_wmeans) > 0 else np.nan
            h3 = hop_wmeans[2] if len(hop_wmeans) > 2 else np.nan

            if not np.isnan(h1) and not np.isnan(h3):
                row["se_gradient_1to3"] = round(h1 - h3, 6)
                row["se_confirmed"] = round(h1 * h3, 6)
            else:
                row["se_gradient_1to3"] = np.nan
                row["se_confirmed"] = np.nan

            c1 = row.get("hop1_count", 0)
            c3 = row.get("hop3_count", 0)
            row["isolation_ratio"] = round(c1 / c3, 4) if c3 > 0 else np.nan

            all_results.append(row)

    return pd.DataFrame(all_results)





# ===== FILE: ./lib/feature/spatial_weights.py =====

from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
from lib.config import EARTH_RADIUS_METER

def build_desirability_field_idw(
    df: pd.DataFrame,
    radius_meters: float = 100.0,  # max distance we care about (e.g., 200m)
    decline_weight: float = -2.0,
    install_weight: float = +1.0,
    power: float = 2.0,  # 2 = classic inverse square (very common & good)
) -> pd.DataFrame:
    """
    Assigns field_weight using inverse-distance-weighted average from HARD points only.
    Ambiguous points (HANGING/INDETERMINATE) borrow smoothly from nearby INSTALLED/DECLINED.
    """
    df = df.copy().reset_index(drop=True)
    df = df.dropna(subset=["latitude", "longitude"])

    # 2. Initialize field_weight
    df["field_weight"] = 0.0
    df.loc[df["final_decision"] == "INSTALLED", "field_weight"] = install_weight
    df.loc[df["final_decision"] == "DECLINED", "field_weight"] = decline_weight

    # 3. Separate hard (truth) and ambiguous points
    hard_mask = df["final_decision"].isin(["INSTALLED", "DECLINED"])
    hard_df = df[hard_mask].copy()
    amb_df = df[~hard_mask].copy()

    if len(hard_df) == 0 or len(amb_df) == 0:
        return df  # nothing to borrow

    # 3. Build BallTree on hard points only
    hard_coords_rad = np.radians(hard_df[["latitude", "longitude"]].values)
    tree = BallTree(hard_coords_rad, metric="haversine")

    # 4. Query from ambiguous points
    amb_coords_rad = np.radians(amb_df[["latitude", "longitude"]].values)
    radius_rad = radius_meters / EARTH_RADIUS_METER 

    # This time we NEED distances → return_distance=True
    idx_all, dists_all = tree.query_radius(
        amb_coords_rad, r=radius_rad, return_distance=True, sort_results=True
    )

    # 5. IDW interpolation for each ambiguous point
    borrowed_weights = []
    for dists, idx in zip(dists_all, idx_all):
        if len(idx) == 0 or len(dists) == 0:
            borrowed_weights.append(0.0)
            continue

        # Remove self if accidentally included (shouldn't happen)
        valid = dists > 0
        dists, idx = dists[valid], idx[valid]
        if len(dists) == 0:
            borrowed_weights.append(0.0)
            continue

        # Inverse distance weighting
        weights = 1.0 / (dists**power)  # core IDW formula
        signals = hard_df.iloc[idx]["field_weight"].values

        weighted_avg = np.sum(weights * signals) / np.sum(weights)
        borrowed_weights.append(weighted_avg)

    # 6. Assign back
    df.loc[amb_df.index, "field_weight"] = borrowed_weights

    return df


# ===== FILE: ./lib/geometry/distance.py =====

import numpy as np
from lib.config import EARTH_RADIUS_METER

def equiv_radius_m(area_m2: float) -> float:
    return float(np.sqrt(area_m2 / np.pi))

def haversine_m(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_METER * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def haversine_rad(a, b):
    """Haversine between two (lat_rad, lon_rad) points. Returns radians."""
    dlat = b[0] - a[0]
    dlon = b[1] - a[1]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[0]) * np.cos(b[0]) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(h))

# ===== FILE: ./lib/geometry/find_boundary.py =====

import pandas as pd
import numpy as np
import folium
from shapely.ops import unary_union, transform
from tqdm import tqdm
import os
from sklearn.cluster import DBSCAN
import pyproj
from functools import partial
import lib.config as config


def run_find_boundary():
    df = pd.read_hdf("poly_stats.h5", "df")

    print("IDENTIFYING HEX CENTROID LAT LONG")
    df["centroid_lat"] = df["poly"].apply(lambda p: p.centroid.y if p else np.nan)
    df["centroid_lon"] = df["poly"].apply(lambda p: p.centroid.x if p else np.nan)

    print("FILTERING SERVICEABLE HEXES")
    df = df[df["color"].isin(["lightgreen", "orange"])].copy()
    print(df["installs"].describe())
    print("p50 value is 4, so filtering for hexagons > 4")

    # Compute deciles dynamically for both thresholds
    print("COMPUTING INSTALLS QUANTILES")
    quantiles = [i / 10 for i in range(0, 11)]  # 0.0 to 1.0
    df_quant = df["installs"].quantile(quantiles).reset_index()
    df_quant.columns = ["quantile", "value"]
    print(df_quant)

    p30 = df_quant[df_quant["quantile"] == 0.3]["value"].values[0]
    p70 = df_quant[df_quant["quantile"] == 0.7]["value"].values[0]
    p90 = df_quant[df_quant["quantile"] == 0.9]["value"].values[0]
    print(f"Dynamic p30: {p30} | p70: {p70} | p90: {p90}")

    if config.ENABLE_BOUNDARY_FILTER == 1:
        # Initial filter: keep only hexes above p30 (your density floor)
        print(f"Filtering hexagons with installs > p30 ({p30})")
        df = df[df["installs"] > p30].copy()
    else:
        print("Boundary filtering DISABLED (using all valid hexes)")

    cluster_results = []

    for pid, df_sub in tqdm(df.groupby("partner_id"), desc="CLUSTERING_DBSCAN"):
        p90_id = -1

        # if len(df_sub) < 2:
        # continue  # Skip if too few points
        coords = np.radians(df_sub[["centroid_lat", "centroid_lon"]].values)
        eps_in_km = df_sub["best_size"].max() * 2
        eps_in_radians = eps_in_km / 6371.0

        db = DBSCAN(eps=eps_in_radians, min_samples=2, metric="haversine")
        labels = db.fit_predict(coords)
        df_sub["cluster_id"] = labels

        # Step A: Keep all proper clusters (label >= 0)
        proper_clusters = df_sub[df_sub["cluster_id"] != -1].copy()
        proper_clusters["cluster_type"] = "dbscan_cluster"

        # Step B: From noise (label == -1), rescue high-value singles (installs >= p80)
        noise = df_sub[df_sub["cluster_id"] == -1].copy()
        high_value_noise = noise[noise["installs"] >= p90].copy()

        if not high_value_noise.empty:
            print(
                f"Partner {pid}: Rescuing {len(high_value_noise)} high-value singleton(s) (>= p90 installs)"
            )
            # Assign unique cluster_ids to rescued singles.
            # Start from a large negative offset to avoid collision with real labels
            high_value_noise["cluster_id"] = p90_id
            p90_id -= 1
            high_value_noise["cluster_type"] = (
                "p90_single_cluster"  # Tag as rule-based single
            )

        # Combine proper clusters + rescued singles
        cluster_results.append(pd.concat(
            [proper_clusters, high_value_noise], ignore_index=True
        ))

    df_cluster = pd.concat(cluster_results, ignore_index=True)

    # Now process all clusters per partner (no partner filtering)
    print("DISSOLVING BOUNDARIES FOR EACH CLUSTER PER PARTNER")
    boundary_summary = []

    # Projection for area calculations (UTM 43N for India; make dynamic if needed)
    project = partial(
        pyproj.transform, pyproj.Proj("epsg:4326"), pyproj.Proj("epsg:32643")
    )

    for (partner_id, cluster_id), group_df in tqdm(
        df_cluster.groupby(["partner_id", "cluster_id"])
    ):
        # if len(group_df) < 2:
        # continue  # Skip small clusters

        # Dissolve hexes into one polygon
        dissolved = unary_union(
            [p.buffer(1e-9) for p in group_df["poly"].tolist() if p]
        ).buffer(-1e-9)
        if dissolved.is_empty or dissolved.geom_type == "MultiPolygon":
            dissolved = (
                max(dissolved.geoms, key=lambda p: p.area)
                if dissolved.geom_type == "MultiPolygon"
                else None
            )
        if not dissolved:
            continue

        # Centroid (simple geometric)
        center_lat, center_lon = dissolved.centroid.y, dissolved.centroid.x

        # Boundary coordinates: list of [lon, lat] for exterior
        boundary_coords = (
            [[lon, lat] for lon, lat in dissolved.exterior.coords]
            if hasattr(dissolved, "exterior")
            else []
        )

        # Projected area in km²
        projected_poly = transform(project, dissolved)
        area_km2 = projected_poly.area / 1e6

        boundary_summary.append(
            {
                "partner_id": partner_id,
                "cluster_id": cluster_id,
                "cluster_type": group_df["cluster_type"].iloc[0]
                if "cluster_type" in group_df.columns
                else "unknown",
                "center_lat": center_lat,
                "center_lon": center_lon,
                "total_installs": group_df["installs"].sum(),
                "total_obs": group_df["total"].sum(), # Total hexes as observations
                "n_hexes": len(group_df),
                "area_km2": round(area_km2, 3),
                "boundary_poly": dissolved,  # For point-in-poly checks
                "boundary_coords": boundary_coords,  # For serialization
            }
        )

    boundaries_df = pd.DataFrame(boundary_summary)
    boundaries_df.to_hdf("partner_cluster_boundaries.h5", key="df", mode="w")
    boundaries_df.drop(
        columns=["boundary_poly"], inplace=True
    )  # Drop shapely obj for CSV
    boundaries_df.to_csv("partner_cluster_boundaries.csv", index=False)
    print("\nFINAL BOUNDARY SUMMARY")
    print(boundaries_df.head(10))

    # Example: Check if point is inside a boundary (reload polys if needed)
    # For a new point (lat, lon), loop over boundaries_df, create Polygon from boundary_coords, check Point(lon, lat).within(poly)

    # Plotting: One overview map with all dissolved polygons
    print("GENERATING OVERVIEW MAP WITH ALL CLUSTER BOUNDARIES")
    overview = folium.Map(
        location=[
            boundaries_df["center_lat"].mean(),
            boundaries_df["center_lon"].mean(),
        ],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    for _, row in boundaries_df.iterrows():
        if not row["boundary_coords"]:
            continue
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in row["boundary_coords"]],
            color="#2ecc71",
            weight=3,
            fill=True,
            fill_opacity=0.3,
            tooltip=f"Partner {row['partner_id']} Cluster {row['cluster_id']}<br>Installs: {row['total_installs']}<br>Area: {row['area_km2']} km²",
        ).add_to(overview)

        folium.Marker(
            [row["center_lat"], row["center_lon"]],
            icon=folium.Icon(color="blue", icon="info-sign"),
            tooltip=f"Center: {row['total_installs']} installs",
        ).add_to(overview)

    os.makedirs("virtual_boundary", exist_ok=True)

    overview.save("virtual_boundary/all_clusters_overview.html")
    print("DONE. Open 'virtual_boundary/all_clusters_overview.html' to view polygons.")

    # New addition: Generate individual maps for each partner, showing their clusters
    print("GENERATING INDIVIDUAL PARTNER MAPS WITH CLUSTER BOUNDARIES")
    unique_partners = boundaries_df["partner_id"].unique()
    for partner_id in tqdm(unique_partners, desc="PARTNER_MAPS"):
        partner_df = boundaries_df[boundaries_df["partner_id"] == partner_id]
        if partner_df.empty:
            continue

        # Center the map on the mean of this partner's cluster centers
        mean_lat = partner_df["center_lat"].mean()
        mean_lon = partner_df["center_lon"].mean()
        partner_map = folium.Map(
            location=[mean_lat, mean_lon],
            zoom_start=12,  # Zoom in a bit more for individual partners
            tiles="CartoDB positron",
        )

        # Add polygons for each cluster, with distinct colors if desired (here using same color for simplicity)
        for _, row in partner_df.iterrows():
            if not row["boundary_coords"]:
                continue
            folium.Polygon(
                locations=[(lat, lon) for lon, lat in row["boundary_coords"]],
                color="#2ecc71",
                weight=3,
                fill=True,
                fill_opacity=0.3,
                tooltip=f"Cluster {row['cluster_id']}<br>Installs: {row['total_installs']}<br>Area: {row['area_km2']} km²",
            ).add_to(partner_map)
            folium.Marker(
                [row["center_lat"], row["center_lon"]],
                icon=folium.Icon(color="blue", icon="info-sign"),
                tooltip=f"Cluster {row['cluster_id']} Center: {row['total_installs']} installs",
            ).add_to(partner_map)

        # Save the map
        partner_map.save(f"virtual_boundary/partner_{partner_id}_clusters.html")
        print(
            f"Generated map for Partner {partner_id}: 'virtual_boundary/partner_{partner_id}_clusters.html'"
        )

    print("ALL PARTNER MAPS GENERATED.")

    """
    HOW TO USE IT LATER:

    poly_df = pd.read_hdf('partner_cluster_boundaries.h5', 'df')
    pt = Point(lon, lat)
    matches = poly_df[poly_df['boundary_poly'].apply(lambda p: p.contains(pt))]

    """

    return boundaries_df

if __name__ == "__main__":
    run_find_boundary()

# ===== FILE: ./lib/geometry/geometric_features.py =====

# playground/geometric_features.py

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.special import expit
import lib.config as config


def sigmoid(x):
    return expit(x)


def compute_local_geometry(lead_lat, lead_lon, neighbor_df, radius_m=250):
    """
    Computes geometric shape descriptors of the historical points surrounding a lead.

    Args:
        lead_lat, lead_lon: Coordinates of the new lead
        neighbor_df: DataFrame of historical decisions within radius_m
                     Must contain ['latitude', 'longitude', 'final_decision']
        radius_m: The radius used to fetch these neighbors

    Returns:
        dict: Geometric features (anisotropy, density, etc.)
    """

    if len(neighbor_df) < 3:
        return {
            "local_anisotropy": 0.0,
            "local_density": 0.0,
            "hull_area": 0.0,
            "linearity_score": 0.0,
            "spread_m": 0.0,
        }

    # Convert to local meters (approximate projection centered on lead)
    # x = (lon - lead_lon) * 111320 * cos(lat)
    # y = (lat - lead_lat) * 111320

    lat0_rad = np.radians(lead_lat)
    meters_per_deg_lat = config.METERS_PER_DEG_LAT
    meters_per_deg_lon = meters_per_deg_lat * np.cos(lat0_rad)

    coords_m = np.zeros((len(neighbor_df), 2))
    coords_m[:, 0] = (neighbor_df["longitude"] - lead_lon) * meters_per_deg_lon  # x
    coords_m[:, 1] = (neighbor_df["latitude"] - lead_lat) * meters_per_deg_lat  # y

    # 1. Covariance & Eigenvalues (Anisotropy)
    # Centered on the geometric center of the neighbors, not necessarily the lead
    center_m = np.mean(coords_m, axis=0)
    centered_coords = coords_m - center_m

    cov_matrix = np.cov(centered_coords, rowvar=False)
    eig_vals = np.linalg.eigvals(cov_matrix)
    eig_vals = np.sort(eig_vals)[::-1]  # Descending

    # Anisotropy: 1 - (lambda2 / lambda1).
    # Close to 1 = Line/Gully. Close to 0 = Blob/Round.
    if eig_vals[0] > 0:
        anisotropy = 1.0 - (eig_vals[1] / eig_vals[0])
    else:
        anisotropy = 0.0

    # 2. Convex Hull Density
    try:
        hull = ConvexHull(coords_m)
        hull_area = hull.volume  # In 2D, volume is area
        # Points per sq meter * 1000 (scaled for readability)
        density = (len(neighbor_df) / hull_area) * 1000 if hull_area > 1 else 0
    except:
        hull_area = 0.0
        density = 0.0

    # 3. Spread (Root Mean Square Distance from Centroid)
    spread = np.sqrt(np.mean(np.sum(centered_coords**2, axis=1)))

    linearity = eig_vals[0] / (eig_vals[1] + 1e-6)

    # Regime scores (logistic-scaled)
    dense_score = sigmoid(0.8 * density - 0.03 * spread - 1.2 * anisotropy)
    gully_score = sigmoid(2.0 * anisotropy + 0.6 * np.log1p(linearity) + 0.4 * density)
    sparse_score = sigmoid(-1.0 * density + 0.04 * spread)

    return {
        "local_anisotropy": round(anisotropy, 3),
        "local_density": round(density, 3),
        "hull_area": round(hull_area, 1),
        "linearity_score": round(linearity, 1),  # Ratio
        "spread_m": round(spread, 1),
        "dense_score": round(dense_score, 4),
        "gully_score": round(gully_score, 4),
        "sparse_score": round(sparse_score, 4),
    }


def batch_compute_geometry(df_leads, df_history, radius_m=250):
    """
    Batched wrapper for computing geometry for many leads.
    Uses BallTree for efficiency.
    """
    from sklearn.neighbors import BallTree

    # Build tree on history
    hist_rad = np.radians(df_history[["latitude", "longitude"]].values)
    tree = BallTree(hist_rad, metric="haversine")

    # Query leads
    leads_rad = np.radians(df_leads[["latitude", "longitude"]].values)
    radius_rad = radius_m / 6371000.0

    indices_list = tree.query_radius(leads_rad, r=radius_rad)

    results = []
    for i, indices in enumerate(indices_list):
        if len(indices) < 3:
            results.append(
                {
                    "local_anisotropy": 0.0,
                    "local_density": 0.0,
                    "hull_area": 0.0,
                    "linearity_score": 0.0,
                    "spread_m": 0.0,
                }
            )
            continue

        # Get neighbors for this lead
        neighbors = df_history.iloc[indices]

        # Compute metrics
        lead_lat = df_leads.iloc[i]["latitude"]
        lead_lon = df_leads.iloc[i]["longitude"]

        metrics = compute_local_geometry(lead_lat, lead_lon, neighbors, radius_m)
        results.append(metrics)

    return pd.DataFrame(results)


def calculate_adaptive_h(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Computes adaptive bandwidth 'h' for each point based on its k-th nearest neighbor distance
    of the same decision type (Install/Decline).

    1. Deduplicates training data to unique mobile/location level.
    2. Computes H for these unique points.
    3. Merges H back to the original dataframe.
    """
    from sklearn.neighbors import BallTree

    if df_train.empty:
        return df_train

    # Work on a copy to avoid side effects
    df_train = df_train.copy()

    # 1. Deduplicate to unique mobile level
    # We take the 'best' case scenario for a location: max field weight (Install > Decline)
    # and the earliest time it was seen.
    unique_pts = df_train.groupby(
        ["mobile", "latitude", "longitude"], as_index=False
    ).agg({"decision_time": "min", "field_weight": "max"})

    # Initialize adaptive_h column in unique set
    unique_pts["adaptive_h"] = np.nan

    # 2. Compute H for unique points
    # Process for both positive (Installs) and negative (Declines) cohorts
    for subset_mask, subset_name in [
        (unique_pts["field_weight"] >= 0, "install"),
        (unique_pts["field_weight"] < 0, "decline"),
    ]:
        if not subset_mask.any():
            continue

        subset_df = unique_pts[subset_mask]

        # Need enough points to find k neighbors
        # We need k+1 points because the query includes the point itself as the 0-th neighbor
        target_k = config.ADAPTIVE_H_NEIGHBOR_K
        query_k = target_k + 1

        if len(subset_df) < query_k:
            # Not enough neighbors, leave as NaN (will use default later if merged, or keep existing)
            continue

        coords = np.radians(subset_df[["latitude", "longitude"]].values)
        tree = BallTree(coords, metric="haversine")

        # query returns distances to k nearest neighbors
        # column 0 is self (dist~0), column 1 is 1st NN, ..., column k is k-th NN
        dist, _ = tree.query(coords, k=query_k)

        # Get distance to the target neighbor
        neighbor_dist_rad = dist[:, target_k]
        neighbor_dist_m = neighbor_dist_rad * 6371000  # Earth radius in meters

        # Clip to safe bounds to avoid spikes or overly broad fields
        neighbor_dist_m = np.clip(
            neighbor_dist_m, config.ADAPTIVE_H_MIN, config.ADAPTIVE_H_MAX
        )

        unique_pts.loc[subset_mask, "adaptive_h"] = neighbor_dist_m

    # 3. Merge back to original dataframe
    # We map the computed 'adaptive_h' back to the main df based on the full spatial key.
    # This handles cases where one mobile appears at multiple distinct locations.

    # Select only the columns we need to merge
    merge_cols = ["mobile", "latitude", "longitude", "adaptive_h"]

    # Left merge to attach adaptive_h where calculated
    df_merged = df_train.merge(
        unique_pts[merge_cols], on=["mobile", "latitude", "longitude"], how="left"
    )

    # Update 'h' column: use adaptive_h if present, otherwise keep existing default
    df_merged["h"] = df_merged["adaptive_h"].fillna(df_merged["h"])

    # Drop the temporary column
    df_merged = df_merged.drop(columns=["adaptive_h"])

    return df_merged


# ===== FILE: ./lib/geometry/hex.py =====

# diagnostic_hex_maps_pure.py
# NO H3. NO UBER. JUST TRUTH.
import pandas as pd
import folium
from shapely.geometry import Polygon
import os
from math import radians, cos, sin, sqrt
from shapely import contains_xy
from lib import config


def create_hex_grid(center_lat, center_lon, radius_km=3.0, hex_size_km=0.25):
    """
    Perfect regular hexagonal tiling using local flat approximation.
    Accurate to <0.5% error within 10 km — more than enough for 250m hexes.
    """
    # Constants in metres
    
    meters_per_deg_lon = config.METERS_PER_DEG_LAT * cos(radians(center_lat))

    # Hex geometry (flat-top orientation)
    hex_radius_m = hex_size_km * 1000  # center → vertex
    dx = hex_radius_m * sqrt(3)  # horizontal spacing
    dy = hex_radius_m * 1.5  # row spacing

    n_rings = int(radius_km / hex_size_km) + 2
    hexes = []

    for i in range(-n_rings, n_rings + 1):
        for j in range(-n_rings, n_rings + 1):
            if abs(i) + abs(j) + abs(-i - j) > 2 * n_rings:
                continue

            # Offset to metres
            x_m = dx * (j + i / 2.0)
            y_m = dy * i

            # Hex centre in lat/lon
            c_lat = center_lat + y_m / config.METERS_PER_DEG_LAT
            c_lon = center_lon + x_m / meters_per_deg_lon

            # Six vertices around this centre
            vertices = []
            for k in range(6):
                angle = radians(60 * k + 30)  # flat-top
                vx = c_lon + (hex_radius_m / meters_per_deg_lon) * cos(angle)
                vy = c_lat + (hex_radius_m / config.METERS_PER_DEG_LAT) * sin(angle)
                vertices.append((vx, vy))

            hexes.append(Polygon(vertices))

    return hexes


def find_best_hexes(center_lat, center_lon, sources, hex_sizes=None, radius_km=3.0):
    src_lats = sources["latitude"].values
    src_lons = sources["longitude"].values

    # Allow caller to inject hex_sizes; fall back to default list for backward compatibility
    if hex_sizes is None:
        hex_sizes = [
            0.025,
            0.05,
            0.08,
            0.10,
            0.12,
            0.14,
            0.16,
            0.18,
            0.20,
            0.22,
            0.24,
            0.25,
        ]

    # DEFAULT VALUE OF SIZE IF A PARTNER DOESN'T MEET THE SE SEPARATION CRITERION: EG. NEW PARTNERS
    best_size = config.DEFAULT_HEX_SIZE
    max_separation = 0
    for hex_size_km in hex_sizes:
        hexes = create_hex_grid(
            center_lat, center_lon, radius_km=radius_km, hex_size_km=hex_size_km
        )
        poly_id = 0
        poly_chars = []
        for hex_poly in hexes:
            mask = contains_xy(hex_poly, src_lons, src_lats)
            hex_src = sources[mask]

            installs = len(hex_src[hex_src["is_installed"] == 1])
            declines = len(hex_src[hex_src["is_declined"] == 1])
            total = len(hex_src)
            if total == 0:
                continue

            poly_id += 1
            poly_chars.append(
                {
                    "poly_id": poly_id,
                    "installs": installs,
                    "declines": declines,
                    "total": total,
                }
            )

        df_poly = pd.DataFrame(
            poly_chars, columns=["poly_id", "installs", "declines", "total"]
        )
        df_poly["se"] = df_poly["installs"] / df_poly["total"]
        # print(f"SHAPE OF DF POLY -> NUMBER OF HEXAGONS: {df_poly.shape[0]}, {df_poly['poly_id'].nunique()}")
        if df_poly.shape[0] > 20:
            # df_poly['se_rng'] = pd.qcut(df_poly['se'], q=5,labels=False, duplicates='drop')+1
            df_poly["se_rng"] = pd.cut(
                df_poly["se"],
                bins=[i / 5 for i in range(6)],
                labels=[1, 2, 3, 4, 5],
                include_lowest=True,
            ).astype(int)
            # print(f"FOR {hex_size_km} POLY SIZE, BINS FORMED ARE: {df_poly.groupby(['se_rng']).agg(nmbr_hex=('poly_id','count'), se = ('se','mean'))}")
            # print(df_poly['se_rng'].value_counts())
            if df_poly["se_rng"].nunique() == 5:
                mask1 = df_poly["se_rng"] == 5
                mask2 = df_poly["se_rng"] == 2

                se_low = (
                    df_poly[mask2]["installs"].sum() / df_poly[mask2]["total"].sum()
                )
                se_high = (
                    df_poly[mask1]["installs"].sum() / df_poly[mask1]["total"].sum()
                )

                separation = se_high - se_low

                # print(f"FOR {hex_size_km}, SE LOW IS {se_low}, SE HIGH IS {se_high}, SEPARATION IS: {separation}")

                if separation > max_separation:
                    max_separation = separation
                    best_size = hex_size_km

    print(f"MAX SEPARATION IS: {max_separation}, BEST SIZE IS: {best_size}")

    return best_size


def compute_hexes(
    hexes,
    center_lat,
    center_lon,
    sources,
    bad_se,
    mid_se,
    best_size,
    METERS_PER_DEG_LAT,
    cos_lat,
    pid,
):
    hex_stats = []
    src_lats = sources["latitude"].values
    src_lons = sources["longitude"].values

    poly_id = 0

    for hex_poly in hexes:
        # Fast vectorised containment
        mask = contains_xy(hex_poly, src_lons, src_lats)
        hex_src = sources[mask]

        if len(hex_src) == 0:
            continue

        # ------------------------------------------------------------------
        # Historical stats
        # ------------------------------------------------------------------

        installs = len(hex_src[hex_src["is_installed"] == 1])
        declines = len(hex_src[hex_src["is_declined"] == 1])
        total = len(hex_src)

        se = installs / total if total > 0 else 0
        color = (
            "crimson" if se <= bad_se else "orange" if se <= mid_se else "lightgreen"
        )

        # ------------------------------------------------------------------
        # Store everything
        # ------------------------------------------------------------------
        poly_id += 1
        hex_stats.append(
            {
                "partner_id": pid,
                "best_size": best_size,
                "poly_id": poly_id,
                "poly": hex_poly,
                "se": round(se, 4),
                "installs": int(installs),
                "declines": int(declines),
                "total": int(total),
                "color": color,
            }
        )

    # ------------------------------------------------------------------
    # Map generation (only if we have hexes)
    # ------------------------------------------------------------------

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron"
    )

    for stat in hex_stats:
        tooltip = f"""
        <b>Historical SE</b>: {stat["se"]:.1%} ({stat["installs"]}/{stat["total"]})<br>
        <b>Installs:</b> {stat["installs"]}<br>
        <b>Declines:</b> {stat["declines"]}<br>
        <b>Total Decisions:</b> {stat["total"]}<br>


        """
        folium.Polygon(
            locations=[(c[1], c[0]) for c in stat["poly"].exterior.coords],
            color=stat["color"],
            weight=2,
            fill=True,
            fillOpacity=0.65,
            tooltip=tooltip.replace("\n", "<br>"),
        ).add_to(m)

    # Historical points
    for _, r in sources.iterrows():
        color = "lightpink" if r["is_installed"] == 1 else "darkorange"
        folium.CircleMarker(
            [r.latitude, r.longitude], radius=4, color=color, fill=True
        ).add_to(m)

    os.makedirs("DIAGNOSTIC_HEX_MAPS_DYNAMIC_HEX", exist_ok=True)

    m.save(
        f"DIAGNOSTIC_HEX_MAPS_DYNAMIC_HEX/partner_{pid}_PURE_CORRECT_{best_size}.html"
    )

    return hex_stats
