# playground/score_test.py

import argparse
from typing import cast
import pandas as pd
import numpy as np
import os
import data_lib.config as config
import datetime as dt
from data_lib.data_fetch.get_data import get_test_data, get_g1_distance
from data_lib.compute import process
from data_lib.geometry.geometric_features import batch_compute_geometry, calculate_adaptive_h
from step3_simpulate import run_declines_simulation
from data_lib.data_fetch.get_ops_data import build_partner_ops_vector
from data_lib.stocks.gatekeeper import run_gates
from data_lib.feature.ops_features import compute_operational_score
from data_lib.feature.composite import compute_composite
import data_lib.stocks.stocks_config as sc
import json


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

    # B_OPERATIONAL
    df_ops = build_partner_ops_vector(config.TEST_START_DATE, config.TEST_END_DATE)

    # G — GATEKEEPER
    df_test, df_ops = run_gates(df_test, df_train, df_ops)
    if not df_ops.empty:
        df_ops = compute_operational_score(df_ops)
    
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
    
    # R — COMPOSITE
    df = compute_composite(df, df_ops)

    # E — EXPOSURE CHECK
    if not df_ops.empty and "nmbr_active_leads" in df_ops.columns and "expected_daily_slots" in df_ops.columns:
        exposure_check = df_ops[["partner_id", "nmbr_active_leads", "expected_daily_slots"]].copy()
        exposure_check["exposure_ratio"] = (
            exposure_check["nmbr_active_leads"]
            / exposure_check["expected_daily_slots"].replace(0, np.nan)
        ).fillna(0)
        exposure_check["e_concentrated"] = (
            exposure_check["exposure_ratio"] > sc.EXPOSURE_CONCENTRATION_FACTOR
        ).astype(int)
        df = df.merge(
            exposure_check[["partner_id", "exposure_ratio", "e_concentrated"]],
            on="partner_id", how="left",
        )
        df["e_concentrated"] = df["e_concentrated"].fillna(0).astype(int)
        downgrade_mask = (df["e_concentrated"] == 1) & (df["confidence_tier"] != "DECLINE")
        tier_downgrade = {"HIGH": "MOD", "MOD": "LOW", "LOW": "DECLINE"}
        df.loc[downgrade_mask, "confidence_tier"] = df.loc[downgrade_mask, "confidence_tier"].map(tier_downgrade)
        df["r_serviceable"] = np.where(df["confidence_tier"].isin(["HIGH", "MOD"]), 1, 0)

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
    # R tier reports
    tier_dist = df.groupby(["min_dist_bucket", "confidence_tier"], dropna=False).agg(
        total=("mobile", "count"), installs=("installed_decision", "sum")).reset_index()
    tier_dist["install_rate"] = tier_dist["installs"] / tier_dist["total"]
    tier_dist.to_csv(os.path.join(REPORTS_DIR, "install_rate_by_min_dist_bucket_and_confidence_tier.csv"), index=False)

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
    # H — CONFIG SNAPSHOT
    config_snapshot = {}
    for field in sc.H_SNAPSHOT_FIELDS:
        config_snapshot[field] = getattr(config, field, None)
    df["config_snapshot"] = json.dumps(config_snapshot)

    if not df_ops.empty:
        df_ops.to_csv(os.path.join(REPORTS_DIR, "partner_ops_vector.csv"), index=False)

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
