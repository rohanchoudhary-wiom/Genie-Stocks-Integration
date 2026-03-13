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
from steps.step3_simpulate import run_declines_simulation
from data_lib.data_fetch.get_ops_data import build_partner_ops_vector
from data_lib.stocks.gatekeeper import run_gates
from data_lib.feature.ops_features import compute_operational_score
from data_lib.feature.composite import compute_composite
import data_lib.stocks.stocks_config as sc
import json


def evaluate_bucket(df, bucket_name):
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


def score_temporal_windows(
    df_train_full: pd.DataFrame,
    df_test: pd.DataFrame,
    df_poly: pd.DataFrame,
    df_bound: pd.DataFrame,
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each temporal window, filter df_train to only recent source points,
    re-run process() to get a time-windowed Gaussian field, and merge the
    temporal field columns back onto scored_df.

    This gives us hard-cutoff temporal field variants alongside the
    all-time exponentially-decayed field.

    Columns added per window:
        predicted_field_hex_{wd}d   — kernel-sum-weighted field from last wd days only
        kernel_sum_{wd}d            — total kernel weight (confidence proxy)
        total_sources_{wd}d         — number of source points used
    """
    ref_date = pd.Timestamp(config.TRAIN_END_DATE)

    for wd in config.TEMPORAL_WINDOWS:
        cutoff = ref_date - pd.Timedelta(days=wd)
        df_train_w = df_train_full[df_train_full["decision_time"] >= cutoff].copy()

        n_src = len(df_train_w)
        print(f"\n[TEMPORAL FIELD {wd}d] {n_src} source points (cutoff={cutoff.date()})")

        if n_src < 50:
            print(f"[TEMPORAL FIELD {wd}d] Too few sources, skipping")
            scored_df[f"predicted_field_hex_{wd}d"] = np.nan
            scored_df[f"total_sources_field_{wd}d"] = 0
            continue

        try:
            scored_w = process(
                df_train_w,
                df_test,
                df_poly,
                df_bound,
                lambda_decay=config.LAMBDA_DECAY,
                max_radius_m=int(config.MIN_DIST_CUTOFF_M),
            )

            if scored_w is None or scored_w.empty:
                print(f"[TEMPORAL FIELD {wd}d] process() returned empty")
                scored_df[f"predicted_field_hex_{wd}d"] = np.nan
                scored_df[f"total_sources_field_{wd}d"] = 0
                continue

            # Extract only the field columns we need — avoid overwriting all-time consensus columns
            temporal_cols = scored_w[["mobile"]].copy()
            temporal_cols[f"predicted_field_hex_{wd}d"] = scored_w.get(
                "predicted_field_hex", np.nan
            )
            temporal_cols[f"total_sources_field_{wd}d"] = scored_w.get(
                "total_sources_all_hexes", 0
            )

            # Merge onto main scored_df
            scored_df = scored_df.merge(temporal_cols, on="mobile", how="left", suffixes=("", f"_dup_{wd}"))

            # Drop any dups
            scored_df = scored_df.loc[:, ~scored_df.columns.str.endswith(f"_dup_{wd}")]

            n_valid = scored_df[f"predicted_field_hex_{wd}d"].notna().sum()
            print(f"[TEMPORAL FIELD {wd}d] SUCCESS — {n_valid}/{len(scored_df)} leads with field")

        except Exception as e:
            print(f"[TEMPORAL FIELD {wd}d] ERROR: {e}")
            scored_df[f"predicted_field_hex_{wd}d"] = np.nan
            scored_df[f"total_sources_field_{wd}d"] = 0

    # ── Derived: field momentum ──
    scored_df["field_momentum"] = (
        scored_df[f"predicted_field_hex_{config.TEMPORAL_WINDOWS[0]}d"] - scored_df[f"predicted_field_hex_{config.TEMPORAL_WINDOWS[-1]}d"]
    )

    return scored_df


def main(simulate=False):
    print("--- STARTING TEST SCORING (PLAYGROUND) ---")

    # Base directory resolution
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)

    ARTIFACTS_DIR = os.path.join(PARENT_DIR, "artifacts")
    REPORTS_DIR = os.path.join(PARENT_DIR, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    cache_scored_csv = os.path.join(REPORTS_DIR, "scored_df.csv")
    cache_scored_h5 = os.path.join(REPORTS_DIR, "scored_df.h5")

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

    # Keep full train for temporal windowing later
    df_train_full = df_train.copy()

    # 2. Load Test Data
    print(f"Test window: {config.TEST_START_DATE} -> {config.TEST_END_DATE}")
    df_test = get_test_data(config.TEST_START_DATE, config.TEST_END_DATE)
    if df_test.empty:
        print("No test data.")
        return

    # Aggregate to mobile-level (one row per lead/location)
    before = len(df_test)
    df_test = df_test.groupby(["mobile", "latitude", "longitude"], as_index=False).agg(
        partner_id=("partner_id", "first"),
        nmbr_partners=("partner_id", "count"),
        decision_time=("decision_time", "min"),
        installed_decision=("installed_decision", "max"),
        installed_time=("installed_time", "max"),
    )
    print(f"Aggregated test decisions: {before} rows -> {len(df_test)} unique mobiles")

    # B_OPERATIONAL
    df_ops_train = build_partner_ops_vector(config.TRAIN_START_DATE, config.TRAIN_END_DATE)
    df_ops_test = build_partner_ops_vector(config.TEST_START_DATE, config.TEST_END_DATE)
    print(f"\n\nDF OPS built {df_ops_train.shape}\n {df_ops_test.shape}\n\n")

    # G — GATEKEEPER
    df_test, df_ops_train = run_gates(df_test, df_train, df_ops_train)
    if not df_ops_train.empty:
        df_ops_train = compute_operational_score(df_ops_train)

    # 3. Geometric Features — NOW WITH TEMPORAL
    print("Computing Geometric 'Pattern-of-History' Features (+ temporal windows)...")
    df_train_geom = df_train.copy()
    if config.GEOM_INSTALL_FILTER == 1:
        df_train_geom = df_train_geom[df_train_geom["installed_decision"] == 1].copy()

    geom_features = batch_compute_geometry(
        df_test, df_train_geom, radius_m=int(config.GEOM_SEARCH_RADIUS_M),
        reference_date=config.TEST_START_DATE,
    )
    df_test = pd.concat([df_test.reset_index(drop=True), geom_features], axis=1)

    # Derive geometry regime labels
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

    # 4. Standard Scoring (ALL-TIME: Hex + Field + Boundaries)
    print("Running Production Scoring Engine (with cache)...")

    if os.path.exists(cache_scored_h5):
        print("Loading scored df output...")
        scored_df = pd.read_hdf(cache_scored_h5, "df")
    else:
        print("Cache not found. Running process()...")
        scored_df = process(
            df_train,
            df_test,
            df_poly,
            df_bound,
            lambda_decay=config.LAMBDA_DECAY,
            max_radius_m=int(config.MIN_DIST_CUTOFF_M),
        )
        print("Saving cache...")
        scored_df.to_csv(cache_scored_csv, index=False)
        category_cols = scored_df.select_dtypes(include=["category"]).columns
        if len(category_cols) > 0:
            scored_df[category_cols] = scored_df[category_cols].astype(str)
        scored_df.to_hdf(cache_scored_h5, key="df", mode="w")

    if scored_df is None or scored_df.empty:
        print("Scoring returned empty dataframe.")
        return

    required_cols = [
        "parent_color", "parent_installs", "predicted_field_hex", "contested_field",
    ]
    missing = [c for c in required_cols if c not in scored_df.columns]
    if missing:
        print(f"Scoring output missing required columns: {missing}")
        return

    # ══════════════════════════════════════════════════════════════
    # 4b. TEMPORAL FIELD SCORING — run process() per time window
    # ══════════════════════════════════════════════════════════════
    print("\n--- TEMPORAL FIELD SCORING ---")
    scored_df = score_temporal_windows(
        df_train_full=df_train_full,
        df_test=df_test,
        df_poly=df_poly,
        df_bound=df_bound,
        scored_df=scored_df,
    )

    # Report temporal field coverage
    for wd in config.TEMPORAL_WINDOWS:
        col = f"predicted_field_hex_{wd}d"
        if col in scored_df.columns:
            n = scored_df[col].notna().sum()
            print(f"  {col}: {n}/{len(scored_df)} leads")

    # ══════════════════════════════════════════════════════════════
    # 5. G1 distance merge + decision logic
    # ══════════════════════════════════════════════════════════════
    test_start = dt.date.fromisoformat(config.TEST_START_DATE)
    g1_start = (
        test_start - dt.timedelta(days=config.G1_LOOKBACK_EXTRA_DAYS)
    ).isoformat()
    g1_end = config.TEST_END_DATE

    print(f"G1 logs window: {g1_start} -> {g1_end}")
    df_g1 = get_g1_distance(g1_start, g1_end)
    if df_g1.empty:
        print("Warning: G1 distance data empty; min_dist buckets will be NaN")
        df = scored_df.copy()
    else:
        keep_cols = ["mobile", "min_dist", "g1_is_bdo_lead", "serviceable", "zone_alias"]
        df = scored_df.merge(df_g1[keep_cols], on="mobile", how="inner")

    bins = [-np.inf, 20, 40, 60, 100, np.inf]
    labels = ["0-20", "20-40", "40-60", "60-100", "100+"]
    df["min_dist_bucket"] = pd.cut(
        df["min_dist"], bins=bins, labels=labels, right=True, include_lowest=True
    )

    # Decision logic
    indeterminate_mask = (df["parent_color"] == "lightgreen") & (
        df["parent_installs"] <= config.INDETERMINATE_INSTALLS_CUTOFF
    )
    df["parent_color"] = np.where(indeterminate_mask, "indeterminate", df["parent_color"])
    df["parent_color_super"] = np.where(
        df["parent_color"] == "indeterminate", "lightgreen", df["parent_color"]
    )

    df["gravity_score"] = np.where(
        df["predicted_field_hex"].isna(),
        -99,
        np.where(
            (df["predicted_field_hex"] > config.FIELD_THRESHOLD) & (df["total"] > config.PARENT_TOTAL_THRESHOLD), 2,
            np.where(df["predicted_field_hex"] > config.FIELD_THRESHOLD, 1, 0)
        )
    )

    df["comp_gravity_score"] = np.where(
        df["contested_field"].isna(),
        -99,
        np.where(df["contested_field"] > config.CONTEST_FIELD_THRESHOLD, 1, 0),
    )

    conditions = [
        (df["gravity_score"] == 2) & (df["comp_gravity_score"] == 1),
        (df["gravity_score"] == 2),
        (df["gravity_score"] == 1) & (df["comp_gravity_score"] == 1),
        (df["gravity_score"] == 1),
        (df["comp_gravity_score"] == 1),
    ]
    choices = [
        "A. STRONG Comp+ STRONG Field",
        "B. STRONG Field",
        "C. STRONG Comp + WEAK Field",
        "D. WEAK FIELD",
        "E. STRONG Comp",
    ]
    df["bk_class_score"] = np.select(conditions, choices, default="F. Bad Field")

    booking_classifier_v1 = (df["parent_color_super"].isin(["lightgreen"])) & (
        df["bk_class_score"].isin([
            "A. STRONG Comp+ STRONG Field", "B. STRONG Field",
            "C. STRONG Comp + WEAK Field", "E. STRONG Comp",
        ])
    )
    booking_classifier_v2 = (df["parent_color_super"].isin(["orange"])) & (
        df["bk_class_score"].isin(["A. STRONG Comp+ STRONG Field", "B. STRONG Field"])
    )

    df["booking_classifier_new"] = np.where(
        booking_classifier_v1 | booking_classifier_v2, 1, 0
    )
    df["final_serviceability"] = df["booking_classifier_new"]

    # Geometric flags
    df["is_sparse"] = np.where(
        df["local_density"] < config.SPARSE_DENSITY_THRESHOLD, 1, 0
    )
    df["is_deep"] = np.where(
        df["depth_score_point_hex"] > config.DEPTH_SCORE_THRESHOLD, 1, 0
    )

    # R — COMPOSITE (now includes temporal norms + spatial_shrunk_{wd}d)
    df = compute_composite(df, df_ops_train)

    # E — EXPOSURE CHECK
    if not df_ops_train.empty and "nmbr_active_leads" in df_ops_train.columns and "expected_daily_slots" in df_ops_train.columns:
        exposure_check = df_ops_train[["partner_id", "nmbr_active_leads", "expected_daily_slots"]].copy()
        exposure_check["exposure_ratio"] = (
            exposure_check["nmbr_active_leads"]
            / exposure_check["expected_daily_slots"].replace(0, np.nan)
        ).fillna(0)
        exposure_check["e_concentrated"] = (
            exposure_check["exposure_ratio"] > sc.EXPOSURE_CONCENTRATION_FACTOR
        ).astype(int)
        tier_downgrade = {"HIGH": "MOD", "MOD": "LOW", "LOW": "DECLINE"}

    # ══════════════════════════════════════════════════════════════
    # 6. Evaluation / Bucketing
    # ══════════════════════════════════════════════════════════════
    print("\n--- RESULTS: SEPARATION ANALYSIS ---")

    results = []
    results.append(evaluate_bucket(df[df["parent_color_super"] == "lightgreen"], "Green Hex"))
    results.append(evaluate_bucket(df[df["parent_color_super"] == "orange"], "Orange Hex"))
    results.append(evaluate_bucket(df[df["parent_color_super"] == "crimson"], "Crimson Hex"))
    results.append(evaluate_bucket(
        df[(df["parent_color_super"] == "lightgreen") & (df["parent_overlap"] == True)],
        "Green + Contested",
    ))
    results.append(evaluate_bucket(
        df[(df["parent_color_super"] == "lightgreen") & (df["parent_overlap"] == False)],
        "Green + Solo",
    ))
    results.append(evaluate_bucket(
        df[(df["parent_color_super"] == "lightgreen") & (df["is_sparse"] == 0)],
        "Green + Dense History",
    ))
    results.append(evaluate_bucket(
        df[(df["parent_color_super"] == "lightgreen") & (df["is_sparse"] == 1)],
        "Green + Sparse History",
    ))
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    # ── Temporal SE separation check ──
    print("\n--- TEMPORAL FEATURE SEPARATION ---")
    for wd in config.TEMPORAL_WINDOWS:
        for col in [f"weighted_se_{wd}d", f"predicted_field_hex_{wd}d", f"norm_se_{wd}d"]:
            if col in df.columns:
                valid = df[col].notna()
                if valid.any():
                    p10 = df.loc[valid, col].quantile(0.1)
                    p90 = df.loc[valid, col].quantile(0.9)
                    print(f"  {col:>35s} — p10={p10:.4f}  p90={p90:.4f}  gap={p90 - p10:.4f}")

    # Boundary proximity analysis
    df["boundary_dist_bucket"] = "INSIDE"
    outside_mask = df["nearest_boundary_dist_m"].notna()
    bins_dist = [0, 20, 40, 60, 100, np.inf]
    labels_dist = ["0-20", "20-40", "40-60", "60-100", "100+"]
    boundary_bins = pd.cut(
        df.loc[outside_mask, "nearest_boundary_dist_m"],
        bins=bins_dist, labels=labels_dist, right=True, include_lowest=True,
    )
    df.loc[outside_mask, "boundary_dist_bucket"] = pd.Series(
        boundary_bins, index=df.index[outside_mask]
    ).astype(str)

    dist_fsb = (
        df.groupby(["min_dist_bucket", "final_serviceability", "boundary_dist_bucket"], dropna=False)
        .agg(total=("mobile", "count"), installs=("installed_decision", "sum"))
        .reset_index()
    )
    dist_fsb["install_rate"] = dist_fsb["installs"] / dist_fsb["total"]
    dist_fsb_path = os.path.join(REPORTS_DIR, "install_rate_by_min_dist_bucket_final_serviceability_distance_from_boundary.csv")
    dist_fsb.to_csv(dist_fsb_path, index=False)

    dist_count = (
        df.groupby(["min_dist_bucket", "final_serviceability", "nmbr_boundaries_within_100m"], dropna=False)
        .agg(total=("mobile", "count"), installs=("installed_decision", "sum"))
        .reset_index()
    )
    dist_count["install_rate"] = dist_count["installs"] / dist_count["total"]
    path_boundary_count = os.path.join(REPORTS_DIR, "install_rate_by_min_dist_bucket_final_serviceability_boundary_count.csv")
    dist_count.to_csv(path_boundary_count, index=False)

    dist_fs = (
        df.groupby(["min_dist_bucket", "final_serviceability"], dropna=False)
        .agg(total=("mobile", "count"), installs=("installed_decision", "sum"))
        .reset_index()
    )
    dist_fs["install_rate"] = dist_fs["installs"] / dist_fs["total"]
    dist_fs_path = os.path.join(REPORTS_DIR, "install_rate_by_min_dist_bucket_and_final_serviceability.csv")
    dist_fs.to_csv(dist_fs_path, index=False)

    dist_regime = (
        df.groupby(["min_dist_bucket", "geometry_regime", "final_serviceability"], dropna=False)
        .agg(total=("mobile", "count"), installs=("installed_decision", "sum"))
        .reset_index()
    )
    dist_regime["install_rate"] = dist_regime["installs"] / dist_regime["total"]
    dist_regime_path = os.path.join(REPORTS_DIR, "install_rate_by_min_dist_bucket_and_geometry_regime.csv")
    dist_regime.to_csv(dist_regime_path, index=False)

    if simulate:
        run_declines_simulation(df_train, df_poly, df_bound, g1_start, g1_end, REPORTS_DIR)

    # Save
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

    if not df_ops_train.empty:
        df_ops_train.to_csv(os.path.join(REPORTS_DIR, "partner_ops_train_vector.csv"), index=False)
    if not df_ops_test.empty:
        df_ops_test.to_csv(os.path.join(REPORTS_DIR, "partner_ops_test_vector.csv"), index=False)

    print(f"\nDetailed scores saved to {report_path}")
    print(f"Distance x FS summary saved to {dist_fs_path}")
    print(f"Distance x FS x regime summary saved to {dist_regime_path}")

    # ── Summary of all temporal columns in final output ──
    temporal_cols = [c for c in df.columns if any(f"_{wd}d" in c for wd in config.TEMPORAL_WINDOWS)]
    print(f"\n[TEMPORAL] {len(temporal_cols)} temporal columns in final output:")
    for c in sorted(temporal_cols):
        n_valid = df[c].notna().sum() if df[c].dtype != object else "N/A"
        print(f"  {c}: {n_valid}/{len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test scoring (playground)")
    parser.add_argument("--simulate", action="store_true", help="Run declines simulation")
    args = parser.parse_args()
    main(simulate=args.simulate)
