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
