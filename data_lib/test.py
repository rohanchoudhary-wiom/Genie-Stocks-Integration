# ~/Apps/maanasbot/portfolio/genie_ml/matchmaking/self_learning/revamp/new_code/test.py
# FINAL, CORRECT & PRODUCTION-READY VERSION — December 2025

import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.strtree import STRtree
from data_lib.params import overlap_columns
from data_lib.config import TEMPORAL_WINDOWS

def get_overlap(search_radius_deg: float = 0.027):
    print("Loading data...")
    df_hex = pd.read_hdf("artifacts/poly_stats.h5", "df")
    df_bound = pd.read_hdf("artifacts/partner_cluster_boundaries.h5", "df")

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
        "artifacts/poly_stats_updated.h5", key="df", mode="w", complevel=9, complib="blosc"
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
            "artifacts/crimson_ranks.h5", key="df", mode="w", complevel=9, complib="blosc"
        )
        df_crimson.to_csv("artifacts/crimson_ranks.csv", index=False)
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
    # NEW:
    temporal_cols = ["install_velocity"]
    for wd in TEMPORAL_WINDOWS:
        temporal_cols += [f"se_{wd}d", f"installs_{wd}d", f"declines_{wd}d", f"total_{wd}d"]

    final_columns = overlap_columns + temporal_cols
    df_final = df_final[[c for c in final_columns if c in df_final.columns]]

    # Save
    for col in df_final.select_dtypes(include="string").columns:
        df_final[col] = df_final[col].astype("object")
    df_final.to_hdf(
        "artifacts/poly_stats_final.h5", key="df", mode="w", complevel=9, complib="blosc"
    )

    return df_final


# get_overlap()
