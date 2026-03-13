# playground/geometric_features.py

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.special import expit
import data_lib.config as config


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

    lat0_rad = np.radians(lead_lat)
    meters_per_deg_lat = config.METERS_PER_DEG_LAT
    meters_per_deg_lon = meters_per_deg_lat * np.cos(lat0_rad)

    coords_m = np.zeros((len(neighbor_df), 2))
    coords_m[:, 0] = (neighbor_df["longitude"] - lead_lon) * meters_per_deg_lon  # x
    coords_m[:, 1] = (neighbor_df["latitude"] - lead_lat) * meters_per_deg_lat  # y

    # 1. Covariance & Eigenvalues (Anisotropy)
    center_m = np.mean(coords_m, axis=0)
    centered_coords = coords_m - center_m

    cov_matrix = np.cov(centered_coords, rowvar=False)
    eig_vals = np.linalg.eigvals(cov_matrix)
    eig_vals = np.sort(eig_vals)[::-1]  # Descending

    if eig_vals[0] > 0:
        anisotropy = 1.0 - (eig_vals[1] / eig_vals[0])
    else:
        anisotropy = 0.0

    # 2. Convex Hull Density
    try:
        hull = ConvexHull(coords_m)
        hull_area = hull.volume  # In 2D, volume is area
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
        "linearity_score": round(linearity, 1),
        "spread_m": round(spread, 1),
        "dense_score": round(dense_score, 4),
        "gully_score": round(gully_score, 4),
        "sparse_score": round(sparse_score, 4),
    }


# Keys we extract for temporal variants (subset of full geometry output)
_TEMPORAL_GEOM_KEYS = [
    "local_anisotropy", "local_density", "hull_area", "spread_m",
    "dense_score", "gully_score", "sparse_score",
]


def batch_compute_geometry(df_leads, df_history, radius_m=250, reference_date=None):
    """
    Batched wrapper for computing geometry for many leads.
    Uses BallTree for efficiency.

    If reference_date is provided and df_history has decision_time,
    also computes temporal geometry for each window in TEMPORAL_WINDOWS.

    ALL-TIME columns produced:
        local_anisotropy, local_density, hull_area, linearity_score, spread_m,
        dense_score, gully_score, sparse_score

    TEMPORAL columns produced (per window wd in TEMPORAL_WINDOWS):
        local_anisotropy_{wd}d, local_density_{wd}d, hull_area_{wd}d,
        spread_m_{wd}d, dense_score_{wd}d, gully_score_{wd}d, sparse_score_{wd}d
    """
    from sklearn.neighbors import BallTree

    # ══════════════════════════════════════════════════════════════
    # ALL-TIME GEOMETRY
    # ══════════════════════════════════════════════════════════════

    hist_rad = np.radians(df_history[["latitude", "longitude"]].values)
    tree = BallTree(hist_rad, metric="haversine")

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

        neighbors = df_history.iloc[indices]
        lead_lat = df_leads.iloc[i]["latitude"]
        lead_lon = df_leads.iloc[i]["longitude"]

        metrics = compute_local_geometry(lead_lat, lead_lon, neighbors, radius_m)
        results.append(metrics)

    # ══════════════════════════════════════════════════════════════
    # TEMPORAL GEOMETRY
    # ══════════════════════════════════════════════════════════════

    has_temporal = (
        reference_date is not None
        and "decision_time" in df_history.columns
    )

    if has_temporal:
        ref_ts = pd.Timestamp(reference_date)
        print(f"[GEOMETRY] Computing temporal geometry for windows: {config.TEMPORAL_WINDOWS}")

        for wd in config.TEMPORAL_WINDOWS:
            cutoff = ref_ts - pd.Timedelta(days=wd)
            hist_w = df_history[df_history["decision_time"] >= cutoff]

            if len(hist_w) < 3:
                # Not enough recent history — fill NaN for all leads
                for i in range(len(results)):
                    for k in _TEMPORAL_GEOM_KEYS:
                        results[i][f"{k}_{wd}d"] = np.nan
                print(f"[GEOMETRY]   {wd}d: only {len(hist_w)} points, all NaN")
                continue

            hist_w_rad = np.radians(hist_w[["latitude", "longitude"]].values)
            tree_w = BallTree(hist_w_rad, metric="haversine")
            idx_w = tree_w.query_radius(leads_rad, r=radius_rad)

            n_valid = 0
            for i, indices in enumerate(idx_w):
                if len(indices) < 3:
                    for k in _TEMPORAL_GEOM_KEYS:
                        results[i][f"{k}_{wd}d"] = np.nan
                    continue

                neighbors = hist_w.iloc[indices]
                m_w = compute_local_geometry(
                    df_leads.iloc[i]["latitude"],
                    df_leads.iloc[i]["longitude"],
                    neighbors,
                    radius_m,
                )
                for k in _TEMPORAL_GEOM_KEYS:
                    results[i][f"{k}_{wd}d"] = m_w.get(k, np.nan)
                n_valid += 1

            print(f"[GEOMETRY]   {wd}d: {n_valid}/{len(results)} leads with valid geometry")

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

    df_train = df_train.copy()

    unique_pts = df_train.groupby(
        ["mobile", "latitude", "longitude"], as_index=False
    ).agg({"decision_time": "min", "field_weight": "max"})

    unique_pts["adaptive_h"] = np.nan

    for subset_mask, subset_name in [
        (unique_pts["field_weight"] >= 0, "install"),
        (unique_pts["field_weight"] < 0, "decline"),
    ]:
        if not subset_mask.any():
            continue

        subset_df = unique_pts[subset_mask]

        target_k = config.ADAPTIVE_H_NEIGHBOR_K
        query_k = target_k + 1

        if len(subset_df) < query_k:
            continue

        coords = np.radians(subset_df[["latitude", "longitude"]].values)
        tree = BallTree(coords, metric="haversine")

        dist, _ = tree.query(coords, k=query_k)

        neighbor_dist_rad = dist[:, target_k]
        neighbor_dist_m = neighbor_dist_rad * config.EARTH_RADIUS_METER

        neighbor_dist_m = np.clip(
            neighbor_dist_m, config.ADAPTIVE_H_MIN, config.ADAPTIVE_H_MAX
        )

        unique_pts.loc[subset_mask, "adaptive_h"] = neighbor_dist_m

    merge_cols = ["mobile", "latitude", "longitude", "adaptive_h"]

    df_merged = df_train.merge(
        unique_pts[merge_cols], on=["mobile", "latitude", "longitude"], how="left"
    )

    df_merged["h"] = df_merged["adaptive_h"].fillna(df_merged["h"])
    df_merged = df_merged.drop(columns=["adaptive_h"])

    return df_merged
