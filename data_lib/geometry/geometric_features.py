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
