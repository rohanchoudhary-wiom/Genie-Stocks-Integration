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
