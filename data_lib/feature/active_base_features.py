import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from data_lib.config import EARTH_RADIUS_METER

def compute_active_base_features(
    df_test: pd.DataFrame,
    df_active: pd.DataFrame,
    radii_m: list = [50, 100, 200, 500],
) -> pd.DataFrame:
    """
    Per test lead: count nearby active customers, distance to nearest,
    partner-specific density, customer recency.

    Returns one row per mobile in df_test.
    """
    if df_active.empty:
        out = pd.DataFrame({"mobile": df_test["mobile"]})
        for r in radii_m:
            out[f"ab_count_{r}m"] = 0
        out["ab_nearest_dist_m"] = np.nan
        out["ab_partner_count_200m"] = 0
        out["ab_partner_frac_200m"] = np.nan
        return out

    # BallTree on all active base points
    ab_rad = np.radians(df_active[["latitude", "longitude"]].values)
    tree = BallTree(ab_rad, metric="haversine")

    test_rad = np.radians(df_test[["latitude", "longitude"]].values)

    # Pre-query at max radius
    max_r = max(radii_m)
    max_r_rad = max_r / EARTH_RADIUS_METER
    idxs_all, dists_all = tree.query_radius(
        test_rad, r=max_r_rad, return_distance=True
    )

    # Nearest neighbor (k=1)
    dist_nearest, _ = tree.query(test_rad, k=1)
    nearest_m = dist_nearest[:, 0] * EARTH_RADIUS_METER

    # Get partner_id for each test lead (for partner-specific count)
    test_pids = df_test["partner_id"].values if "partner_id" in df_test.columns else None
    ab_pids = df_active["partner_id"].values

    rows = []
    for i in range(len(df_test)):
        row = {"mobile": df_test.iloc[i]["mobile"]}

        idxs = idxs_all[i]
        dists_m = dists_all[i] * EARTH_RADIUS_METER

        # Count within each radius
        for r in radii_m:
            row[f"ab_count_{r}m"] = int((dists_m <= r).sum())

        row["ab_nearest_dist_m"] = round(float(nearest_m[i]), 1)

        # Partner-specific: how many of MY partner's customers are nearby
        if test_pids is not None and len(idxs) > 0:
            pid = test_pids[i]
            mask_200 = dists_m <= 200
            neighbor_pids = ab_pids[idxs[mask_200]]
            partner_count = int((neighbor_pids == pid).sum())
            total_200 = int(mask_200.sum())
            row["ab_partner_count_200m"] = partner_count
            row["ab_partner_frac_200m"] = (
                round(partner_count / total_200, 4) if total_200 > 0 else np.nan
            )
        else:
            row["ab_partner_count_200m"] = 0
            row["ab_partner_frac_200m"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)