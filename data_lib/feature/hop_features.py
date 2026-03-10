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
from data_lib.geometry.distance import haversine_rad
from config import EARTH_RADIUS_KILOMETER


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



