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

Temporal variants (per window in TEMPORAL_WINDOWS):
    hop{1,2,3}_se_{wd}d_wmean
    se_gradient_1to3_{wd}d
    se_confirmed_{wd}d
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from data_lib.geometry.distance import haversine_rad
from data_lib.config import EARTH_RADIUS_KILOMETER, TEMPORAL_WINDOWS


def compute_hop_features(df_poly: pd.DataFrame, n_hops: int = 3) -> pd.DataFrame:
    """
    For every non-empty hex in df_poly, compute distance-weighted
    neighbor SE aggregates at hop 1, 2, 3 — all-time + per temporal window.

    Parameters
    ----------
    df_poly : DataFrame
        Must contain: partner_id, poly_id, se, total, best_size, poly (Shapely).
        May contain: se_30d, se_60d, se_365d for temporal hop features.
    n_hops : int
        Number of hop rings (default 3).

    Returns
    -------
    DataFrame with columns:
        partner_id, poly_id,
        hop{1..3}_se_wmean, hop{1..3}_se_std, hop{1..3}_count,
        se_gradient_1to3, se_confirmed, isolation_ratio,
        hop{1..3}_se_{wd}d_wmean,
        se_gradient_1to3_{wd}d, se_confirmed_{wd}d
    """
    available_windows = [wd for wd in TEMPORAL_WINDOWS if f"se_{wd}d" in df_poly.columns]

    # ── Extract centroids once ──
    keep_cols = ["partner_id", "poly_id", "se", "total", "best_size", "poly"]
    keep_cols += [f"se_{wd}d" for wd in available_windows]
    df = df_poly[keep_cols].copy()

    df["clat"] = df["poly"].apply(lambda p: p.centroid.y if p else np.nan)
    df["clon"] = df["poly"].apply(lambda p: p.centroid.x if p else np.nan)
    df = df.dropna(subset=["clat", "clon", "se"])

    all_results = []

    for pid, grp in df.groupby("partner_id"):

        # ──────────────────────────────────────────────
        # SOLO HEX — all features NaN
        # ──────────────────────────────────────────────
        if len(grp) < 2:
            row = {"partner_id": pid, "poly_id": grp["poly_id"].iloc[0]}
            for h in range(1, n_hops + 1):
                row[f"hop{h}_se_wmean"] = np.nan
                row[f"hop{h}_se_std"] = np.nan
                row[f"hop{h}_count"] = 0
                for wd in available_windows:
                    row[f"hop{h}_se_{wd}d_wmean"] = np.nan
            row["se_gradient_1to3"] = np.nan
            row["se_confirmed"] = np.nan
            row["isolation_ratio"] = np.nan
            for wd in available_windows:
                row[f"se_gradient_1to3_{wd}d"] = np.nan
                row[f"se_confirmed_{wd}d"] = np.nan
            all_results.append(row)
            continue

        # ──────────────────────────────────────────────
        # MULTI-HEX PARTNER — build tree, compute hops
        # ──────────────────────────────────────────────
        grp = grp.reset_index(drop=True)
        coords_rad = np.radians(grp[["clat", "clon"]].values)
        se_arr = grp["se"].values

        # Pre-extract temporal SE arrays
        temporal_se = {}
        for wd in available_windows:
            temporal_se[wd] = grp[f"se_{wd}d"].values

        best_size_km = grp["best_size"].iloc[0]
        tree = BallTree(coords_rad, metric="haversine")

        # Hop radii: centroid-to-centroid of adjacent hexes
        step_km = best_size_km * 2.0

        for i in range(len(grp)):
            row = {
                "partner_id": pid,
                "poly_id": grp["poly_id"].iloc[i],
            }

            prev_indices = {i}  # exclude self
            hop_wmeans = []
            hop_wmeans_temporal = {wd: [] for wd in available_windows}

            for h in range(1, n_hops + 1):
                outer_r_km = step_km * h + step_km * 0.3
                outer_r_rad = outer_r_km / EARTH_RADIUS_KILOMETER

                idxs_outer = tree.query_radius(
                    coords_rad[i].reshape(1, -1), r=outer_r_rad
                )[0]

                # Ring = outer minus already-counted (inner hops + self)
                ring_idxs = np.array([j for j in idxs_outer if j not in prev_indices])

                # ── EMPTY RING ──
                if len(ring_idxs) == 0:
                    row[f"hop{h}_se_wmean"] = np.nan
                    row[f"hop{h}_se_std"] = np.nan
                    row[f"hop{h}_count"] = 0
                    hop_wmeans.append(np.nan)
                    for wd in available_windows:
                        row[f"hop{h}_se_{wd}d_wmean"] = np.nan
                        hop_wmeans_temporal[wd].append(np.nan)
                    continue

                # ── COMPUTE DISTANCES + IDW WEIGHTS ──
                ring_coords = coords_rad[ring_idxs]
                dists_rad = np.array([
                    haversine_rad(coords_rad[i], ring_coords[k])
                    for k in range(len(ring_idxs))
                ])
                dists_km = dists_rad * EARTH_RADIUS_KILOMETER
                dists_km = np.maximum(dists_km, 1e-6)

                weights = 1.0 / (h * dists_km)

                # ── ALL-TIME SE ──
                ring_se = se_arr[ring_idxs]
                wmean = np.average(ring_se, weights=weights)
                wvar = np.average((ring_se - wmean) ** 2, weights=weights)
                wstd = np.sqrt(wvar)

                row[f"hop{h}_se_wmean"] = round(wmean, 6)
                row[f"hop{h}_se_std"] = round(wstd, 6)
                row[f"hop{h}_count"] = len(ring_idxs)
                hop_wmeans.append(wmean)

                # ── TEMPORAL SE ──
                for wd, se_t_arr in temporal_se.items():
                    ring_se_t = se_t_arr[ring_idxs]
                    valid_t = ~np.isnan(ring_se_t)
                    if valid_t.any():
                        wmean_t = np.average(ring_se_t[valid_t], weights=weights[valid_t])
                        row[f"hop{h}_se_{wd}d_wmean"] = round(wmean_t, 6)
                        hop_wmeans_temporal[wd].append(wmean_t)
                    else:
                        row[f"hop{h}_se_{wd}d_wmean"] = np.nan
                        hop_wmeans_temporal[wd].append(np.nan)

                # Expand prev_indices for next hop ring
                prev_indices.update(ring_idxs.tolist())

            # ── CROSS-HOP INTERACTIONS: ALL-TIME ──
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

            # ── CROSS-HOP INTERACTIONS: TEMPORAL ──
            for wd in available_windows:
                wm = hop_wmeans_temporal[wd]
                h1_t = wm[0] if len(wm) > 0 else np.nan
                h3_t = wm[2] if len(wm) > 2 else np.nan
                if not np.isnan(h1_t) and not np.isnan(h3_t):
                    row[f"se_gradient_1to3_{wd}d"] = round(h1_t - h3_t, 6)
                    row[f"se_confirmed_{wd}d"] = round(h1_t * h3_t, 6)
                else:
                    row[f"se_gradient_1to3_{wd}d"] = np.nan
                    row[f"se_confirmed_{wd}d"] = np.nan

            all_results.append(row)

    return pd.DataFrame(all_results)
