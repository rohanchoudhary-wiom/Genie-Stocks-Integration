from tqdm.auto import tqdm

from sklearn.neighbors import BallTree
from data_lib.geometry.distance import equiv_radius_m, haversine_m
import numpy as np
import pandas as pd

from data_lib.feature.hop_features import compute_hop_features
from pyproj import Transformer

import geopandas as gpd
from shapely.geometry import Point
from shapely import contains_xy

from functools import reduce
import operator

import data_lib.config as config
import data_lib.params as params

# GLOBAL CONFIG — NEVER CHANGE PER PARTNER
DEFAULT_LAMBDA = 0.005
best_w_neg = -2.0
min_decisions = 250

min_samples = 50
EARTH_RADIUS_M = 6371000  # metres (standard for most geospatial work)

COLOR_NUMERIC_MAP = {
    "lightgreen": 3,
    "orange": 2,
    "crimson": 1,
    "indeterminate": 2,
}


def compute_adaptive_gaussian_field(
    df_target: pd.DataFrame,
    df_source: pd.DataFrame,
    lambda_decay: float = DEFAULT_LAMBDA,
    max_radius_m: float = 15000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adaptive Gaussian KDE with exponential time decay, causal.
    Uses BallTree radius query → O(1) memory.
    Returns `kernel_sum` — the denominator Σ(k_i) — as per-target confidence weight.
    """
    if "decision_time" in df_target.columns:
        eval_times = df_target["decision_time"].values.astype("datetime64[ns]")
    else:
        eval_times = np.full(len(df_target), np.datetime64("now"))

    required_source = ["latitude", "longitude", "field_weight", "h", "decision_time"]
    df_source = df_source[required_source].copy()

    if len(df_source) == 0:
        df_target["predicted_field"] = np.nan
        df_target["kernel_sum"] = 0.0
        return df_target

    source_coords_rad = np.radians(df_source[["latitude", "longitude"]].values)
    tree = BallTree(source_coords_rad, metric="haversine")

    source_times = df_source["decision_time"].values.astype("datetime64[ns]")
    source_h = df_source["h"].values
    source_weights = df_source["field_weight"].values

    target_coords_rad = np.radians(df_target[["latitude", "longitude"]].values)
    radius_rad = max_radius_m / EARTH_RADIUS_M

    predicted = np.zeros(len(df_target))
    kernel_sums = np.zeros(len(df_target))

    iterator = (
        tqdm(
            enumerate(zip(target_coords_rad, eval_times)),
            total=len(df_target),
            desc=f"Gaussian field ({len(df_target)} targets)",
        )
        if verbose
        else enumerate(zip(target_coords_rad, eval_times))
    )

    for i, (target_coord, eval_time) in iterator:
        indices, dist_rad = tree.query_radius(
            target_coord.reshape(1, -1),
            r=radius_rad,
            return_distance=True,
            sort_results=False,
        )
        indices = indices[0]
        dist_m = dist_rad[0] * EARTH_RADIUS_M

        if len(indices) == 0:
            predicted[i] = np.nan
            kernel_sums[i] = 0.0
            continue

        neigh_times = source_times[indices]
        neigh_h = source_h[indices]
        neigh_w = source_weights[indices]

        ages_days = (eval_time - neigh_times) / np.timedelta64(1, "D")
        valid_mask = ~np.isnan(ages_days)
        past_mask = valid_mask & (ages_days >= 0)
        time_weights = np.zeros_like(ages_days, dtype=float)
        time_weights[past_mask] = np.exp(-lambda_decay * ages_days[past_mask])

        effective_weights = neigh_w * time_weights
        kernel = np.exp(-0.5 * (dist_m**2) / (neigh_h**2))

        weighted_sum = np.sum(kernel * effective_weights)
        norm = np.sum(kernel)

        predicted[i] = weighted_sum / (norm + 1e-12)
        kernel_sums[i] = norm

    df_target["predicted_field"] = predicted
    df_target["kernel_sum"] = kernel_sums
    return df_target


def compute_contested_metrics_engine(
    joined_gdf_4326: gpd.GeoDataFrame,
    df_target: pd.DataFrame,
    gdf_source_4326: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Computes contested overlap geometry (cached) + per-mobile field.

    REQUIRED COLUMNS
    ----------------
    joined_gdf_4326 : mobile, boundary_poly_keep (Polygon, EPSG:4326)
    df_target       : mobile, latitude, longitude, decision_time
    gdf_source_4326 : latitude, longitude, decision_time, field_weight, h, installed_decision

    RETURNS
    -------
    DataFrame indexed by mobile with contested metrics.
    """
    sidx = gdf_source_4326.sindex

    to_4326 = Transformer.from_crs("EPSG:7755", "EPSG:4326", always_xy=True)

    poly_cache = {}
    zonal_cache = {}
    results = []

    for mobile, grp in tqdm(
        joined_gdf_4326.groupby("mobile", sort=False),
        total=joined_gdf_4326["mobile"].nunique(),
    ):
        polys = [p for p in grp["boundary_poly_keep"].tolist() if p is not None]

        base = {
            "mobile": mobile,
            "n_overlapping_partners": len(polys),
            "contested_area_km2": np.nan,
            "contested_radius_m": np.nan,
            "contested_centroid_lat": np.nan,
            "contested_centroid_lon": np.nan,
            "contested_installs": np.nan,
            "contested_total": np.nan,
            "contested_se": np.nan,
            "contested_field": np.nan,
        }

        if len(polys) == 0:
            results.append(base)
            continue

        tr = df_target.loc[df_target["mobile"] == mobile].iloc[0]
        lat0, lon0 = float(tr["latitude"]), float(tr["longitude"])
        t0 = pd.to_datetime(tr["decision_time"])
        pt = Point(lon0, lat0)

        sig = tuple(sorted(p.wkb for p in polys))

        if sig not in poly_cache:
            cp = polys[0] if len(polys) == 1 else reduce(operator.and_, polys)

            if cp is None or cp.is_empty or cp.geom_type == "GeometryCollection":
                poly_cache[sig] = None
                zonal_cache[sig] = None
                results.append(base)
                continue
            else:
                if cp.geom_type == "MultiPolygon":
                    candidates = [g for g in cp.geoms if g.contains(pt) or g.intersects(pt)]
                    cp = max(candidates, key=lambda g: g.area) if candidates else max(cp.geoms, key=lambda g: g.area)

                poly_cache[sig] = cp

                cp_proj = gpd.GeoSeries([cp], crs="EPSG:4326").to_crs("EPSG:7755").iloc[0]
                area_m2 = float(cp_proj.area)
                area_km2 = area_m2 / 1e6
                radius_m = equiv_radius_m(area_m2)

                rp = cp_proj.representative_point()
                rp_lon, rp_lat = to_4326.transform(rp.x, rp.y)

                minx, miny, maxx, maxy = cp.bounds
                cand_idx = list(sidx.intersection((minx, miny, maxx, maxy)))

                if len(cand_idx) == 0:
                    inside = gdf_source_4326.iloc[[]]
                else:
                    cand = gdf_source_4326.iloc[cand_idx]
                    mask = contains_xy(cp, cand["longitude"].to_numpy(), cand["latitude"].to_numpy())
                    inside = cand.loc[mask]

                zonal_cache[sig] = {
                    "area_km2": area_km2,
                    "radius_m": radius_m,
                    "centroid_lat": rp_lat,
                    "centroid_lon": rp_lon,
                    "inside": inside,
                }

        if poly_cache[sig] is None:
            results.append(base)
            continue

        zonal = zonal_cache[sig]
        inside = zonal["inside"]

        n_total = len(inside)
        n_installs = int(inside["installed_decision"].sum()) if n_total > 0 else 0
        se = (n_installs / n_total) if n_total > 0 else np.nan

        base.update(
            {
                "contested_area_km2": round(zonal["area_km2"], 4),
                "contested_radius_m": round(zonal["radius_m"], 1),
                "contested_centroid_lat": zonal["centroid_lat"],
                "contested_centroid_lon": zonal["centroid_lon"],
                "contested_installs": n_installs,
                "contested_total": n_total,
                "contested_se": round(se, 4) if not np.isnan(se) else np.nan,
            }
        )

        if n_total > 0:
            lat = inside["latitude"].to_numpy()
            lon = inside["longitude"].to_numpy()

            dlat = np.radians(lat - lat0)
            dlon = np.radians(lon - lon0)
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(np.radians(lat0)) * np.cos(np.radians(lat)) * np.sin(dlon / 2) ** 2
            )
            d = 2 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            h = inside["h"].to_numpy(dtype=float)
            w = inside["field_weight"].to_numpy(dtype=float)

            dt_days = (
                t0 - pd.to_datetime(inside["decision_time"])
            ).dt.total_seconds().to_numpy() / 86400.0
            valid_mask = ~np.isnan(dt_days)
            past_mask = valid_mask & (dt_days >= 0)
            time_decay = np.zeros_like(dt_days, dtype=float)
            time_decay[past_mask] = np.exp(-config.LAMBDA_DECAY * dt_days[past_mask])

            eff_w = w * time_decay
            k = np.exp(-0.5 * (d**2) / (h**2 + 1e-12))
            base["contested_field"] = float(np.sum(k * eff_w) / (np.sum(k) + 1e-12))

        results.append(base)

    return pd.DataFrame(results)


def add_boundary_details_precise(df_target, df_bound, df_source):
    geometry_points = [
        Point(lon, lat)
        for lon, lat in zip(df_target["longitude"], df_target["latitude"])
    ]

    gdf_target = gpd.GeoDataFrame(df_target.copy(), geometry=geometry_points, crs="EPSG:4326")
    gdf_bound = gpd.GeoDataFrame(df_bound.copy(), geometry="boundary_poly", crs="EPSG:4326")
    gdf_bound["boundary_poly_keep"] = gdf_bound["boundary_poly"]

    joined_gdf_4326 = gpd.sjoin(
        gdf_target, gdf_bound, how="left", predicate="within"
    ).reset_index(drop=True)

    joined_gdf = joined_gdf_4326.to_crs(7755)

    poly_keep_proj = gpd.GeoSeries(joined_gdf["boundary_poly_keep"], crs="EPSG:4326").to_crs(7755)

    joined_gdf["dist_to_boundary_edge_m"] = poly_keep_proj.boundary.distance(joined_gdf.geometry)
    joined_gdf["dist_to_boundary_edge_m"] = joined_gdf["dist_to_boundary_edge_m"].where(
        joined_gdf["boundary_poly_keep"].notna(), np.nan
    )

    outside_mask = joined_gdf["boundary_poly_keep"].isna()

    joined_gdf["dist_to_nearest_boundary_m"] = np.nan
    joined_gdf["nmbr_boundaries_within_100m"] = np.nan

    if outside_mask.any():
        joined_gdf.loc[outside_mask, "nmbr_boundaries_within_100m"] = 0

        gdf_outside = joined_gdf.loc[outside_mask].copy()
        gdf_bound_proj = gdf_bound.to_crs(7755)

        try:
            bound_geom_col = gdf_bound_proj.geometry.name
            nearest = gpd.sjoin_nearest(
                gdf_outside[["mobile", "geometry"]],
                gdf_bound_proj[[bound_geom_col]],
                how="left",
                distance_col="dist_temp",
            )
            dist_map = nearest.groupby(level=0)["dist_temp"].min()
            joined_gdf.loc[dist_map.index, "dist_to_nearest_boundary_m"] = dist_map
        except AttributeError:
            pass

        buf = gdf_outside[["mobile", "geometry"]].copy()
        buf["geometry"] = buf.geometry.buffer(100)

        bound_geom_col = gdf_bound_proj.geometry.name
        within_100 = gpd.sjoin(buf, gdf_bound_proj[[bound_geom_col]], predicate="intersects")

        if not within_100.empty:
            count_map = within_100.groupby("mobile").size()
            joined_gdf.loc[outside_mask, "nmbr_boundaries_within_100m"] += (
                joined_gdf.loc[outside_mask, "mobile"].map(count_map).fillna(0)
            )

    df_joined = pd.DataFrame(joined_gdf.drop(columns="geometry"))

    df_joined["dist_to_cluster_center_m"] = haversine_m(
        df_joined["latitude"], df_joined["longitude"],
        df_joined["center_lat"], df_joined["center_lon"],
    )

    df = df_joined[params.one_to_many_cols].copy()

    df["near_edge"] = np.where(df["dist_to_boundary_edge_m"] < 0.7 * df["dist_to_cluster_center_m"], 1, 0)
    df["depth_score"] = df["dist_to_cluster_center_m"] - df["dist_to_boundary_edge_m"]
    df["is_solo_cluster"] = np.where(df["cluster_type"] == "p90_single_cluster", 1, 0)

    df_summary = (
        df.groupby(["mobile"])
        .agg(
            nmbr_overlap_clusters=("cluster_id", "count"),
            mean_dist_to_edge_m=("dist_to_boundary_edge_m", "mean"),
            near_edge_instances=("near_edge", "sum"),
            mean_dist_to_center_m=("dist_to_cluster_center_m", "mean"),
            total_area_boundaries=("area_km2", "sum"),
            n_hexes=("n_hexes", "sum"),
            nmbr_partners=("partner_id", "nunique"),
            worst_depth_score=("depth_score", "max"),
            any_near_edge=("near_edge", "max"),
            is_solo_cluster=("is_solo_cluster", "max"),
            nearest_boundary_dist_m=("dist_to_nearest_boundary_m", "min"),
            nmbr_boundaries_within_100m=("nmbr_boundaries_within_100m", "max"),
        )
        .reset_index()
    )

    if df_source is not None and len(df_source) > 0:
        gdf_source = gpd.GeoDataFrame(
            df_source.copy(),
            geometry=gpd.points_from_xy(df_source.longitude, df_source.latitude),
            crs="EPSG:4326",
        )

        if "installed_decision" not in gdf_source.columns:
            gdf_source["installed_decision"] = (gdf_source["final_decision"] == "INSTALLED").astype(int)

        joined_gdf_for_contested = joined_gdf_4326[["mobile", "boundary_poly_keep"]].copy()

        df_contested = compute_contested_metrics_engine(
            joined_gdf_4326=joined_gdf_for_contested,
            df_target=df_target,
            gdf_source_4326=gdf_source,
        )

        df_summary = df_summary.merge(df_contested, on="mobile", how="left")

    return df_summary


# ==============================================================
# GET PARENT HEXAGON — mobile-level consensus across all hexes
# ==============================================================

def compute_hex_consensus_features(df_target: pd.DataFrame, df_poly: pd.DataFrame) -> pd.DataFrame:
    """
    Spatial-joins every target to ALL covering hexagons (not just the best one).
    Per hex row: adjusts color for indeterminate, shrinks SE, computes geometric
    features and hop features. Then aggregates everything to mobile level via
    evidence-weighted consensus.

    Returns a DataFrame at mobile level with:
      - weighted_se, weighted_se_shrunk, parent_color
      - median geometric features (dist_to_edge, dist_to_center, near_edge, depth_score)
      - total-weighted hop features
      - n_covering_partners, total, installs, declines
    """

    # ── Hex centroids ──
    df_poly["center_lat"] = pd.to_numeric(
        df_poly["poly"].apply(lambda p: p.centroid.y if p is not None else np.nan),
        errors="coerce",
    )
    df_poly["center_lon"] = pd.to_numeric(
        df_poly["poly"].apply(lambda p: p.centroid.x if p is not None else np.nan),
        errors="coerce",
    )

    # ── Spatial join: targets × all hexes ──
    geometry_points = [Point(lon, lat) for lon, lat in zip(df_target["longitude"], df_target["latitude"])]
    gdf_target = gpd.GeoDataFrame(df_target, geometry=geometry_points, crs="EPSG:4326")
    gdf_hex = gpd.GeoDataFrame(df_poly, geometry="poly", crs="EPSG:4326")
    gdf_hex["poly_keep"] = gdf_hex["poly"]

    joined = gpd.sjoin(gdf_target, gdf_hex, how="left", predicate="within")

    # ── Geometric features per (mobile, hex) row ──
    joined["dist_to_cluster_center_point_hex"] = haversine_m(
        joined["latitude"], joined["longitude"],
        joined["center_lat"], joined["center_lon"],
    )

    joined_proj = joined.to_crs(7755)
    poly_keep_proj = gpd.GeoSeries(joined["poly_keep"], crs="EPSG:4326").to_crs(7755)

    joined["dist_to_boundary_edge_point_hex"] = poly_keep_proj.boundary.distance(joined_proj.geometry)
    joined["dist_to_boundary_edge_point_hex"] = joined["dist_to_boundary_edge_point_hex"].where(
        joined["poly_keep"].notna(), np.nan
    )
    joined["near_edge_point_hex"] = np.where(
        joined["dist_to_boundary_edge_point_hex"] < 0.7 * joined["dist_to_cluster_center_point_hex"], 1, 0
    )
    joined["depth_score_point_hex"] = (
        joined["dist_to_cluster_center_point_hex"] - joined["dist_to_boundary_edge_point_hex"]
    )

    # ── Hop features: merge at (partner_id, poly_id) level before aggregation ──
    print("[HOP FEATURES] Computing 3-hop neighbor SE aggregates...")
    df_hop = compute_hop_features(df_poly, n_hops=3)
    print(f"[HOP FEATURES] {len(df_hop)} hex rows with hop features")
    joined = joined.merge(df_hop, on=["partner_id", "poly_id"], how="left")

    # ── A.1: Indeterminate check per partner ──
    joined["color_adj"] = joined["color"].copy()
    indeterminate_mask = (joined["color"] == "lightgreen") & (
        joined["installs"] <= config.INDETERMINATE_INSTALLS_CUTOFF
    )
    joined.loc[indeterminate_mask, "color_adj"] = "indeterminate"

    # ── A.2: Encode color numerically ──
    joined["color_numeric"] = joined["color_adj"].map(COLOR_NUMERIC_MAP).fillna(0)

    # ── A.3: Asymmetric credibility weight (shrink then aggregate) ──
    ratio = np.maximum(
        config.MIN_SHRINKAGE_RATIO,
        joined["total"] / (joined["total"] + config.SHRINKAGE_K),
    )
    joined["se_shrunk"] = np.where(
        joined["se"] >= 0,
        joined["se"] * ratio,
        joined["se"] / ratio,
    )

    # ── A.4: Weighted intermediates for consensus ──
    joined["_w_color"] = joined["color_numeric"] * joined["total"]
    joined["install_shrunk"] = joined["se_shrunk"] * joined["total"]

    # Hop: total-weighted intermediates
    for h in [1, 2, 3]:
        joined[f"_w_hop{h}_wmean"] = joined[f"hop{h}_se_wmean"] * joined["total"]
    joined["_w_gradient"] = joined["se_gradient_1to3"] * joined["total"]
    joined["_w_confirmed"] = joined["se_confirmed"] * joined["total"]
    joined["_w_isolation"] = joined["isolation_ratio"] * joined["total"]

    # ── A.5: Mobile-level consensus groupby ──
    consensus = (
        joined.groupby(["mobile", "latitude", "longitude"])
        .agg(
            # SE / color consensus
            install_shrunk=("install_shrunk", "sum"),
            _sum_w_color=("_w_color", "sum"),
            total=("total", "sum"),
            installs=("installs", "sum"),
            declines=("declines", "sum"),
            n_covering_partners=("partner_id", "nunique"),
            # Geometric: median across covering hexes
            dist_to_boundary_edge_point_hex=("dist_to_boundary_edge_point_hex", "median"),
            dist_to_cluster_center_point_hex=("dist_to_cluster_center_point_hex", "median"),
            near_edge_point_hex=("near_edge_point_hex", "median"),
            depth_score_point_hex=("depth_score_point_hex", "median"),
            parent_overlap=("is_overlap", "mean"),
            # Hop counts: sum
            hop1_count=("hop1_count", "sum"),
            hop2_count=("hop2_count", "sum"),
            hop3_count=("hop3_count", "sum"),
            # Hop std: worst-case
            hop1_se_std=("hop1_se_std", "max"),
            hop2_se_std=("hop2_se_std", "max"),
            hop3_se_std=("hop3_se_std", "max"),
            # Hop weighted intermediates
            _w_hop1=("_w_hop1_wmean", "sum"),
            _w_hop2=("_w_hop2_wmean", "sum"),
            _w_hop3=("_w_hop3_wmean", "sum"),
            _w_gradient=("_w_gradient", "sum"),
            _w_confirmed=("_w_confirmed", "sum"),
            _w_isolation=("_w_isolation", "sum"),
        )
        .reset_index()
    )

    # ── Derived consensus columns ──
    total_safe = consensus["total"].replace(0, np.nan)

    consensus["weighted_se"] = np.where(
        consensus["total"] > 0, consensus["installs"] / consensus["total"], 0
    )
    consensus["weighted_se_shrunk"] = np.where(
        consensus["total"] > 0, consensus["install_shrunk"] / consensus["total"], 0
    )
    consensus["parent_color_numeric"] = consensus["_sum_w_color"] / total_safe

    def numeric_to_color(v):
        if pd.isna(v):
            return np.nan
        if v >= 2.5:
            return "lightgreen"
        if v >= 1.5:
            return "orange"
        return "crimson"

    consensus["parent_color"] = consensus["parent_color_numeric"].apply(numeric_to_color)

    # Hop SE wmeans: divide weighted sums by total
    for h in [1, 2, 3]:
        consensus[f"hop{h}_se_wmean"] = consensus[f"_w_hop{h}"] / total_safe
    consensus["se_gradient_1to3"] = consensus["_w_gradient"] / total_safe
    consensus["se_confirmed"] = consensus["_w_confirmed"] / total_safe
    consensus["isolation_ratio"] = consensus["_w_isolation"] / total_safe

    # Drop all intermediates
    consensus.drop(
        columns=[
            "_sum_w_color",
            "_w_hop1", "_w_hop2", "_w_hop3",
            "_w_gradient", "_w_confirmed", "_w_isolation",
        ],
        inplace=True,
    )

    return consensus


# ==============================================================
# PROCESS — main pipeline entry point
# ==============================================================

def process(
    df_source,
    df_target,
    df_poly,
    df_bound,
    lambda_decay=DEFAULT_LAMBDA,
    max_radius_m=500,
):
    try:
        df_poly["centroid_lat"] = pd.to_numeric(
            df_poly["poly"].apply(lambda p: p.centroid.y if p else np.nan), errors="coerce"
        )
        df_poly["centroid_lon"] = pd.to_numeric(
            df_poly["poly"].apply(lambda p: p.centroid.x if p else np.nan), errors="coerce"
        )

        # ── Boundary features (mobile level) ──
        df_target_boundary = add_boundary_details_precise(df_target, df_bound, df_source)
        df_target_new = pd.merge(df_target, df_target_boundary, how="left", on="mobile")

        _df_poly_snapshot = df_poly.copy(deep=True)

        # ── Hex consensus features (mobile level, includes hop features) ──
        df_hex_consensus = compute_hex_consensus_features(df_target_new, df_poly)

        # Merge consensus back onto df_target_new to retain boundary columns
        df_target_with_hex = pd.merge(
            df_target_new, df_hex_consensus,
            on=["mobile", "latitude", "longitude"],
            how="left",
        )

        # ── Source points GeoDataFrame (reused by ALL-HEX FIELD) ──
        gdf_source_points = gpd.GeoDataFrame(
            df_source,
            geometry=gpd.points_from_xy(df_source.longitude, df_source.latitude),
            crs="EPSG:4326",
        )

        # ==============================================================
        # COMBINED FIELD ACROSS ALL OVERLAPPING PARTNER HEXAGONS
        # ==============================================================
        agg_all_hex = None

        try:
            print("[ALL-HEX FIELD] Starting combined field computation across all overlapping hexagons...")

            _df_poly_copy = _df_poly_snapshot.copy()
            _df_poly_copy.rename(columns={"partner_id": "all_poly_partner_id"}, inplace=True)

            gdf_all_polys = gpd.GeoDataFrame(_df_poly_copy, geometry="poly", crs="EPSG:4326")

            gdf_target_pts = gpd.GeoDataFrame(
                df_target[["mobile", "latitude", "longitude", "decision_time"]].copy(),
                geometry=gpd.points_from_xy(df_target.longitude, df_target.latitude),
                crs="EPSG:4326",
            )

            all_joined = gpd.sjoin(gdf_target_pts, gdf_all_polys, how="inner", predicate="within")
            print(f"[ALL-HEX FIELD] Targets × all hexagons sjoin: {len(all_joined)} rows")

            if len(all_joined) == 0:
                raise ValueError("Empty sjoin — skipping all-hex field")

            # Filter source polys to only those with targets (perf optimisation)
            unique_polys = (
                all_joined.groupby(["all_poly_partner_id", "poly_id"])["mobile"]
                .nunique()
                .reset_index()
            )
            # Rename back so merge keys align with _df_poly_snapshot
            unique_polys.rename(columns={"all_poly_partner_id": "partner_id"}, inplace=True)

            _df_poly_copy_src = pd.merge(
                _df_poly_snapshot,
                unique_polys[["partner_id", "poly_id"]],
                how="inner",
                on=["partner_id", "poly_id"],
            )
            _df_poly_copy_src.rename(columns={"partner_id": "all_poly_partner_id"}, inplace=True)

            gdf_all_polys_for_src = gpd.GeoDataFrame(
                _df_poly_copy_src, geometry="poly", crs="EPSG:4326"
            )

            all_joined_src = gpd.sjoin(
                gdf_source_points, gdf_all_polys_for_src, how="inner", predicate="within"
            )
            print(f"[ALL-HEX FIELD] Sources × filtered hexagons sjoin: {len(all_joined_src)} rows")

            all_joined_src = all_joined_src[
                all_joined_src["partner_id"] == all_joined_src["all_poly_partner_id"]
            ].copy()
            print(f"[ALL-HEX FIELD] After partner match filter: {len(all_joined_src)} source rows")

            df_all_source_in_hex = all_joined_src.drop(
                columns=["geometry", "index_right", "all_poly_partner_id"], errors="ignore"
            )

            # ── Per-hex field computation ──
            all_hex_results = []
            all_hex_groups = all_joined.groupby(["all_poly_partner_id", "poly_id"])
            print(f"[ALL-HEX FIELD] Processing {all_hex_groups.ngroups} hex groups...")

            for (pid, polyid), tgt_grp in tqdm(
                all_hex_groups, total=all_hex_groups.ngroups, desc="All-hex combined field"
            ):
                src_local = df_all_source_in_hex[
                    (df_all_source_in_hex["partner_id"] == pid)
                    & (df_all_source_in_hex["poly_id"] == polyid)
                ]
                tgt_sub = tgt_grp[["mobile", "latitude", "longitude", "decision_time"]].copy()

                if len(src_local) == 0:
                    tgt_sub["_field_all"] = np.nan
                    tgt_sub["_n_sources"] = 0
                    tgt_sub["_kernel_sum"] = 0.0
                else:
                    field_df = compute_adaptive_gaussian_field(
                        df_target=tgt_sub,
                        df_source=src_local,
                        lambda_decay=lambda_decay,
                        max_radius_m=max_radius_m,
                        verbose=False,
                    )
                    tgt_sub["_field_all"] = field_df["predicted_field"]
                    tgt_sub["_n_sources"] = len(src_local)
                    tgt_sub["_kernel_sum"] = field_df["kernel_sum"]

                tgt_sub["_hex_partner"] = pid
                tgt_sub["_hex_poly"] = polyid
                all_hex_results.append(
                    tgt_sub[["mobile", "_hex_partner", "_hex_poly", "_field_all", "_n_sources", "_kernel_sum"]]
                )

            # ── Aggregate per-hex fields to mobile level ──
            if all_hex_results:
                df_all_hex = pd.concat(all_hex_results, ignore_index=True)
                print(f"[ALL-HEX FIELD] Concatenated {len(df_all_hex)} field rows across all hexes")

                df_valid = df_all_hex.dropna(subset=["_field_all"]).copy()
                df_valid["_weighted_field"] = df_valid["_field_all"] * df_valid["_n_sources"]
                df_valid["_ks_weighted_field"] = df_valid["_field_all"] * df_valid["_kernel_sum"]

                agg_all_hex = (
                    df_valid.groupby("mobile")
                    .agg(
                        _sum_weighted_field=("_weighted_field", "sum"),
                        _sum_sources=("_n_sources", "sum"),
                        _sum_ks_weighted_field=("_ks_weighted_field", "sum"),
                        _sum_kernel_sums=("_kernel_sum", "sum"),
                        predicted_field_hex_all_min=("_field_all", "min"),
                        predicted_field_hex_all_max=("_field_all", "max"),
                        predicted_field_hex_all_std=("_field_all", "std"),
                        n_overlapping_hexes_field=("_field_all", "count"),
                    )
                    .reset_index()
                )

                agg_all_hex["predicted_field_hex_all_wmean"] = (
                    agg_all_hex["_sum_weighted_field"]
                    / agg_all_hex["_sum_sources"].replace(0, np.nan)
                )
                agg_all_hex["predicted_field_hex_all_kswmean"] = (
                    agg_all_hex["_sum_ks_weighted_field"]
                    / agg_all_hex["_sum_kernel_sums"].replace(0, np.nan)
                )

                unweighted = (
                    df_valid.groupby("mobile")["_field_all"]
                    .mean()
                    .rename("predicted_field_hex_all_mean")
                    .reset_index()
                )
                agg_all_hex = agg_all_hex.merge(unweighted, on="mobile", how="left")
                agg_all_hex.rename(columns={"_sum_sources": "total_sources_all_hexes"}, inplace=True)
                agg_all_hex.drop(
                    columns=["_sum_weighted_field", "_sum_ks_weighted_field", "_sum_kernel_sums"],
                    inplace=True,
                )

                print(f"[ALL-HEX FIELD] SUCCESS — {len(agg_all_hex)} mobiles with combined field")
            else:
                print("[ALL-HEX FIELD] WARNING: no hex results produced")
                agg_all_hex = None

        except Exception as e_allhex:
            print(f"[ALL-HEX FIELD] ERROR (non-fatal, existing columns unaffected): {e_allhex}")
            import traceback
            traceback.print_exc()
            agg_all_hex = None

        # ── Final assembly ──
        df_target_final = df_target_with_hex

        if agg_all_hex is not None and len(agg_all_hex) > 0:
            df_target_final = pd.merge(df_target_final, agg_all_hex, on="mobile", how="left")
        else:
            for col in [
                "predicted_field_hex_all_wmean", "predicted_field_hex_all_kswmean",
                "predicted_field_hex_all_mean", "predicted_field_hex_all_min",
                "predicted_field_hex_all_max", "predicted_field_hex_all_std",
                "n_overlapping_hexes_field", "total_sources_all_hexes",
            ]:
                df_target_final[col] = np.nan
            print("[ALL-HEX FIELD] Columns added as NaN (computation failed or empty)")

        # Alias for backward compatibility with downstream scoring
        df_target_final["predicted_field_hex"] = df_target_final.get(
            "predicted_field_hex_all_kswmean", np.nan
        )
        # Alias: downstream expects parent_installs
        df_target_final["parent_installs"] = df_target_final.get("installs", np.nan)

        return df_target_final

    except Exception as e:
        print(f"ERROR IN PROCESSING OF HEXAGONS AND BOUNDARIES: {e}")
        return pd.DataFrame()
    