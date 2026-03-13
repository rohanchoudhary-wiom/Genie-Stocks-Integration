from data_lib.config import TEMPORAL_WINDOWS

one_to_many_cols = [
    "mobile", "partner_id", "cluster_id", "center_lat", "center_lon",
    "total_installs", "total_obs", "n_hexes", "area_km2",
    "dist_to_boundary_edge_m", "dist_to_cluster_center_m", "cluster_type",
    "dist_to_nearest_boundary_m", "nmbr_boundaries_within_100m",
]

hex_cols = [
    "partner_id", "poly_id", "local_field_x1000", "is_overlap",
    "distance_from_boundary_m", "distance_to_own_boundary_m", "rank", "best_size",
    "dist_to_boundary_edge_point_hex", "dist_to_cluster_center_point_hex",
    "near_edge_point_hex", "depth_score_point_hex",
] + [f"se_{wd}d" for wd in TEMPORAL_WINDOWS] + \
    [f"installs_{wd}d" for wd in TEMPORAL_WINDOWS] + \
    [f"total_{wd}d" for wd in TEMPORAL_WINDOWS] + \
    ["install_velocity"]

overlap_columns = [
    "partner_id", "poly_id", "best_size", "poly", "se",
    "installs", "declines", "total", "color",
    "is_overlap", "distance_from_boundary_m",
    "distance_to_own_boundary_m", "rank",
] 
