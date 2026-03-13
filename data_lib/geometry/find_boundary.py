import pandas as pd
import numpy as np
import folium
from shapely.ops import unary_union, transform
from tqdm import tqdm
import os
from sklearn.cluster import DBSCAN
import pyproj
from functools import partial
import data_lib.config as config


def run_find_boundary():
    df = pd.read_hdf("artifacts/poly_stats.h5", "df")

    print("IDENTIFYING HEX CENTROID LAT LONG")
    df["centroid_lat"] = df["poly"].apply(lambda p: p.centroid.y if p else np.nan)
    df["centroid_lon"] = df["poly"].apply(lambda p: p.centroid.x if p else np.nan)

    print("FILTERING SERVICEABLE HEXES")
    df = df[df["color"].isin(["lightgreen", "orange"])].copy()
    print(df["installs"].describe())
    print("p50 value is 4, so filtering for hexagons > 4")

    print("COMPUTING INSTALLS QUANTILES")
    quantiles = [i / 10 for i in range(0, 11)]
    df_quant = df["installs"].quantile(quantiles).reset_index()
    df_quant.columns = ["quantile", "value"]
    print(df_quant)

    p30 = df_quant[df_quant["quantile"] == 0.3]["value"].values[0]
    p70 = df_quant[df_quant["quantile"] == 0.7]["value"].values[0]
    p90 = df_quant[df_quant["quantile"] == 0.9]["value"].values[0]
    print(f"Dynamic p30: {p30} | p70: {p70} | p90: {p90}")

    if config.ENABLE_BOUNDARY_FILTER == 1:
        print(f"Filtering hexagons with installs > p30 ({p30})")
        df = df[df["installs"] > p30].copy()
    else:
        print("Boundary filtering DISABLED (using all valid hexes)")

    cluster_results = []

    for pid, df_sub in tqdm(df.groupby("partner_id"), desc="CLUSTERING_DBSCAN"):
        p90_id = -1

        coords = np.radians(df_sub[["centroid_lat", "centroid_lon"]].values)
        eps_in_km = df_sub["best_size"].max() * 2
        eps_in_radians = eps_in_km / 6371.0

        db = DBSCAN(eps=eps_in_radians, min_samples=2, metric="haversine")
        labels = db.fit_predict(coords)
        df_sub["cluster_id"] = labels

        proper_clusters = df_sub[df_sub["cluster_id"] != -1].copy()
        proper_clusters["cluster_type"] = "dbscan_cluster"

        noise = df_sub[df_sub["cluster_id"] == -1].copy()
        high_value_noise = noise[noise["installs"] >= p90].copy()

        if not high_value_noise.empty:
            print(
                f"Partner {pid}: Rescuing {len(high_value_noise)} high-value singleton(s) (>= p90 installs)"
            )
            high_value_noise["cluster_id"] = p90_id
            p90_id -= 1
            high_value_noise["cluster_type"] = "p90_single_cluster"

        cluster_results.append(pd.concat(
            [proper_clusters, high_value_noise], ignore_index=True
        ))

    df_cluster = pd.concat(cluster_results, ignore_index=True)

    print("DISSOLVING BOUNDARIES FOR EACH CLUSTER PER PARTNER")
    boundary_summary = []

    project = partial(
        pyproj.transform, pyproj.Proj("epsg:4326"), pyproj.Proj("epsg:32643")
    )

    for (partner_id, cluster_id), group_df in tqdm(
        df_cluster.groupby(["partner_id", "cluster_id"])
    ):
        dissolved = unary_union(
            [p.buffer(1e-9) for p in group_df["poly"].tolist() if p]
        ).buffer(-1e-9)
        if dissolved.is_empty or dissolved.geom_type == "MultiPolygon":
            dissolved = (
                max(dissolved.geoms, key=lambda p: p.area)
                if dissolved.geom_type == "MultiPolygon"
                else None
            )
        if not dissolved:
            continue

        center_lat, center_lon = dissolved.centroid.y, dissolved.centroid.x

        boundary_coords = (
            [[lon, lat] for lon, lat in dissolved.exterior.coords]
            if hasattr(dissolved, "exterior")
            else []
        )

        projected_poly = transform(project, dissolved)
        area_km2 = projected_poly.area / 1e6

        # ── Build row dict with ALL keys including temporal ──
        row = {
            "partner_id": partner_id,
            "cluster_id": cluster_id,
            "cluster_type": group_df["cluster_type"].iloc[0]
            if "cluster_type" in group_df.columns
            else "unknown",
            "center_lat": center_lat,
            "center_lon": center_lon,
            "total_installs": group_df["installs"].sum(),
            "total_obs": group_df["total"].sum(),
            "n_hexes": len(group_df),
            "area_km2": round(area_km2, 3),
            "boundary_poly": dissolved,
            "boundary_coords": boundary_coords,
        }

        # ── Temporal cluster stats (summed from hex-level) ──
        for wd in config.TEMPORAL_WINDOWS:
            inst_col = f"installs_{wd}d"
            tot_col = f"total_{wd}d"
            if inst_col in group_df.columns:
                t_inst = group_df[inst_col].sum()
                t_tot = group_df[tot_col].sum()
                row[f"total_installs_{wd}d"] = int(t_inst)
                row[f"total_obs_{wd}d"] = int(t_tot)
                row[f"cluster_se_{wd}d"] = round(t_inst / t_tot, 4) if t_tot > 0 else np.nan
            else:
                row[f"total_installs_{wd}d"] = np.nan
                row[f"total_obs_{wd}d"] = np.nan
                row[f"cluster_se_{wd}d"] = np.nan

        boundary_summary.append(row)

    boundaries_df = pd.DataFrame(boundary_summary)
    boundaries_df.to_hdf("artifacts/partner_cluster_boundaries.h5", key="df", mode="w")
    boundaries_df.drop(
        columns=["boundary_poly"], inplace=True
    )
    boundaries_df.to_csv("artifacts/partner_cluster_boundaries.csv", index=False)
    print("\nFINAL BOUNDARY SUMMARY")
    print(boundaries_df.head(10))

    print("GENERATING OVERVIEW MAP WITH ALL CLUSTER BOUNDARIES")
    overview = folium.Map(
        location=[
            boundaries_df["center_lat"].mean(),
            boundaries_df["center_lon"].mean(),
        ],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    for _, row in boundaries_df.iterrows():
        if not row["boundary_coords"]:
            continue
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in row["boundary_coords"]],
            color="#2ecc71",
            weight=3,
            fill=True,
            fill_opacity=0.3,
            tooltip=f"Partner {row['partner_id']} Cluster {row['cluster_id']}<br>Installs: {row['total_installs']}<br>Area: {row['area_km2']} km²",
        ).add_to(overview)

        folium.Marker(
            [row["center_lat"], row["center_lon"]],
            icon=folium.Icon(color="blue", icon="info-sign"),
            tooltip=f"Center: {row['total_installs']} installs",
        ).add_to(overview)

    VIRTUAL_BOUNDARY_DIR = 'artifacts/virtual_boundary'
    os.makedirs(VIRTUAL_BOUNDARY_DIR, exist_ok=True)

    overview.save(f"{VIRTUAL_BOUNDARY_DIR}/all_clusters_overview.html")
    print(f"DONE. Open '{VIRTUAL_BOUNDARY_DIR}/all_clusters_overview.html' to view polygons.")

    print("GENERATING INDIVIDUAL PARTNER MAPS WITH CLUSTER BOUNDARIES")
    unique_partners = boundaries_df["partner_id"].unique()
    for partner_id in tqdm(unique_partners, desc="PARTNER_MAPS"):
        partner_df = boundaries_df[boundaries_df["partner_id"] == partner_id]
        if partner_df.empty:
            continue

        mean_lat = partner_df["center_lat"].mean()
        mean_lon = partner_df["center_lon"].mean()
        partner_map = folium.Map(
            location=[mean_lat, mean_lon],
            zoom_start=12,
            tiles="CartoDB positron",
        )

        for _, row in partner_df.iterrows():
            if not row["boundary_coords"]:
                continue
            folium.Polygon(
                locations=[(lat, lon) for lon, lat in row["boundary_coords"]],
                color="#2ecc71",
                weight=3,
                fill=True,
                fill_opacity=0.3,
                tooltip=f"Cluster {row['cluster_id']}<br>Installs: {row['total_installs']}<br>Area: {row['area_km2']} km²",
            ).add_to(partner_map)
            folium.Marker(
                [row["center_lat"], row["center_lon"]],
                icon=folium.Icon(color="blue", icon="info-sign"),
                tooltip=f"Cluster {row['cluster_id']} Center: {row['total_installs']} installs",
            ).add_to(partner_map)

        partner_map.save(f"{VIRTUAL_BOUNDARY_DIR}/partner_{partner_id}_clusters.html")
        print(
            f"Generated map for Partner {partner_id}: '{VIRTUAL_BOUNDARY_DIR}/partner_{partner_id}_clusters.html'"
        )

    print("ALL PARTNER MAPS GENERATED.")

    return boundaries_df

if __name__ == "__main__":
    run_find_boundary()
    