# diagnostic_hex_maps_pure.py
# NO H3. NO UBER. JUST TRUTH.
import pandas as pd
import folium
from shapely.geometry import Polygon
import os
from math import radians, cos, sin, sqrt
from shapely import contains_xy
from data_lib import config
import numpy as np

def create_hex_grid(center_lat, center_lon, radius_km=3.0, hex_size_km=0.25):
    """
    Perfect regular hexagonal tiling using local flat approximation.
    Accurate to <0.5% error within 10 km — more than enough for 250m hexes.
    """
    # Constants in metres
    
    meters_per_deg_lon = config.METERS_PER_DEG_LAT * cos(radians(center_lat))

    # Hex geometry (flat-top orientation)
    hex_radius_m = hex_size_km * 1000  # center → vertex
    dx = hex_radius_m * sqrt(3)  # horizontal spacing
    dy = hex_radius_m * 1.5  # row spacing

    n_rings = int(radius_km / hex_size_km) + 2
    hexes = []

    for i in range(-n_rings, n_rings + 1):
        for j in range(-n_rings, n_rings + 1):
            if abs(i) + abs(j) + abs(-i - j) > 2 * n_rings:
                continue

            # Offset to metres
            x_m = dx * (j + i / 2.0)
            y_m = dy * i

            # Hex centre in lat/lon
            c_lat = center_lat + y_m / config.METERS_PER_DEG_LAT
            c_lon = center_lon + x_m / meters_per_deg_lon

            # Six vertices around this centre
            vertices = []
            for k in range(6):
                angle = radians(60 * k + 30)  # flat-top
                vx = c_lon + (hex_radius_m / meters_per_deg_lon) * cos(angle)
                vy = c_lat + (hex_radius_m / config.METERS_PER_DEG_LAT) * sin(angle)
                vertices.append((vx, vy))

            hexes.append(Polygon(vertices))

    return hexes


def find_best_hexes(center_lat, center_lon, sources, hex_sizes=None, radius_km=3.0):
    src_lats = sources["latitude"].values
    src_lons = sources["longitude"].values

    # Allow caller to inject hex_sizes; fall back to default list for backward compatibility
    if hex_sizes is None:
        hex_sizes = [
            0.025,
            0.05,
            0.08,
            0.10,
            0.12,
            0.14,
            0.16,
            0.18,
            0.20,
            0.22,
            0.24,
            0.25,
        ]

    # DEFAULT VALUE OF SIZE IF A PARTNER DOESN'T MEET THE SE SEPARATION CRITERION: EG. NEW PARTNERS
    best_size = config.DEFAULT_HEX_SIZE
    max_separation = 0
    for hex_size_km in hex_sizes:
        hexes = create_hex_grid(
            center_lat, center_lon, radius_km=radius_km, hex_size_km=hex_size_km
        )
        poly_id = 0
        poly_chars = []
        for hex_poly in hexes:
            mask = contains_xy(hex_poly, src_lons, src_lats)
            hex_src = sources[mask]

            installs = len(hex_src[hex_src["is_installed"] == 1])
            declines = len(hex_src[hex_src["is_declined"] == 1])
            total = len(hex_src)
            if total == 0:
                continue

            poly_id += 1
            poly_chars.append(
                {
                    "poly_id": poly_id,
                    "installs": installs,
                    "declines": declines,
                    "total": total,
                }
            )

        df_poly = pd.DataFrame(
            poly_chars, columns=["poly_id", "installs", "declines", "total"]
        )
        df_poly["se"] = df_poly["installs"] / df_poly["total"]
        # print(f"SHAPE OF DF POLY -> NUMBER OF HEXAGONS: {df_poly.shape[0]}, {df_poly['poly_id'].nunique()}")
        if df_poly.shape[0] > 20:
            # df_poly['se_rng'] = pd.qcut(df_poly['se'], q=5,labels=False, duplicates='drop')+1
            df_poly["se_rng"] = pd.cut(
                df_poly["se"],
                bins=[i / 5 for i in range(6)],
                labels=[1, 2, 3, 4, 5],
                include_lowest=True,
            ).astype(int)
            # print(f"FOR {hex_size_km} POLY SIZE, BINS FORMED ARE: {df_poly.groupby(['se_rng']).agg(nmbr_hex=('poly_id','count'), se = ('se','mean'))}")
            # print(df_poly['se_rng'].value_counts())
            if df_poly["se_rng"].nunique() == 5:
                mask1 = df_poly["se_rng"] == 5
                mask2 = df_poly["se_rng"] == 2

                se_low = (
                    df_poly[mask2]["installs"].sum() / df_poly[mask2]["total"].sum()
                )
                se_high = (
                    df_poly[mask1]["installs"].sum() / df_poly[mask1]["total"].sum()
                )

                separation = se_high - se_low

                # print(f"FOR {hex_size_km}, SE LOW IS {se_low}, SE HIGH IS {se_high}, SEPARATION IS: {separation}")

                if separation > max_separation:
                    max_separation = separation
                    best_size = hex_size_km

    print(f"MAX SEPARATION IS: {max_separation}, BEST SIZE IS: {best_size}")

    return best_size


def compute_hexes(
    hexes,
    center_lat,
    center_lon,
    sources,
    bad_se,
    mid_se,
    best_size,
    pid,
    reference_date=None,
):
    hex_stats = []
    src_lats = sources["latitude"].values
    src_lons = sources["longitude"].values

    poly_id = 0
    # ── Pre-compute temporal cutoffs ──
    has_temporal = reference_date is not None and "decision_time" in sources.columns
    if has_temporal:
        ref_ts = pd.Timestamp(reference_date)
        cutoffs = {wd: ref_ts - pd.Timedelta(days=wd) for wd in config.TEMPORAL_WINDOWS}

    for hex_poly in hexes:
        # Fast vectorised containment
        mask = contains_xy(hex_poly, src_lons, src_lats)
        hex_src = sources[mask]

        if len(hex_src) == 0:
            continue

        # ------------------------------------------------------------------
        # Historical stats
        # ------------------------------------------------------------------

        installs = len(hex_src[hex_src["is_installed"] == 1])
        declines = len(hex_src[hex_src["is_declined"] == 1])
        total = len(hex_src)

        se = installs / total if total > 0 else 0
        color = (
            "crimson" if se <= bad_se else "orange" if se <= mid_se else "lightgreen"
        )

        # ------------------------------------------------------------------
        # Store everything
        # ------------------------------------------------------------------
        poly_id += 1

        # NEW:
        row = {
            "partner_id": pid,
            "best_size": best_size,
            "poly_id": poly_id,
            "poly": hex_poly,
            "se": round(se, 4),
            "installs": int(installs),
            "declines": int(declines),
            "total": int(total),
            "color": color,
        }

        if has_temporal:
            dt_col = hex_src["decision_time"]
            is_inst = hex_src["is_installed"] == 1
            is_decl = hex_src["is_declined"] == 1
            for wd in config.TEMPORAL_WINDOWS:
                wmask = dt_col >= cutoffs[wd]
                w_inst = int(is_inst[wmask].sum())
                w_decl = int(is_decl[wmask].sum())
                w_tot = int(wmask.sum())
                w_se = round(w_inst / w_tot, 4) if w_tot > 0 else np.nan
                row[f"se_{wd}d"] = w_se
                row[f"installs_{wd}d"] = w_inst
                row[f"declines_{wd}d"] = w_decl
                row[f"total_{wd}d"] = w_tot
            row["install_velocity"] = round(
                row[f"installs_{config.TEMPORAL_WINDOWS[0]}d"] / max(row[f"installs_{config.TEMPORAL_WINDOWS[-1]}d"], 1), 4
            )
        else:
            for wd in config.TEMPORAL_WINDOWS:
                row[f"se_{wd}d"] = np.nan
                row[f"installs_{wd}d"] = np.nan
                row[f"declines_{wd}d"] = np.nan
                row[f"total_{wd}d"] = np.nan
            row["install_velocity"] = np.nan

        hex_stats.append(row)

    # ------------------------------------------------------------------
    # Map generation (only if we have hexes)
    # ------------------------------------------------------------------

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron"
    )

    for stat in hex_stats:
        tooltip = f"""
        <b>Historical SE</b>: {stat["se"]:.1%} ({stat["installs"]}/{stat["total"]})<br>
        <b>Installs:</b> {stat["installs"]}<br>
        <b>Declines:</b> {stat["declines"]}<br>
        <b>Total Decisions:</b> {stat["total"]}<br>


        """
        folium.Polygon(
            locations=[(c[1], c[0]) for c in stat["poly"].exterior.coords],
            color=stat["color"],
            weight=2,
            fill=True,
            fillOpacity=0.65,
            tooltip=tooltip.replace("\n", "<br>"),
        ).add_to(m)

    # Historical points
    for _, r in sources.iterrows():
        color = "lightpink" if r["is_installed"] == 1 else "darkorange"
        folium.CircleMarker(
            [r.latitude, r.longitude], radius=4, color=color, fill=True
        ).add_to(m)

    os.makedirs("artifacts/DIAGNOSTIC_HEX_MAPS_DYNAMIC_HEX", exist_ok=True)

    m.save(
        f"artifacts/DIAGNOSTIC_HEX_MAPS_DYNAMIC_HEX/partner_{pid}_PURE_CORRECT_{best_size}.html"
    )

    return hex_stats
