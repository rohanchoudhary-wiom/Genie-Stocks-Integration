import numpy as np
from data_lib.config import EARTH_RADIUS_METER

def equiv_radius_m(area_m2: float) -> float:
    return float(np.sqrt(area_m2 / np.pi))

def haversine_m(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_METER * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def haversine_rad(a, b):
    """Haversine between two (lat_rad, lon_rad) points. Returns radians."""
    dlat = b[0] - a[0]
    dlon = b[1] - a[1]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[0]) * np.cos(b[0]) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(h))