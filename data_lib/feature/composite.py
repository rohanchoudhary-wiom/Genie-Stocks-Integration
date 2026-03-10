"""
lib/composite.py

R (Promise Governor) — Composite scoring.

Takes the spatial features (from B_spatial in compute.py) and the
operational features (from B_operational in ops_features.py) and
fuses them into a single composite score per lead.

Architecture doc formula:
    partner_score = spatial_shrunk × operational_score

Where spatial_shrunk is the weighted combination of:
    1. field signal         (predicted_field_hex)
    2. contested field      (contested_field)
    3. evidence strength    (log parent_total, parent_se)
    4. hex colour mapping   (parent_color_numeric)

And operational_score is from ops_features.py.

Final output: confidence tier (HIGH / MOD / LOW / DECLINE).
"""

import numpy as np
import pandas as pd

import lib.config as config
import lib.stocks.stocks_config as sc


def _safe_normalize(series: pd.Series, method: str = "rank") -> pd.Series:
    """
    Normalise a series to [0, 1].

    method='rank'    → percentile-rank (robust to outliers)
    method='minmax'  → standard min-max
    """
    s = series.copy()
    valid = s.dropna()
    if len(valid) == 0:
        return pd.Series(np.nan, index=series.index)

    if method == "rank":
        ranked = s.rank(pct=True, na_option="keep")
        return ranked
    else:
        mn, mx = valid.min(), valid.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index).where(s.notna(), np.nan)
        return (s - mn) / (mx - mn)


def compute_spatial_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and normalises the spatial signal components from scored test data.

    Expects columns from compute.process():
        predicted_field_hex, contested_field, parent_total,
        parent_se, parent_color_numeric

    Adds: norm_field, norm_contested, norm_evidence, norm_color, spatial_shrunk
    """
    df = df.copy()

    # 1. Field signal — already signed float, higher = more install-likely
    if "predicted_field_hex" in df.columns:
        df["norm_field"] = _safe_normalize(df["predicted_field_hex"], method="rank")
    else:
        df["norm_field"] = np.nan

    # 2. Contested field — same sign convention as field
    if "contested_field" in df.columns:
        df["norm_contested"] = _safe_normalize(df["contested_field"], method="rank")
    else:
        df["norm_contested"] = np.nan

    # 3. Evidence strength — log(parent_total) × sign(parent_se)
    if "parent_total" in df.columns and "parent_se" in df.columns:
        log_evidence = np.log1p(df["parent_total"].fillna(0).clip(0))
        sign_se = np.sign(df["parent_se"].fillna(0))
        raw_evidence = log_evidence * sign_se
        df["norm_evidence"] = _safe_normalize(raw_evidence, method="rank")
    else:
        df["norm_evidence"] = np.nan

    # 4. Hex colour numeric — already 1–3 from compute.py
    if "parent_color_numeric" in df.columns:
        df["norm_color"] = _safe_normalize(df["parent_color_numeric"], method="minmax")
    else:
        df["norm_color"] = np.nan

    # --- Weighted combination → spatial_shrunk ---
    # Use static weights here; the Dirichlet optimiser in notebooks
    # produces better weights offline, which you inject via config.
    w = np.array([sc.R_WEIGHT_FIELD, sc.R_WEIGHT_CONTESTED,
                   sc.R_WEIGHT_EVIDENCE,
                   sc.R_WEIGHT_EVIDENCE])  # color gets same weight as evidence for now

    components = df[["norm_field", "norm_contested", "norm_evidence", "norm_color"]].values

    # Weighted mean ignoring NaN: sum(w_i * x_i) / sum(w_i) for non-NaN
    mask = ~np.isnan(components)
    w_expanded = np.broadcast_to(w, components.shape)
    w_active = np.where(mask, w_expanded, 0)

    numerator = np.nansum(components * w_expanded, axis=1)
    denominator = np.sum(w_active, axis=1)
    denominator = np.where(denominator == 0, np.nan, denominator)

    df["spatial_shrunk"] = numerator / denominator

    return df


def fuse_spatial_operational(
    df: pd.DataFrame,
    df_ops_scored: pd.DataFrame,
) -> pd.DataFrame:
    """
    R fusion: partner_score = spatial_shrunk × operational_score.

    df            — test leads with spatial_shrunk (from compute_spatial_components)
    df_ops_scored — partner ops vector with operational_score (from ops_features.py)

    The merge is on partner_id.  If a lead spans multiple partners (via
    n_covering_partners > 1), the parent partner_id from get_parent_hexagon
    is used as the primary key.

    Adds: operational_score, composite_score, confidence_tier
    """
    df = df.copy()

    if df_ops_scored is not None and not df_ops_scored.empty:
        ops_cols = ["partner_id", "operational_score",
                    "capacity_score", "reliability_score", "infrastructure_score",
                    "gate_blocked"]
        ops_cols = [c for c in ops_cols if c in df_ops_scored.columns]

        df = df.merge(
            df_ops_scored[ops_cols],
            on="partner_id",
            how="left",
        )
    else:
        df["operational_score"] = sc.R_OPS_FLOOR
        df["gate_blocked"] = 0

    # Fill missing operational_score with floor (not enough data, not necessarily bad)
    df["operational_score"] = df["operational_score"].fillna(sc.R_OPS_FLOOR)

    # --- Composite = spatial × operational ---
    df["composite_score"] = (
        df["spatial_shrunk"].fillna(0) * df["operational_score"]
    )

    # --- Confidence tiers ---
    conditions = [
        df["composite_score"] >= sc.R_TIER_MOD,
        df["composite_score"] >= sc.R_TIER_LOW,
        df["composite_score"] >= sc.R_TIER_DECLINE,
    ]
    choices = ["HIGH", "MOD", "LOW"]
    df["confidence_tier"] = np.select(conditions, choices, default="DECLINE")

    # Gate-blocked partners → force DECLINE
    if "gate_blocked" in df.columns:
        df.loc[df["gate_blocked"] == 1, "confidence_tier"] = "DECLINE"

    # --- Final serviceability (R decision) ---
    df["r_serviceable"] = np.where(
        df["confidence_tier"].isin(["HIGH", "MOD"]), 1, 0
    )

    _report_tiers(df)

    return df


def _report_tiers(df: pd.DataFrame) -> None:
    """Print tier distribution and SE per tier."""
    print("\n--- R (PROMISE GOVERNOR) — TIER DISTRIBUTION ---")

    if "installed_decision" not in df.columns:
        tier_counts = df["confidence_tier"].value_counts().sort_index()
        for tier, count in tier_counts.items():
            print(f"  {tier:>8s}: {count:>7,}")
        return

    for tier in ["HIGH", "MOD", "LOW", "DECLINE"]:
        mask = df["confidence_tier"] == tier
        sub = df[mask]
        n = len(sub)
        if n == 0:
            print(f"  {tier:>8s}:       0")
            continue
        installs = sub["installed_decision"].sum()
        se = installs / n
        print(f"  {tier:>8s}: {n:>7,}  |  SE={se:.4f}  |  installs={int(installs)}")

    # Separation: HIGH SE - DECLINE SE
    high_mask = df["confidence_tier"] == "HIGH"
    decl_mask = df["confidence_tier"] == "DECLINE"
    if high_mask.any() and decl_mask.any():
        se_high = df.loc[high_mask, "installed_decision"].mean()
        se_decl = df.loc[decl_mask, "installed_decision"].mean()
        print(f"\n  Separation (HIGH - DECLINE): {se_high - se_decl:.4f}")


def compute_composite(
    df_scored: pd.DataFrame,
    df_ops_scored: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Top-level composite scoring entry point.

    Args:
        df_scored     : output of compute.process() with spatial features
        df_ops_scored : output of ops_features.compute_operational_score()

    Returns:
        df with composite_score, confidence_tier, r_serviceable added.
    """
    print("\n--- R (PROMISE GOVERNOR) — COMPOSITE SCORING ---")

    df = compute_spatial_components(df_scored)
    df = fuse_spatial_operational(df, df_ops_scored)

    return df