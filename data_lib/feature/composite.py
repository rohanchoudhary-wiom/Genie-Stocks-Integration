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

import data_lib.config as config
import data_lib.stocks.stocks_config as sc


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

    ALL-TIME components:
        norm_field, norm_contested, norm_evidence, norm_color → spatial_shrunk

    TEMPORAL components (per window in TEMPORAL_WINDOWS):
        norm_se_{wd}d           — from weighted_se_{wd}d_shrunk
        norm_se_raw_{wd}d       — from weighted_se_{wd}d
        norm_contested_{wd}d    — from contested_se_{wd}d
        norm_hop1_se_{wd}d      — from hop1_se_{wd}d_wmean
        norm_gradient_{wd}d     — from se_gradient_1to3_{wd}d
        norm_density_{wd}d      — from local_density_{wd}d
        norm_evidence_{wd}d     — from total_{wd}d (log)
        norm_field_{wd}d        — from predicted_field_hex_{wd}d

    DERIVED:
        norm_momentum           — from se_momentum (30d shrunk - 365d shrunk)
        spatial_shrunk_{wd}d    — weighted combination of temporal norms per window
    """
    df = df.copy()

    # ════════════════════════════════════════════════════════════
    # ALL-TIME COMPONENTS
    # ════════════════════════════════════════════════════════════

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
    
    # ════════════════════════════════════════════════════════════
    # DERIVED: SE MOMENTUM (30d vs 365d)
    # ════════════════════════════════════════════════════════════

    if "se_momentum" in df.columns:
        df["norm_momentum"] = _safe_normalize(df["se_momentum"], method="rank")

    # Field momentum (from score_temporal_windows)
    if "field_momentum" in df.columns:
        df["norm_field_momentum"] = _safe_normalize(df["field_momentum"], method="rank")

    # --- Weighted combination → spatial_shrunk ---
    spatial_cols = ["norm_field", "norm_contested", "norm_evidence", "norm_color"]
    spatial_weights = [sc.R_WEIGHT_FIELD, sc.R_WEIGHT_CONTESTED,
                       sc.R_WEIGHT_EVIDENCE, sc.R_WEIGHT_EVIDENCE]

    if "norm_momentum" in df.columns:
        spatial_cols.append("norm_momentum")
        spatial_weights.append(sc.R_WEIGHT_EVIDENCE * 0.5)

    if "norm_field_momentum" in df.columns:
        spatial_cols.append("norm_field_momentum")
        spatial_weights.append(sc.R_WEIGHT_FIELD * 0.5)

    w = np.array(spatial_weights)
    components = df[spatial_cols].values

    # Weighted mean ignoring NaN: sum(w_i * x_i) / sum(w_i) for non-NaN
    mask = ~np.isnan(components)
    w_expanded = np.broadcast_to(w, components.shape)
    w_active = np.where(mask, w_expanded, 0)

    numerator = np.nansum(components * w_expanded, axis=1)
    denominator = np.sum(w_active, axis=1)
    denominator = np.where(denominator == 0, np.nan, denominator)

    df["spatial_shrunk"] = numerator / denominator

    # ════════════════════════════════════════════════════════════
    # TEMPORAL COMPONENTS
    # ════════════════════════════════════════════════════════════

    for wd in config.TEMPORAL_WINDOWS:
        # Temporal hex SE (shrunk consensus)
        col = f"weighted_se_{wd}d_shrunk"
        if col in df.columns:
            df[f"norm_se_{wd}d"] = _safe_normalize(df[col], method="rank")

        # Temporal hex SE (raw consensus)
        col_raw = f"weighted_se_{wd}d"
        if col_raw in df.columns:
            df[f"norm_se_raw_{wd}d"] = _safe_normalize(df[col_raw], method="rank")

        # Temporal contested SE
        col_c = f"contested_se_{wd}d"
        if col_c in df.columns:
            df[f"norm_contested_{wd}d"] = _safe_normalize(df[col_c], method="rank")

        # Temporal hop SE (most local ring)
        for hop in (1, 2, 3):
            col = f"hop{hop}_se_{wd}d_wmean"
            if col in df.columns:
                df[f"norm_hop{hop}_se_{wd}d"] = _safe_normalize(df[col], method="rank")

        # Temporal SE gradient (local vs wider)
        col_g = f"se_gradient_1to3_{wd}d"
        if col_g in df.columns:
            df[f"norm_gradient_{wd}d"] = _safe_normalize(df[col_g], method="rank")
        
        # Temporal SE confirmed (hop1 × hop3)
        col_conf = f"se_confirmed_{wd}d"
        if col_conf in df.columns:
            df[f"norm_confirmed_{wd}d"] = _safe_normalize(df[col_conf], method="rank")

        # Temporal geometry: local density
        col_d = f"local_density_{wd}d"
        if col_d in df.columns:
            df[f"norm_density_{wd}d"] = _safe_normalize(df[col_d], method="rank")

        # Temporal evidence: log(total_{wd}d)
        col_t = f"total_{wd}d"
        if col_t in df.columns:
            raw_ev = np.log1p(df[col_t].fillna(0).clip(0))
            df[f"norm_evidence_{wd}d"] = _safe_normalize(raw_ev, method="rank")

        # Temporal field (from score_temporal_windows)
        col_f = f"predicted_field_hex_{wd}d"
        if col_f in df.columns:
            df[f"norm_field_{wd}d"] = _safe_normalize(df[col_f], method="rank")

    # ════════════════════════════════════════════════════════════
    # TEMPORAL SPATIAL_SHRUNK per window
    # ════════════════════════════════════════════════════════════
    # Same weighted-mean logic as all-time, but using temporal norms

    for wd in config.TEMPORAL_WINDOWS:
        t_cols = [
            f"norm_field_{wd}d",
            f"norm_se_{wd}d",
            f"norm_contested_{wd}d",
            f"norm_evidence_{wd}d",
            f"norm_hop1_se_{wd}d",
            f"norm_hop2_se_{wd}d",
            f"norm_hop3_se_{wd}d",
            f"norm_gradient_{wd}d",
            f"norm_confirmed_{wd}d",
            f"norm_density_{wd}d",
        ]
        present = [c for c in t_cols if c in df.columns]

        if len(present) >= 2:
            t_components = df[present].values
            t_w = np.ones(len(present)) / len(present)  # equal weight
            t_mask = ~np.isnan(t_components)
            t_w_exp = np.broadcast_to(t_w, t_components.shape)
            t_w_active = np.where(t_mask, t_w_exp, 0)

            t_num = np.nansum(t_components * t_w_exp, axis=1)
            t_den = np.sum(t_w_active, axis=1)
            t_den = np.where(t_den == 0, np.nan, t_den)

            df[f"spatial_shrunk_{wd}d"] = t_num / t_den

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
    Also adds temporal composites: composite_score_{wd}d
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

    # --- Temporal composites ---
    for wd in config.TEMPORAL_WINDOWS:
        col = f"spatial_shrunk_{wd}d"
        if col in df.columns:
            df[f"composite_score_{wd}d"] = (
                df[col].fillna(0) * df["operational_score"]
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
        Plus temporal: spatial_shrunk_{wd}d, composite_score_{wd}d,
        norm_se_{wd}d, norm_contested_{wd}d, norm_momentum, etc.
    """
    print("\n--- R (PROMISE GOVERNOR) — COMPOSITE SCORING ---")

    df = compute_spatial_components(df_scored)

    # Report temporal norm coverage
    for wd in config.TEMPORAL_WINDOWS:
        col = f"norm_se_{wd}d"
        if col in df.columns:
            n_valid = df[col].notna().sum()
            print(f"  norm_se_{wd}d: {n_valid}/{len(df)} leads with temporal SE norm")

        col_f = f"norm_field_{wd}d"
        if col_f in df.columns:
            n_valid_f = df[col_f].notna().sum()
            print(f"  norm_field_{wd}d: {n_valid_f}/{len(df)} leads with temporal field norm")

        col_s = f"spatial_shrunk_{wd}d"
        if col_s in df.columns:
            n_valid_s = df[col_s].notna().sum()
            print(f"  spatial_shrunk_{wd}d: {n_valid_s}/{len(df)} leads with temporal spatial shrunk")

    if "norm_momentum" in df.columns:
        n_m = df["norm_momentum"].notna().sum()
        print(f"  norm_momentum: {n_m}/{len(df)} leads")

    if "norm_field_momentum" in df.columns:
        n_fm = df["norm_field_momentum"].notna().sum()
        print(f"  norm_field_momentum: {n_fm}/{len(df)} leads")

    #df = fuse_spatial_operational(df, df_ops_scored)

    return df
