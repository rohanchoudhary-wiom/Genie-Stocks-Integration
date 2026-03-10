"""
lib/feature/ops_features.py

Normalises raw ops-vector columns into [0, 1] signals suitable for
the R composite formula.

Three signal families (per the architecture doc):
    CAPACITY     — nmbr_active_leads, active_tickets, queue_velocity
    RELIABILITY  — late_arrive_median, plan_created_rate, late_severity_max
    INFRASTRUCTURE — has_shock (binary)

Each family is reduced to a single 0–1 score.
The three family scores are combined into `operational_score`.
"""

import numpy as np
import pandas as pd

import data_lib.stocks.stocks_config as sc


def _clip_and_invert(series: pd.Series, high: float) -> pd.Series:
    """
    Normalise to [0, 1] then invert so that LOWER raw values → HIGHER score.
    Useful for load/tardiness metrics where less is better.
    """
    clipped = series.clip(0, high) / high
    return 1.0 - clipped


def _clip_direct(series: pd.Series, high: float) -> pd.Series:
    """
    Normalise to [0, 1] directly — higher raw → higher score.
    Useful for queue_velocity, plan_created_rate.
    """
    return (series.clip(0, high) / high)


def compute_capacity_score(df_ops: pd.DataFrame) -> pd.Series:
    """
    CAPACITY score: lower pending + lower tickets + higher velocity → better.

    Components:
        pending_norm   = 1 - clip(nmbr_active_leads / 10)     inverted
        ticket_norm    = 1 - clip(active_tickets / 5)          inverted
        velocity_norm  = clip(queue_velocity / 1.0)            direct (0–1 already)

    Simple average of available components.
    """
    parts = []

    if "nmbr_active_leads" in df_ops.columns:
        parts.append(_clip_and_invert(df_ops["nmbr_active_leads"].fillna(0), high=10.0))

    if "active_tickets" in df_ops.columns:
        parts.append(_clip_and_invert(df_ops["active_tickets"].fillna(0), high=5.0))

    if "queue_velocity" in df_ops.columns:
        parts.append(_clip_direct(df_ops["queue_velocity"].fillna(0), high=1.0))

    if not parts:
        return pd.Series(np.nan, index=df_ops.index, name="capacity_score")

    stacked = np.column_stack(parts)
    return pd.Series(np.nanmean(stacked, axis=1), index=df_ops.index, name="capacity_score")


def compute_reliability_score(df_ops: pd.DataFrame) -> pd.Series:
    """
    RELIABILITY score: lower tardiness + higher plan rate → better.

    Components:
        late_norm        = 1 - clip(late_arrive_median / 7)         inverted
        severity_norm    = 1 - clip(late_severity_max / 3)          inverted
        plan_rate_norm   = clip(plan_created_rate / 1.0)            direct

    Simple average.
    """
    parts = []

    if "late_arrive_median" in df_ops.columns:
        parts.append(_clip_and_invert(df_ops["late_arrive_median"].fillna(0), high=7.0))

    if "late_severity_max" in df_ops.columns:
        parts.append(_clip_and_invert(df_ops["late_severity_max"].fillna(0), high=3.0))

    if "plan_created_rate" in df_ops.columns:
        parts.append(_clip_direct(df_ops["plan_created_rate"].fillna(0), high=1.0))

    if not parts:
        return pd.Series(np.nan, index=df_ops.index, name="reliability_score")

    stacked = np.column_stack(parts)
    return pd.Series(np.nanmean(stacked, axis=1), index=df_ops.index, name="reliability_score")


def compute_infrastructure_score(df_ops: pd.DataFrame) -> pd.Series:
    """
    INFRASTRUCTURE score: no shock → 1.0, shock → 0.0.
    Binary for now; will add severity weighting when outage data matures.
    """
    if "has_shock" in df_ops.columns:
        return pd.Series(
            1.0 - df_ops["has_shock"].fillna(0).clip(0, 1),
            index=df_ops.index,
            name="infrastructure_score",
        )
    return pd.Series(1.0, index=df_ops.index, name="infrastructure_score")


def compute_operational_score(df_ops: pd.DataFrame) -> pd.DataFrame:
    """
    Combines the three signal families into a single `operational_score` per partner.

    operational_score = mean(capacity, reliability, infrastructure)
                        floored at R_OPS_FLOOR to avoid zeroing good spatial signals.

    Adds columns: capacity_score, reliability_score, infrastructure_score, operational_score.
    """
    df = df_ops.copy()

    df["capacity_score"] = compute_capacity_score(df)
    df["reliability_score"] = compute_reliability_score(df)
    df["infrastructure_score"] = compute_infrastructure_score(df)

    # Average of available family scores (handles NaN gracefully)
    family_cols = ["capacity_score", "reliability_score", "infrastructure_score"]
    df["operational_score"] = df[family_cols].mean(axis=1)

    # Floor so that sparse ops data doesn't zero out spatial signal
    df["operational_score"] = df["operational_score"].clip(lower=sc.R_OPS_FLOOR)

    # Partners blocked by G get operational_score = 0
    if "gate_blocked" in df.columns:
        df.loc[df["gate_blocked"] == 1, "operational_score"] = 0.0

    print(f"[OPS FEATURES] operational_score — "
          f"mean={df['operational_score'].mean():.3f}, "
          f"median={df['operational_score'].median():.3f}, "
          f"zeros={( df['operational_score'] == 0).sum()}")

    return df