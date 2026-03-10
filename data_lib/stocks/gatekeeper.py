"""
lib/gate.py

G (Gatekeeper) — Pre-filter layer.
Runs BEFORE spatial scoring.  Removes leads and partners that should
never reach the scoring engine.

Gates implemented:
    1. Density router   — route leads by local decision density
    2. Partner block    — non-responder gate (decline_rate > threshold)
    3. Capacity gate    — block partners whose pending exceeds capacity
    4. Shock gate       — block partners with active shocks from S

Returns:
    df_test      — filtered test leads with `density_regime` column
    df_partners  — filtered partner ops vector with `gate_blocked` flags
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

import data_lib.config as config
import data_lib.stocks.stocks_config as sc



def compute_lead_density(
    df_test: pd.DataFrame,
    df_train: pd.DataFrame,
    radius_m: float = None,
) -> pd.Series:
    """
    For each test lead, count settled decisions within radius_m
    using BallTree on training data.

    Returns: Series aligned to df_test index with integer counts.
    """
    if radius_m is None:
        radius_m = sc.GATE_DENSITY_RADIUS_M

    train_mask = df_train["final_decision"].isin(["INSTALLED", "DECLINED"])
    train_settled = df_train[train_mask]

    if train_settled.empty:
        return pd.Series(0, index=df_test.index, name="hard_density")

    train_rad = np.radians(train_settled[["latitude", "longitude"]].values)
    tree = BallTree(train_rad, metric="haversine")

    test_rad = np.radians(df_test[["latitude", "longitude"]].values)
    radius_rad = radius_m / config.EARTH_RADIUS_METER

    counts = tree.query_radius(test_rad, r=radius_rad, count_only=True)

    return pd.Series(counts, index=df_test.index, name="hard_density")


def density_router(df_test: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Gate 1: Density router.
    Adds `hard_density` and `density_regime` to df_test.
        Regime 1: dense area (hard_density >= threshold) — full scoring
        Regime 2: sparse area — may use different scoring strategy
    """
    print(f"[GATE] Density router (radius={sc.GATE_DENSITY_RADIUS_M}m, threshold={sc.GATE_DENSITY_THRESHOLD})...")

    df_test = df_test.copy()
    df_test["hard_density"] = compute_lead_density(df_test, df_train)
    df_test["density_regime"] = np.where(
        df_test["hard_density"] >= sc.GATE_DENSITY_THRESHOLD, 1, 2
    )

    n_r1 = (df_test["density_regime"] == 1).sum()
    n_r2 = (df_test["density_regime"] == 2).sum()
    print(f"[GATE]   Regime 1 (dense): {n_r1:,}  |  Regime 2 (sparse): {n_r2:,}")

    return df_test


def partner_block_gate(df_ops: pd.DataFrame) -> pd.DataFrame:
    """
    Gate 2: Non-responder gate.
    Flags partners with decline_rate > threshold AND enough observations.

    Adds `gate_nonresponder` (0/1) to df_ops.
    """
    df_ops = df_ops.copy()

    if "decline_rate_30d" not in df_ops.columns:
        df_ops["gate_nonresponder"] = 0
        return df_ops

    mask = (
        (df_ops["decline_rate_30d"] >= sc.GATE_DECLINE_RATE_BLOCK)
        & (df_ops["total_decisions"] >= sc.GATE_DECLINE_RATE_MIN_OBS)
    )
    df_ops["gate_nonresponder"] = mask.astype(int)

    n_blocked = mask.sum()
    print(f"[GATE] Non-responder gate: {n_blocked} partners blocked "
          f"(decline_rate >= {sc.GATE_DECLINE_RATE_BLOCK}, min_obs >= {sc.GATE_DECLINE_RATE_MIN_OBS})")

    return df_ops


def capacity_gate(df_ops: pd.DataFrame) -> pd.DataFrame:
    """
    Gate 3: Capacity overload gate.
    Flags partners whose pending leads >= factor × expected_daily_slots.

    Adds `gate_capacity` (0/1) to df_ops.
    """
    df_ops = df_ops.copy()

    has_cols = (
        "nmbr_active_leads" in df_ops.columns
        and "expected_daily_slots" in df_ops.columns
    )
    if not has_cols:
        df_ops["gate_capacity"] = 0
        return df_ops

    threshold = sc.GATE_CAPACITY_OVERLOAD_FACTOR * df_ops["expected_daily_slots"].fillna(0)
    mask = df_ops["nmbr_active_leads"].fillna(0) >= threshold
    # Don't block if expected_daily_slots is NaN or 0 (no data, not necessarily overloaded)
    mask = mask & (df_ops["expected_daily_slots"].fillna(0) > 0)

    df_ops["gate_capacity"] = mask.astype(int)

    n_blocked = mask.sum()
    print(f"[GATE] Capacity gate: {n_blocked} partners blocked "
          f"(pending >= {sc.GATE_CAPACITY_OVERLOAD_FACTOR}× expected_daily_slots)")

    return df_ops


def shock_gate(df_ops: pd.DataFrame) -> pd.DataFrame:
    """
    Gate 4: Shock gate.
    Flags partners with active shocks from S ledger.

    Adds `gate_shock` (0/1) to df_ops.
    """
    df_ops = df_ops.copy()

    if "has_shock" not in df_ops.columns:
        df_ops["gate_shock"] = 0
        return df_ops

    df_ops["gate_shock"] = (df_ops["has_shock"] > 0).astype(int)

    n_blocked = df_ops["gate_shock"].sum()
    print(f"[GATE] Shock gate: {n_blocked} partners blocked (active shocks)")

    return df_ops


def compute_gate_blocked(df_ops: pd.DataFrame) -> pd.DataFrame:
    """
    Combines all gate flags into a single `gate_blocked` column.
    A partner is blocked if ANY gate fires.
    """
    df_ops = df_ops.copy()

    gate_cols = [c for c in df_ops.columns if c.startswith("gate_")]
    if not gate_cols:
        df_ops["gate_blocked"] = 0
        return df_ops

    df_ops["gate_blocked"] = df_ops[gate_cols].max(axis=1).astype(int)

    n_total = len(df_ops)
    n_blocked = df_ops["gate_blocked"].sum()
    print(f"[GATE] TOTAL: {n_blocked}/{n_total} partners blocked by at least one gate")

    return df_ops


def run_gates(
    df_test: pd.DataFrame,
    df_train: pd.DataFrame,
    df_ops: pd.DataFrame,
) -> tuple:
    """
    Full gatekeeper pipeline.

    Args:
        df_test  : test leads (mobile, lat, lng, ...)
        df_train : training decisions (lat, lng, final_decision, ...)
        df_ops   : partner ops vector from build_partner_ops_vector()

    Returns:
        df_test  : with hard_density, density_regime columns added
        df_ops   : with gate_nonresponder, gate_capacity, gate_shock, gate_blocked columns
    """
    print("\n--- G (GATEKEEPER) ---")

    # Lead-level gates
    df_test = density_router(df_test, df_train)

    # Partner-level gates
    if df_ops.empty:
        print("[GATE] WARNING: empty ops vector, skipping partner gates")
        df_ops = pd.DataFrame(columns=["partner_id", "gate_blocked"])
    else:
        df_ops = partner_block_gate(df_ops)
        df_ops = capacity_gate(df_ops)
        df_ops = shock_gate(df_ops)
        df_ops = compute_gate_blocked(df_ops)

    return df_test, df_ops