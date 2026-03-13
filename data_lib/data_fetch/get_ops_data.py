"""
lib/data_fetch/get_ops_data.py

Snowflake queries that feed stocks B_operational, E, S, and G.
Uses the same _query_snowflake_df() helper from get_data.py.
"""

import pandas as pd
import numpy as np
from data_lib.data_fetch.get_data import _query_snowflake_df
import data_lib.config as config
import data_lib.stocks.stocks_config as sc


# =====================================================================
# B_OPERATIONAL — CAPACITY
# =====================================================================

def get_pending_leads_per_partner(start_dt: str, end_dt: str) -> pd.DataFrame:
    """
    Lead lifecycle reconstruction from task_logs + booking_logs.
    Returns one row per (mobile, partner_id) with terminal event timestamps.
    Rows where ALL terminal events are NULL → pending (lives in E).

    Feeds: nmbr_active_leads, long_held_leads, queue_velocity per partner.
    """
    query = f"""
    WITH interest AS (
        SELECT
            mobile,
            TRY_PARSE_JSON(data):partner_id::STRING  AS partner_id,
            added_time                                 AS interest_time,
            ROW_NUMBER() OVER (
                PARTITION BY mobile, TRY_PARSE_JSON(data):partner_id::STRING
                ORDER BY added_time DESC
            ) AS rn
        FROM prod_db.public.task_logs
        WHERE event_name = 'NOTIF_SENT'
          AND CAST(added_time AS DATE) BETWEEN '{start_dt}' AND '{end_dt}'
    ),
    decline AS (
        SELECT
            mobile,
            TRY_PARSE_JSON(data):partner_id::STRING  AS partner_id,
            added_time                                 AS decline_time,
            ROW_NUMBER() OVER (
                PARTITION BY mobile, TRY_PARSE_JSON(data):partner_id::STRING
                ORDER BY added_time ASC
            ) AS rn
        FROM prod_db.public.task_logs
        WHERE event_name IN ('DECLINED','NOT_INTERESTED')
          AND CAST(added_time AS DATE) BETWEEN '{start_dt}' AND '{end_dt}'
    ),
    install AS (
        SELECT
            mobile,
            TRY_PARSE_JSON(data):lco_account_id::STRING AS partner_id,
            added_time                                    AS install_time,
            ROW_NUMBER() OVER (
                PARTITION BY mobile, TRY_PARSE_JSON(data):lco_account_id::STRING
                ORDER BY added_time ASC
            ) AS rn
        FROM prod_db.public.booking_logs
        WHERE event_name = 'lead_state_changed'
          AND TRY_PARSE_JSON(data):state::STRING = 'installed'
          AND CAST(added_time AS DATE) BETWEEN '{start_dt}' AND '{end_dt}'
    ),
    cancel AS (
        SELECT
            mobile,
            TRY_PARSE_JSON(data):lco_account_id::STRING AS partner_id,
            added_time                                    AS cancel_time,
            ROW_NUMBER() OVER (
                PARTITION BY mobile, TRY_PARSE_JSON(data):lco_account_id::STRING
                ORDER BY added_time ASC
            ) AS rn
        FROM prod_db.public.booking_logs
        WHERE event_name = 'lead_state_changed'
          AND TRY_PARSE_JSON(data):state::STRING IN ('cancelled','unservisable','refunded')
          AND CAST(added_time AS DATE) BETWEEN '{start_dt}' AND '{end_dt}'
    )
    SELECT
        i.mobile,
        i.partner_id,
        i.interest_time,
        d.decline_time,
        inst.install_time,
        c.cancel_time,
        COALESCE(d.decline_time, inst.install_time, c.cancel_time) AS decision_time
    FROM interest i
    LEFT JOIN decline  d    ON i.mobile = d.mobile    AND i.partner_id = d.partner_id    AND d.rn = 1
    LEFT JOIN install  inst ON i.mobile = inst.mobile AND i.partner_id = inst.partner_id AND inst.rn = 1
    LEFT JOIN cancel   c    ON i.mobile = c.mobile    AND i.partner_id = c.partner_id    AND c.rn = 1
    WHERE i.rn = 1
    """
    try:
        df = _query_snowflake_df(query)
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.lower()
        for col in ["interest_time", "decline_time", "install_time", "cancel_time", "decision_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df["mobile"] = df["mobile"].astype(str)
        df["partner_id"] = df["partner_id"].astype(str)
        return df
    except Exception as e:
        print(f"[get_pending_leads_per_partner] ERROR: {e}")
        return pd.DataFrame()


def compute_lead_capacity(df_leads: pd.DataFrame, eval_time: pd.Timestamp = None) -> pd.DataFrame:
    """
    From lead lifecycle rows, compute per-partner capacity metrics at eval_time.

    Returns: partner_id, nmbr_active_leads, long_held_leads_24h,
             resolved_leads, queue_velocity
    """
    if df_leads.empty:
        return pd.DataFrame(columns=[
            "partner_id", "nmbr_active_leads", "long_held_leads_24h",
            "resolved_leads", "queue_velocity",
        ])

    if eval_time is None:
        eval_time = pd.Timestamp.now()

    df = df_leads.copy()
    df["is_pending"] = df["decision_time"].isna().astype(int)
    df["is_resolved"] = (~df["decision_time"].isna()).astype(int)
    df["hours_open"] = (eval_time - df["interest_time"]).dt.total_seconds() / 3600.0
    df["is_long_held"] = ((df["is_pending"] == 1) & (df["hours_open"] > 24)).astype(int)

    agg = df.groupby("partner_id").agg(
        nmbr_active_leads=("is_pending", "sum"),
        long_held_leads_24h=("is_long_held", "sum"),
        resolved_leads=("is_resolved", "sum"),
        total_leads=("mobile", "count"),
    ).reset_index()

    agg["queue_velocity"] = (
        agg["resolved_leads"] / agg["total_leads"].replace(0, np.nan)
    ).fillna(0).round(4)

    return agg.drop(columns=["total_leads"])


# =====================================================================
# B_OPERATIONAL — TICKET TASKS
# =====================================================================

def get_ticket_tasks(start_dt: str, end_dt: str) -> pd.DataFrame:
    """
    Service/pickup/cash_collection tickets assigned to partners.
    Uses LEAD() window to compute how long partner held each task.
    """
    types_str = ",".join([f"'{t}'" for t in sc.OPS_TICKET_TYPES])

    query = f"""
    WITH assign_events AS (
        SELECT
            account_id                     AS long_customer_id,
            task_id,
            added_time                     AS assign_time,
            event_name                     AS assign_type,
            title,
            assigned_account_id            AS partner_id,
            type                           AS ticket_type,
            LEAD(added_time) OVER (
                PARTITION BY task_id ORDER BY added_time
            )                              AS next_event_time,
            status
        FROM prod_db.public.ticketvanilla_audit
        WHERE event_name IN ('ticket_assigned', 'ticket_reopened', 'task_created')
          AND LOWER(assigned_to) = 'partner'
          AND LOWER(type) IN ({types_str})
          AND CAST(added_time AS DATE) BETWEEN '{start_dt}' AND '{end_dt}'
    )
    SELECT
        long_customer_id,
        task_id,
        partner_id,
        ticket_type,
        assign_time,
        next_event_time,
        DATEDIFF(MINUTE, assign_time, COALESCE(next_event_time, CURRENT_TIMESTAMP)) AS tat_open_minutes,
        CASE WHEN LOWER(status) = 'resolved' THEN 1 ELSE 0 END AS is_resolved,
        CASE WHEN next_event_time IS NULL THEN 1 ELSE 0 END     AS is_still_open
    FROM assign_events
    """
    try:
        df = _query_snowflake_df(query)
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.lower()
        df["assign_time"] = pd.to_datetime(df["assign_time"], errors="coerce")
        df["partner_id"] = df["partner_id"].astype(str)
        df["tat_open_minutes"] = pd.to_numeric(df["tat_open_minutes"], errors="coerce")
        return df
    except Exception as e:
        print(f"[get_ticket_tasks] ERROR: {e}")
        return pd.DataFrame()


def compute_ticket_capacity(df_tickets: pd.DataFrame) -> pd.DataFrame:
    """
    Per-partner ticket load aggregation.

    Returns: partner_id, active_tickets, median_tat_min, max_tat_min,
             ticket_resolution_rate
    """
    if df_tickets.empty:
        return pd.DataFrame(columns=[
            "partner_id", "active_tickets", "median_tat_min",
            "max_tat_min", "ticket_resolution_rate",
        ])

    agg = df_tickets.groupby("partner_id").agg(
        active_tickets=("is_still_open", "sum"),
        total_tickets=("task_id", "nunique"),
        resolved_tickets=("is_resolved", "sum"),
        median_tat_min=("tat_open_minutes", "median"),
        max_tat_min=("tat_open_minutes", "max"),
    ).reset_index()

    agg["ticket_resolution_rate"] = (
        agg["resolved_tickets"] / agg["total_tickets"].replace(0, np.nan)
    ).fillna(0).round(4)

    return agg.drop(columns=["total_tickets", "resolved_tickets"])


# =====================================================================
# B_OPERATIONAL — SLOT CONFIRMATION / RELIABILITY
# =====================================================================

def get_slot_confirmation(start_dt: str, end_dt: str = None) -> pd.DataFrame:
    """
    Slot confirmation → plan creation join.
    Measures partner tardiness: gap between slot and plan.
    """
    end_ref = f"'{end_dt}'" if end_dt else "CURRENT_TIMESTAMP"

    query = f"""
    WITH slots AS (
        SELECT
            mobile,
            TRY_PARSE_JSON(data):partner_id::STRING       AS partner_id,
            TRY_PARSE_JSON(data):slot_selected::TIMESTAMP  AS slot_selected_time,
            TRY_PARSE_JSON(data):nearestDistance::FLOAT     AS nearest_distance,
            added_time                                      AS slot_confirmed_time,
            ROW_NUMBER() OVER (
                PARTITION BY mobile ORDER BY added_time DESC
            ) AS rn
        FROM prod_db.public.task_logs
        WHERE event_name = 'CUSTOMER_SLOT_CONFIRMED'
          AND added_time >= '{start_dt}' and added_time <= {end_ref}
    ),
    plans AS (
        SELECT
            mobile,
            added_time AS plan_created_time,
            ROW_NUMBER() OVER (
                PARTITION BY mobile ORDER BY added_time ASC
            ) AS rn
        FROM prod_db.public.booking_logs
        WHERE event_name = 'trum_plan_created'
          AND added_time >= '{start_dt}' and added_time <= {end_ref}
    )
    SELECT
        s.mobile,
        s.partner_id,
        s.slot_selected_time,
        s.slot_confirmed_time,
        s.nearest_distance,
        p.plan_created_time,
        CASE WHEN p.mobile IS NOT NULL THEN 1 ELSE 0 END AS plan_created_flag,
        DATEDIFF(SECOND, s.slot_confirmed_time,
                 COALESCE(p.plan_created_time, CURRENT_TIMESTAMP)) / 86400.0 AS late_arrive_days
    FROM slots s
    LEFT JOIN plans p ON s.mobile = p.mobile AND p.rn = 1
    WHERE s.rn = 1
    """
    try:
        df = _query_snowflake_df(query)
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.lower()
        df["partner_id"] = df["partner_id"].astype(str)
        for col in ["slot_selected_time", "slot_confirmed_time", "plan_created_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df["late_arrive_days"] = pd.to_numeric(df["late_arrive_days"], errors="coerce")
        df["nearest_distance"] = pd.to_numeric(df["nearest_distance"], errors="coerce")
        return df
    except Exception as e:
        print(f"[get_slot_confirmation] ERROR: {e}")
        return pd.DataFrame()


def compute_reliability(df_slots: pd.DataFrame) -> pd.DataFrame:
    """
    Per-partner reliability features from slot confirmation data.

    Returns: partner_id, late_arrive_median, late_arrive_max,
             late_severity_max, late_close_penalty, plan_created_rate,
             planning_strength
    """
    if df_slots.empty:
        return pd.DataFrame(columns=[
            "partner_id", "late_arrive_median", "late_arrive_max",
            "late_severity_max", "late_close_penalty", "plan_created_rate",
            "planning_strength",
        ])

    df = df_slots.copy()

    df["late_arrive_feature"] = np.where(df["late_arrive_days"] > 1, df["late_arrive_days"], 0)

    bins = [-np.inf, 0, 3, 7, np.inf]
    labels = [0, 1, 2, 3]
    df["late_severity"] = pd.cut(
        df["late_arrive_days"], bins=bins, labels=labels, right=True
    ).astype(float).fillna(0).astype(int)

    df["late_close_penalty"] = (
        df["late_arrive_feature"] * (1.0 / (df["nearest_distance"].fillna(0) + 1.0))
    )

    agg = df.groupby("partner_id").agg(
        late_arrive_median=("late_arrive_feature", "median"),
        late_arrive_max=("late_arrive_feature", "max"),
        late_arrive_sum=("late_arrive_feature", "sum"),
        late_severity_max=("late_severity", "max"),
        late_close_penalty=("late_close_penalty", "sum"),
        plan_created_rate=("plan_created_flag", "mean"),
        total_slots=("mobile", "count"),
    ).reset_index()

    agg["planning_strength"] = agg["plan_created_rate"] * agg["total_slots"]

    return agg.drop(columns=["total_slots", "late_arrive_sum"])


# =====================================================================
# G + B_OPERATIONAL — PER-PARTNER SE + DECLINE RATE + RESPONSE TIME
# =====================================================================

def _build_perf_window_sql(wd: int, max_wd: int, end_ref: str) -> str:
    """
    Build conditional aggregate columns for one temporal window.
    Mobile-level SE: COUNT(DISTINCT mobile) not COUNT(*).
    Largest window needs no date filter (already in WHERE clause).
    """
    if wd == max_wd:
        date_pred = ""
    else:
        date_pred = f"first_notified_time >= DATEADD(DAY, -{wd}, {end_ref}) AND "

    return f"""
        COUNT(DISTINCT CASE WHEN {date_pred}1=1 THEN mobile END)                                      AS total_{wd}d,
        COUNT(DISTINCT CASE WHEN {date_pred}partner_id = lco_account_installed THEN mobile END)        AS installs_{wd}d,
        COUNT(DISTINCT CASE WHEN {date_pred}first_event = 'DECLINED' THEN mobile END)                  AS declines_{wd}d,
        MEDIAN(CASE WHEN {date_pred}1=1 THEN reaction_time_notif END)                                  AS median_response_min_{wd}d"""


def get_partner_performance(end_dt: str = None) -> pd.DataFrame:
    """
    Per-partner SE, decline rate, and median response time from
    t_node_decisions_active.

    One Snowflake round-trip. All windows from config.TEMPORAL_WINDOWS.
    SE computed at mobile level (COUNT DISTINCT mobile).

    Feeds: G (non-responder gate) + B_operational + RF feature matrix.
    """
    end_ref = f"'{end_dt}'" if end_dt else "CURRENT_DATE"
    max_wd = max(config.TEMPORAL_WINDOWS)

    window_fragments = [
        _build_perf_window_sql(wd, max_wd, end_ref)
        for wd in config.TEMPORAL_WINDOWS
    ]

    query = f"""
    SELECT
        partner_id,
        {','.join(window_fragments)}
    FROM t_node_decisions_active
    WHERE first_notified_time >= DATEADD(DAY, -{max_wd}, {end_ref})
      AND first_notified_time <= {end_ref}
      AND first_event IS NOT NULL
      AND first_event NOT IN ('', 'None')
    GROUP BY partner_id
    """
    try:
        df = _query_snowflake_df(query)
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.lower()
        df["partner_id"] = df["partner_id"].astype(str)

        # ── Type-cast all window columns ──
        for wd in config.TEMPORAL_WINDOWS:
            for col in [f"total_{wd}d", f"installs_{wd}d", f"declines_{wd}d"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            df[f"median_response_min_{wd}d"] = pd.to_numeric(
                df[f"median_response_min_{wd}d"], errors="coerce"
            )

        # ── Derived rates per window ──
        for wd in config.TEMPORAL_WINDOWS:
            tot = df[f"total_{wd}d"].replace(0, np.nan)
            df[f"se_{wd}d"] = (df[f"installs_{wd}d"] / tot).fillna(0)
            df[f"decline_rate_{wd}d"] = (df[f"declines_{wd}d"] / tot).fillna(0)

        # ── Backward-compat aliases (gatekeeper + shock ledger read these) ──
        df["total_decisions"] = df[f"total_{max_wd}d"]
        df["median_response_min"] = df[f"median_response_min_{max_wd}d"]

        return df
    except Exception as e:
        print(f"[get_partner_performance] ERROR: {e}")
        return pd.DataFrame()


def get_expected_daily_slots(lookback_days: int = 45, end_dt: str = None) -> pd.DataFrame:
    """
    Rolling average daily installs per partner from booking_logs.
    """
    end_ref = f"'{end_dt}'" if end_dt else "CURRENT_DATE"
    query = f"""
    WITH daily AS (
        SELECT
            TRY_PARSE_JSON(data):lco_account_id::STRING AS partner_id,
            CAST(added_time AS DATE)                     AS dt,
            COUNT(*)                                     AS daily_installs
        FROM prod_db.public.booking_logs
        WHERE event_name = 'lead_state_changed'
          AND TRY_PARSE_JSON(data):state::STRING = 'installed'
          AND added_time >= DATEADD(DAY, -{lookback_days}, {end_ref})
          AND added_time <= {end_ref}
        GROUP BY 1, 2
    )
    SELECT
        partner_id,
        AVG(daily_installs)                AS expected_daily_slots,
        COUNT(DISTINCT dt)                 AS active_days,
        SUM(daily_installs)                AS total_installs_period
    FROM daily
    GROUP BY partner_id
    """
    try:
        df = _query_snowflake_df(query)
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.lower()
        df["partner_id"] = df["partner_id"].astype(str)
        df["expected_daily_slots"] = pd.to_numeric(df["expected_daily_slots"], errors="coerce")
        return df
    except Exception as e:
        print(f"[get_expected_daily_slots] ERROR: {e}")
        return pd.DataFrame()


# =====================================================================
# S — SHOCK LEDGER
# =====================================================================

def get_active_outages(recency_days: int = 7, end_dt: str = None) -> pd.DataFrame:
    """
    Active outages from outage_incidents_aggregated.
    """
    end_ref = f"'{end_dt}'" if end_dt else "CURRENT_DATE"
    query = f"""
    SELECT
        partner_id,
        severity,
        device_count,
        size_bucket,
        status,
        created_at_ist                                                        AS outage_time,
        DATEDIFF(HOUR, created_at_ist, CURRENT_TIMESTAMP)                     AS outage_recency_hours
    FROM prod_db.BUSINESS_EFFICIENCY_ROUTER_OUTAGE_DETECTION_AUDIT_PUBLIC.outage_incidents_aggregated
    WHERE LOWER(status) = 'active'
      AND created_at_ist >= DATEADD(DAY, -{recency_days}, {end_ref})
      AND created_at_ist <= {end_ref}
    """
    try:
        df = _query_snowflake_df(query)
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.lower()
        df["partner_id"] = df["partner_id"].astype(str)
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(0)
        df["device_count"] = pd.to_numeric(df["device_count"], errors="coerce").fillna(0)
        return df
    except Exception as e:
        print(f"[get_active_outages] ERROR: {e}")
        return pd.DataFrame()


def compute_shock_flags(
    df_outages: pd.DataFrame,
    df_perf: pd.DataFrame,
    df_lead_cap: pd.DataFrame,
    df_slots: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds the shock ledger: transient events that should block or throttle partners.

    shock_type in: 'outage', 'decline_spike', 'capacity_overload', 'ticket_overload'
    """
    records = []

    # 1. Outage shocks
    if not df_outages.empty:
        outage_agg = df_outages.groupby("partner_id").agg(
            max_severity=("severity", "max"),
            total_devices=("device_count", "sum"),
            min_recency_hours=("outage_recency_hours", "min"),
        ).reset_index()
        for _, row in outage_agg.iterrows():
            if row["max_severity"] >= sc.SHOCK_OUTAGE_SEVERITY_THRESHOLD:
                records.append({
                    "partner_id": row["partner_id"],
                    "shock_type": "outage",
                    "severity": int(row["max_severity"]),
                    "detail": f"devices={int(row['total_devices'])}, recency_h={row['min_recency_hours']:.0f}",
                })

    # 2. Decline spike
    if not df_perf.empty:
        spike_mask = (
            (df_perf["decline_rate_30d"] >= sc.GATE_DECLINE_RATE_BLOCK)
            & (df_perf["total_decisions"] >= sc.GATE_DECLINE_RATE_MIN_OBS)
        )
        for _, row in df_perf[spike_mask].iterrows():
            records.append({
                "partner_id": row["partner_id"],
                "shock_type": "decline_spike",
                "severity": 2,
                "detail": f"decline_rate={row['decline_rate_30d']:.2f}, n={int(row['total_decisions'])}",
            })

    # 3. Capacity overload
    if not df_lead_cap.empty and not df_slots.empty:
        merged = df_lead_cap.merge(
            df_slots[["partner_id", "expected_daily_slots"]],
            on="partner_id", how="inner",
        )
        overload_mask = (
            merged["nmbr_active_leads"]
            > sc.SHOCK_CAPACITY_FACTOR * merged["expected_daily_slots"]
        )
        for _, row in merged[overload_mask].iterrows():
            records.append({
                "partner_id": row["partner_id"],
                "shock_type": "capacity_overload",
                "severity": 2,
                "detail": f"pending={int(row['nmbr_active_leads'])}, expected_slots={row['expected_daily_slots']:.1f}",
            })

    if not records:
        return pd.DataFrame(columns=["partner_id", "shock_type", "severity", "detail"])

    return pd.DataFrame(records)


# =====================================================================
# DELTAS — snapshot + trend columns for RF
# =====================================================================

def _compute_ops_deltas(ops: pd.DataFrame) -> pd.DataFrame:
    """
    For each pair (smallest_window, larger_window), compute:
        se_delta_{s}_{l}
        decline_rate_delta_{s}_{l}
        response_delta_{s}_{l}

    Smallest window = most recent snapshot, delta = recent - baseline.
    Positive se_delta → partner is improving.
    Positive decline_rate_delta → partner is getting worse.
    """
    windows = sorted(config.TEMPORAL_WINDOWS)
    smallest = windows[0]

    for wd in windows[1:]:
        ops[f"se_delta_{smallest}_{wd}"] = (
            ops[f"se_{smallest}d"] - ops[f"se_{wd}d"]
        )
        ops[f"decline_rate_delta_{smallest}_{wd}"] = (
            ops[f"decline_rate_{smallest}d"] - ops[f"decline_rate_{wd}d"]
        )
        ops[f"response_delta_{smallest}_{wd}"] = (
            ops[f"median_response_min_{smallest}d"].fillna(0)
            - ops[f"median_response_min_{wd}d"].fillna(0)
        )

    return ops


# =====================================================================
# COMBINED: build_partner_ops_vector
# =====================================================================

def build_partner_ops_vector(start_dt: str, end_dt: str) -> pd.DataFrame:
    """
    Master function: pulls all operational data, computes features,
    returns one row per partner_id with the full ops vector.

    Temporal columns per window in config.TEMPORAL_WINDOWS:
        se_{wd}d, decline_rate_{wd}d, median_response_min_{wd}d,
        installs_{wd}d, declines_{wd}d, total_{wd}d

    Delta columns (smallest vs each larger window):
        se_delta_{s}_{l}, decline_rate_delta_{s}_{l}, response_delta_{s}_{l}

    Static columns:
        partner_id,
        nmbr_active_leads, long_held_leads_24h, queue_velocity,
        active_tickets, median_tat_min, max_tat_min, ticket_resolution_rate,
        late_arrive_median, late_arrive_max, late_severity_max,
        late_close_penalty, plan_created_rate, planning_strength,
        expected_daily_slots, active_days,
        has_shock, shock_types,
        total_decisions, median_response_min  (backward-compat aliases)
    """
    print("[OPS VECTOR] Pulling lead lifecycle...")
    df_leads_raw = get_pending_leads_per_partner(start_dt, end_dt)
    eval_time = pd.Timestamp(end_dt)
    df_lead_cap = compute_lead_capacity(df_leads_raw, eval_time)

    print("[OPS VECTOR] Pulling ticket tasks...")
    df_tickets_raw = get_ticket_tasks(start_dt, end_dt)
    df_ticket_cap = compute_ticket_capacity(df_tickets_raw)

    print("[OPS VECTOR] Pulling slot confirmation...")
    df_slots_raw = get_slot_confirmation(sc.OPS_SLOT_LOOKBACK_DATE, end_dt=end_dt)
    df_reliability = compute_reliability(df_slots_raw)

    print(f"[OPS VECTOR] Pulling partner performance (windows={config.TEMPORAL_WINDOWS})...")
    df_perf = get_partner_performance(end_dt=end_dt)

    print("[OPS VECTOR] Pulling expected daily slots...")
    df_slots = get_expected_daily_slots(lookback_days=sc.OPS_SLOTS_ROLLING_DAYS, end_dt=end_dt)

    print("[OPS VECTOR] Computing shock flags...")
    df_outages = get_active_outages(recency_days=sc.SHOCK_OUTAGE_RECENCY_DAYS, end_dt=end_dt)
    df_shocks = compute_shock_flags(df_outages, df_perf, df_lead_cap, df_slots)

    # --- Merge all into a single partner-level frame ---
    if df_perf.empty:
        print("[OPS VECTOR] WARNING: partner performance is empty, returning empty ops vector")
        return pd.DataFrame()

    ops = df_perf.copy()

    if not df_lead_cap.empty:
        ops = ops.merge(df_lead_cap, on="partner_id", how="left")

    if not df_ticket_cap.empty:
        ops = ops.merge(df_ticket_cap, on="partner_id", how="left")

    if not df_reliability.empty:
        ops = ops.merge(df_reliability, on="partner_id", how="left")

    if not df_slots.empty:
        ops = ops.merge(
            df_slots[["partner_id", "expected_daily_slots", "active_days"]],
            on="partner_id", how="left",
        )

    # Shock summary per partner
    if not df_shocks.empty:
        shock_agg = df_shocks.groupby("partner_id").agg(
            has_shock=("shock_type", "count"),
            shock_types=("shock_type", lambda x: "|".join(sorted(set(x)))),
        ).reset_index()
        shock_agg["has_shock"] = (shock_agg["has_shock"] > 0).astype(int)
        ops = ops.merge(shock_agg, on="partner_id", how="left")
    else:
        ops["has_shock"] = 0
        ops["shock_types"] = ""

    # Fill NaN for partners missing some data sources
    fill_zero_cols = [
        "nmbr_active_leads", "long_held_leads_24h", "active_tickets",
        "has_shock",
    ]
    for col in fill_zero_cols:
        if col in ops.columns:
            ops[col] = ops[col].fillna(0)

    ops["shock_types"] = ops["shock_types"].fillna("")

    # ── Compute deltas ──
    ops = _compute_ops_deltas(ops)

    print(f"[OPS VECTOR] Built ops vector for {len(ops)} partners, {len(ops.columns)} columns")

    # ── Report ──
    for wd in config.TEMPORAL_WINDOWS:
        col = f"se_{wd}d"
        if col in ops.columns:
            print(f"  {col}: mean={ops[col].mean():.4f}  median={ops[col].median():.4f}")
    smallest = min(config.TEMPORAL_WINDOWS)
    for wd in sorted(config.TEMPORAL_WINDOWS):
        if wd == smallest:
            continue
        dcol = f"se_delta_{smallest}_{wd}"
        if dcol in ops.columns:
            print(f"  {dcol}: mean={ops[dcol].mean():.4f}  std={ops[dcol].std():.4f}")

    return ops


# =====================================================================
# __main__ — run standalone
# =====================================================================

 
if __name__ == "__main__":
    import os
 
    OUT_DIR = "reports"
    os.makedirs(OUT_DIR, exist_ok=True)
 
    def _build_and_save(label: str, start_dt: str, end_dt: str, filename: str):
        print(f"\n{'='*60}")
        print(f"  {label}: {start_dt} → {end_dt}")
        print(f"{'='*60}")
        df = build_partner_ops_vector(start_dt, end_dt)
        if df.empty:
            print(f"[{label}] Empty ops vector. Nothing to save.")
            return df
        path = os.path.join(OUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"\n[{label}] Saved {len(df)} partners × {len(df.columns)} cols → {path}")
        print(f"\n[{label}] COLUMN INVENTORY:")
        for c in sorted(df.columns):
            n_valid = df[c].notna().sum()
            print(f"  {c:>40s}: {n_valid}/{len(df)}")
        return df
 
    df_ops_train = _build_and_save(
        "TRAIN", config.TRAIN_START_DATE, config.TRAIN_END_DATE,
        "partner_ops_train_vector.csv",
    )
    df_ops_test = _build_and_save(
        "TEST", config.TEST_START_DATE, config.TEST_END_DATE,
        "partner_ops_test_vector.csv",
    )
    