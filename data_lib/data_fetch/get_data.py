from typing import Optional

import pandas as pd
import numpy as np

from data_lib.data_fetch.wiom_data import WiomData


_snowflake_client: Optional[WiomData] = None


def _get_snowflake_client() -> WiomData:
    global _snowflake_client
    if _snowflake_client is None:
        _snowflake_client = WiomData("snowflake")
    return _snowflake_client


def _query_snowflake_df(
    sql: str, *, cache_file: Optional[str] = None, cache_h: int = 1
) -> pd.DataFrame:
    client = _get_snowflake_client()
    if cache_file is None:
        return client.query(sql, cache_h=cache_h)
    return client.query(sql, cache_file=cache_file, cache_h=cache_h)


def standardise_decisions(df):
    # Safety check for empty df
    if df.empty:
        return df

    # Calculate time to decide in minutes
    df["time_to_decide"] = (
        df["last_event_time"] - df["first_event_time"]
    ).dt.total_seconds() / 60

    # Calculate quantile thresholds on the specific slice of Interest->Decline
    mask_interest_decline = (df["first_event"] == "INTERESTED") & (
        df["last_event"] == "DECLINED"
    )

    # If no data for quantiles, return original (or handle gracefully)
    if not mask_interest_decline.any():
        print(
            "Warning: No INTERESTED->DECLINED events found for standardization thresholds."
        )
        return df

    dec_quantiles = (
        df[mask_interest_decline]["time_to_decide"]
        .quantile([i / 10 for i in range(0, 11)])
        .round(2)
        .reset_index()
    )

    # p40 (40th percentile in the list; quantile=0.4) used as cutoff
    try:
        p_40 = dec_quantiles.loc[dec_quantiles["index"] == 0.4, "time_to_decide"].iloc[
            0
        ]
    except IndexError:
        p_40 = 0  # Fallback

    print(f"Standardization Cutoff (p40): {p_40} minutes")

    # Apply Logic
    # 1. INSTALLED remains INSTALLED
    # 2. Fast Decline (<= p40) -> DECLINED
    # 3. Slow Decline (> p40) -> INDETERMINATE
    # 4. HANGING (Interested/Called) -> HANGING

    df["final_decision"] = np.where(
        df["final_decision"] == "INSTALLED",
        df["final_decision"],
        np.where(
            mask_interest_decline & (df["time_to_decide"] <= p_40),
            "DECLINED",
            np.where(
                mask_interest_decline & (df["time_to_decide"] > p_40),
                "INDETERMINATE",
                np.where(
                    df["final_decision"].isin(["INTERESTED", "CALLED"]),
                    "HANGING",
                    df["final_decision"],
                ),
            ),
        ),
    )

    # Filter to only valid decision types (Removing RNR, etc from Training Data)
    valid_types = ["DECLINED", "INSTALLED", "INDETERMINATE", "HANGING"]
    df_clean = df[df["final_decision"].isin(valid_types)].copy()

    print("STANDARDISED DECISIONS (TRAIN):")
    print(df_clean["final_decision"].value_counts())

    return df_clean


def get_g1_distance(start_dt: str, end_dt: str) -> pd.DataFrame:
    """
    Pulls latest G1 serviceability log per mobile in the given date range.
    Returns columns: mobile, decision_time, min_dist, dist_rng, latitude, longitude,
    g1_is_bdo_lead, zone_alias, serviceable (as per log), plus typed fields.
    """
    print(f"GETTING G1 DECLINES FROM {start_dt} TO {end_dt}")

    query = f"""
        select mobile, created_at as decision_time, min_dist, CASE
        WHEN min_dist <= 5 THEN 'A. 0-5m'
        WHEN min_dist <= 10 THEN 'B. 5-10m'
        WHEN min_dist <= 15 THEN 'C. 10-15m'
        WHEN min_dist <= 20 THEN 'D. 15-20m'
        WHEN min_dist <= 25 THEN 'E. 20-25m'
        WHEN min_dist <= 30 THEN 'F. 25-30m'
        WHEN min_dist <= 35 THEN 'G. 30-35m'
        WHEN min_dist <= 40 THEN 'H. 35-40m'
        WHEN min_dist <= 45 THEN 'I. 40-45m'
        WHEN min_dist <= 50 THEN 'J. 45-50m'
        WHEN min_dist <= 55 THEN 'K. 50-55m'
        WHEN min_dist <= 60 THEN 'L. 55-60m'
        WHEN min_dist <= 65 THEN 'M. 60-65m'
        WHEN min_dist <= 70 THEN 'N. 65-70m'
        WHEN min_dist <= 75 THEN 'O. 70-75m'
        WHEN min_dist <= 80 THEN 'P. 75-80m'
        WHEN min_dist <= 85 THEN 'Q. 80-85m'
        WHEN min_dist <= 90 THEN 'R. 85-90m'
        WHEN min_dist <= 95 THEN 'S. 90-95m'
        WHEN min_dist <= 100 THEN 'T. 95-100m'
        ELSE 'U. >100m'
        END AS dist_rng, latitude, longitude, g1_is_bdo_lead, zone_alias, serviceable

        from
        (
            select *
            from
            (
                select *, row_number() over(partition by mobile order by created_at desc) as row_cnt
                from
                (
                    select *, parse_json(response):extra_data.nearest_coordinate.distance as min_dist,
                    PARSE_JSON(response):"serviceable"::boolean AS serviceable,
                    PARSE_JSON(response):"extra_data":"nearest_coordinate":"zone_alias"::string AS zone_alias,
                    PARSE_JSON(response):"genie_downstream_context":"g1_is_bdo_lead"::boolean AS g1_is_bdo_lead,
                    SPLIT(PARSE_JSON(response):genie_downstream_context.g1_lead_coords::string,',')[0]::float AS latitude,
                    SPLIT(PARSE_JSON(response):genie_downstream_context.g1_lead_coords::string,',')[1]::float AS longitude

                    from
                    (
                        select *
                        from prod_db.mysql_rds_genie_genie1.t_serviceability_logs where cast(created_at as date) between '{start_dt}' and '{end_dt}'
                    ) a
                ) a
            ) a
            where row_cnt = 1
        ) a
    """
    try:
        df = _query_snowflake_df(query)
        if df.empty:
            print("NO DATA FOUND OF G1 DECLINES")
            return pd.DataFrame()
        df.columns = df.columns.str.lower()
        df["decision_time"] = pd.to_datetime(df["decision_time"], errors="coerce")
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["mobile"] = df["mobile"].astype(str)
        df["min_dist"] = pd.to_numeric(df["min_dist"], errors="coerce")
        df["g1_is_bdo_lead"] = df["g1_is_bdo_lead"].astype(bool)
        return df
    except Exception as e:
        print(f"EXCEPTION IN PULLING G1 DECLINES: {e}")
        return pd.DataFrame()


def process_dataframe(df):
    if df.empty:
        return df

    df.columns = df.columns.str.lower()
    for col in [
        "first_notified_time",
        "installed_time",
        "first_event_time",
        "last_event_time",
    ]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Logic from prod_main.py
    df["decision_time"] = np.where(
        df["installed_decision"] == 1, df["installed_time"], df["first_event_time"]
    )

    for col in ["partner_id", "mobile"]:
        df[col] = df[col].astype(str)
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_train_data(start_dt: str, end_dt: str):
    """
    Fetches training data between inclusive dates.
    Used to build the maps (Hexes, Boundaries, Spatial Weights).

    Args:
        start_dt: 'YYYY-MM-DD'
        end_dt:   'YYYY-MM-DD'
    """
    print(f"Fetching TRAINING data ({start_dt} to {end_dt})...")

    # Snowflake query using dateadd with current_date()
    query = f"""
        select partner_id, mobile, first_notified_time, 
               case when lco_account_installed = partner_id then 1 else 0 end as installed_decision, 
               latitude, longitude, installed_time,
               case when lco_account_installed = partner_id then 'INSTALLED' else first_event end as final_decision, 
               active_base, partner_tenure, first_event_time, last_event, last_event_time, first_event
        from t_node_decisions_active 
        where first_event not in ('','None') 
        and cast(first_notified_time as date) between '{start_dt}' and '{end_dt}'
    """

    try:
        df = _query_snowflake_df(query)
        if df.empty:
            print("NO TRAINING DATA FOUND")
            return pd.DataFrame()

        df = process_dataframe(df)
        df = standardise_decisions(df)
        return df

    except Exception as e:
        print(f"ERROR FETCHING TRAIN DATA: {e}")
        return pd.DataFrame()


def get_test_data(start_dt: str, end_dt: str):
    """
    Fetches test data between inclusive dates.
    Used to simulate new leads and score them against the maps.

    Args:
        start_dt: 'YYYY-MM-DD'
        end_dt:   'YYYY-MM-DD'
    """
    print(f"Fetching TEST data ({start_dt} to {end_dt})...")

    query = f"""
        select partner_id, mobile, first_notified_time,
               case when lco_account_installed = partner_id then 1 else 0 end as installed_decision,
               latitude, longitude, installed_time,
               case when lco_account_installed = partner_id then 'INSTALLED' else first_event end as final_decision,
               active_base, partner_tenure, first_event_time, last_event, last_event_time, first_event
        from t_node_decisions_active
        where first_event not in ('','None')
        and cast(first_notified_time as date) between '{start_dt}' and '{end_dt}'
    """

    try:
        df = _query_snowflake_df(query)
        if df.empty:
            print("NO TEST DATA FOUND")
            return pd.DataFrame()

        df = process_dataframe(df)
        df = standardise_decisions(df)
        return df

    except Exception as e:
        print(f"ERROR FETCHING TEST DATA: {e}")
        return pd.DataFrame()
