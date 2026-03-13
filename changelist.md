# TEMPORAL BUCKETS — FULL CHANGELIST

---

## FILE 1: `data_lib/config.py`

### ADD after line `EARTH_RADIUS_METER = 6371000`:

```python
# ── TEMPORAL BUCKETS (days lookback) ──
TEMPORAL_WINDOWS = [30, 60, 365]
```

---

## FILE 2: `data_lib/geometry/hex.py`

### CHANGE `compute_hexes` signature:

```python
# OLD:
def compute_hexes(
    hexes, center_lat, center_lon, sources, bad_se, mid_se,
    best_size, METERS_PER_DEG_LAT, cos_lat, pid,
):

# NEW:
def compute_hexes(
    hexes, center_lat, center_lon, sources, bad_se, mid_se,
    best_size, METERS_PER_DEG_LAT, cos_lat, pid,
    reference_date=None,
):
```

### ADD inside `compute_hexes`, right after `hex_stats = []` and before the `for hex_poly in hexes:` loop:

```python
    # ── Pre-compute temporal cutoffs ──
    has_temporal = reference_date is not None and "decision_time" in sources.columns
    if has_temporal:
        ref_ts = pd.Timestamp(reference_date)
        cutoffs = {wd: ref_ts - pd.Timedelta(days=wd) for wd in config.TEMPORAL_WINDOWS}
```

### ADD inside the `for hex_poly in hexes:` loop, right after `hex_stats.append({...})` — REPLACE that append block:

```python
        # OLD:
        hex_stats.append(
            {
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
        )

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
                row["installs_30d"] / max(row["installs_365d"], 1), 4
            )
        else:
            for wd in config.TEMPORAL_WINDOWS:
                row[f"se_{wd}d"] = np.nan
                row[f"installs_{wd}d"] = np.nan
                row[f"declines_{wd}d"] = np.nan
                row[f"total_{wd}d"] = np.nan
            row["install_velocity"] = np.nan

        hex_stats.append(row)
```

### ADD to `import` block at top:

```python
import numpy as np
```

---

## FILE 3: `data_lib/geometry/find_boundary.py`

### ADD inside the `for (partner_id, cluster_id), group_df in ...` loop, right after the existing `boundary_summary.append({...})` — ADD temporal keys to that dict:

```python
        # EXISTING dict already has: "total_installs": group_df["installs"].sum(), etc.
        # ADD these keys to the same dict:

        # ── Temporal cluster stats (summed from hex-level) ──
        for wd in config.TEMPORAL_WINDOWS:
            inst_col = f"installs_{wd}d"
            tot_col = f"total_{wd}d"
            if inst_col in group_df.columns:
                t_inst = group_df[inst_col].sum()
                t_tot = group_df[tot_col].sum()
                boundary_summary[-1][f"total_installs_{wd}d"] = int(t_inst)
                boundary_summary[-1][f"total_obs_{wd}d"] = int(t_tot)
                boundary_summary[-1][f"cluster_se_{wd}d"] = round(t_inst / t_tot, 4) if t_tot > 0 else np.nan
            else:
                boundary_summary[-1][f"total_installs_{wd}d"] = np.nan
                boundary_summary[-1][f"total_obs_{wd}d"] = np.nan
                boundary_summary[-1][f"cluster_se_{wd}d"] = np.nan
```

### ADD to imports at top:

```python
import data_lib.config as config
```

(already has `from data_lib import config` style — use whichever is consistent)

---

## FILE 4: `data_lib/test.py`

### CHANGE in `get_overlap()` — the `final_columns` list. REPLACE:

```python
    # OLD:
    final_columns = [
        "partner_id", "poly_id", "best_size", "poly", "se",
        "installs", "declines", "total", "color",
        "is_overlap", "distance_from_boundary_m",
        "distance_to_own_boundary_m", "rank",
    ]

    # NEW:
    temporal_cols = []
    for wd in [30, 60, 365]:
        temporal_cols += [f"se_{wd}d", f"installs_{wd}d", f"declines_{wd}d", f"total_{wd}d"]
    temporal_cols.append("install_velocity")

    final_columns = [
        "partner_id", "poly_id", "best_size", "poly", "se",
        "installs", "declines", "total", "color",
        "is_overlap", "distance_from_boundary_m",
        "distance_to_own_boundary_m", "rank",
    ] + temporal_cols

    df_final = df_final[[c for c in final_columns if c in df_final.columns]]
```

---

## FILE 5: `data_lib/params.py`

### ADD temporal columns to `one_to_many_cols` — no change needed, these flow via boundary merge automatically.

### ADD to `hex_cols`:

```python
# ADD at end of hex_cols list:
] + [f"se_{wd}d" for wd in [30, 60, 365]] + \
    [f"installs_{wd}d" for wd in [30, 60, 365]] + \
    [f"total_{wd}d" for wd in [30, 60, 365]] + \
    ["install_velocity"]
```

---

## FILE 6: `data_lib/feature/hop_features.py`

### CHANGE inside `compute_hop_features()` — after `se_arr = grp["se"].values`, ADD:

```python
        # ── Temporal SE arrays ──
        temporal_se = {}
        for wd in [30, 60, 365]:
            col = f"se_{wd}d"
            if col in grp.columns:
                temporal_se[wd] = grp[col].values
```

### CHANGE inside the inner `for h in range(1, n_hops + 1):` loop, right after the existing `row[f"hop{h}_se_wmean"] = round(wmean, 6)` block — ADD:

```python
                # ── Temporal hop SE ──
                for wd, se_t_arr in temporal_se.items():
                    ring_se_t = se_t_arr[ring_idxs]
                    valid_t = ~np.isnan(ring_se_t)
                    if valid_t.any():
                        wmean_t = np.average(ring_se_t[valid_t], weights=weights[valid_t])
                        row[f"hop{h}_se_{wd}d_wmean"] = round(wmean_t, 6)
                    else:
                        row[f"hop{h}_se_{wd}d_wmean"] = np.nan
```

### CHANGE the cross-hop interactions block — after existing `se_gradient_1to3`, `se_confirmed`, ADD:

```python
            # ── Temporal cross-hop interactions ──
            for wd in temporal_se.keys():
                h1_t = row.get(f"hop1_se_{wd}d_wmean", np.nan)
                h3_t = row.get(f"hop3_se_{wd}d_wmean", np.nan)
                if not np.isnan(h1_t) and not np.isnan(h3_t):
                    row[f"se_gradient_1to3_{wd}d"] = round(h1_t - h3_t, 6)
                    row[f"se_confirmed_{wd}d"] = round(h1_t * h3_t, 6)
                else:
                    row[f"se_gradient_1to3_{wd}d"] = np.nan
                    row[f"se_confirmed_{wd}d"] = np.nan
```

### FIX the solo-hex fallback row (the `if len(grp) < 2:` block) — ADD temporal NaN keys:

```python
            # ADD after existing hop NaN keys:
            for wd in [30, 60, 365]:
                for h in range(1, n_hops + 1):
                    row[f"hop{h}_se_{wd}d_wmean"] = np.nan
                row[f"se_gradient_1to3_{wd}d"] = np.nan
                row[f"se_confirmed_{wd}d"] = np.nan
```

---

## FILE 7: `data_lib/compute.py`

### CHANGE 7a: `compute_hex_consensus_features()` — after existing shrinkage block (A.3), ADD:

```python
    # ── A.3b: Temporal SE shrinkage ──
    for wd in config.TEMPORAL_WINDOWS:
        se_col = f"se_{wd}d"
        tot_col = f"total_{wd}d"
        if se_col in joined.columns:
            ratio_t = np.maximum(
                config.MIN_SHRINKAGE_RATIO,
                joined[tot_col].fillna(0) / (joined[tot_col].fillna(0) + config.SHRINKAGE_K),
            )
            joined[f"se_{wd}d_shrunk"] = np.where(
                joined[se_col].fillna(0) >= 0,
                joined[se_col].fillna(0) * ratio_t,
                joined[se_col].fillna(0) / ratio_t,
            )
            joined[f"_w_se_{wd}d_shrunk"] = joined[f"se_{wd}d_shrunk"] * joined[tot_col].fillna(0)
```

### CHANGE 7b: in the `joined.groupby().agg(...)` call — ADD these entries to the agg dict:

```python
            # ── Temporal aggregations ──
            **{f"installs_{wd}d": (f"installs_{wd}d", "sum") for wd in config.TEMPORAL_WINDOWS if f"installs_{wd}d" in joined.columns},
            **{f"declines_{wd}d": (f"declines_{wd}d", "sum") for wd in config.TEMPORAL_WINDOWS if f"declines_{wd}d" in joined.columns},
            **{f"total_{wd}d": (f"total_{wd}d", "sum") for wd in config.TEMPORAL_WINDOWS if f"total_{wd}d" in joined.columns},
            **{f"_w_se_{wd}d_shrunk": (f"_w_se_{wd}d_shrunk", "sum") for wd in config.TEMPORAL_WINDOWS if f"_w_se_{wd}d_shrunk" in joined.columns},
            # Temporal hop wmeans (total-weighted)
            **{f"_w_hop{h}_se_{wd}d": (f"_w_hop{h}_se_{wd}d_wmean", "sum")
               for wd in config.TEMPORAL_WINDOWS for h in [1, 2, 3]
               if f"_w_hop{h}_se_{wd}d_wmean" in joined.columns},
```

BUT FIRST — before the groupby, create the temporal hop weighted intermediates:

```python
    # ── A.4b: Temporal hop weighted intermediates ──
    for wd in config.TEMPORAL_WINDOWS:
        for h in [1, 2, 3]:
            col = f"hop{h}_se_{wd}d_wmean"
            if col in joined.columns:
                joined[f"_w_hop{h}_se_{wd}d_wmean"] = joined[col] * joined["total"]
        gcol = f"se_gradient_1to3_{wd}d"
        ccol = f"se_confirmed_{wd}d"
        if gcol in joined.columns:
            joined[f"_w_gradient_{wd}d"] = joined[gcol] * joined["total"]
        if ccol in joined.columns:
            joined[f"_w_confirmed_{wd}d"] = joined[ccol] * joined["total"]
```

And add these to the agg dict too:

```python
            **{f"_w_gradient_{wd}d": (f"_w_gradient_{wd}d", "sum") for wd in config.TEMPORAL_WINDOWS if f"_w_gradient_{wd}d" in joined.columns},
            **{f"_w_confirmed_{wd}d": (f"_w_confirmed_{wd}d", "sum") for wd in config.TEMPORAL_WINDOWS if f"_w_confirmed_{wd}d" in joined.columns},
```

### CHANGE 7c: after the groupby, derive temporal consensus columns — ADD after existing hop derivations:

```python
    # ── Temporal consensus columns ──
    for wd in config.TEMPORAL_WINDOWS:
        tot_col = f"total_{wd}d"
        if tot_col in consensus.columns:
            safe_t = consensus[tot_col].replace(0, np.nan)
            consensus[f"weighted_se_{wd}d"] = np.where(
                consensus[tot_col] > 0,
                consensus[f"installs_{wd}d"] / consensus[tot_col],
                np.nan,
            )
            consensus[f"weighted_se_{wd}d_shrunk"] = consensus.get(
                f"_w_se_{wd}d_shrunk", np.nan
            )
            if f"_w_se_{wd}d_shrunk" in consensus.columns:
                consensus[f"weighted_se_{wd}d_shrunk"] = (
                    consensus[f"_w_se_{wd}d_shrunk"] / safe_t
                )
                consensus.drop(columns=[f"_w_se_{wd}d_shrunk"], inplace=True)

        # Temporal hop consensus
        for h in [1, 2, 3]:
            wcol = f"_w_hop{h}_se_{wd}d"
            if wcol in consensus.columns:
                consensus[f"hop{h}_se_{wd}d_wmean"] = consensus[wcol] / total_safe
                consensus.drop(columns=[wcol], inplace=True)

        for suffix in ["gradient", "confirmed"]:
            wcol = f"_w_{suffix}_{wd}d"
            if wcol in consensus.columns:
                consensus[f"se_{suffix}_1to3_{wd}d"] = consensus[wcol] / total_safe
                consensus.drop(columns=[wcol], inplace=True)

    # SE momentum: 30d shrunk - 365d shrunk
    if "weighted_se_30d_shrunk" in consensus.columns and "weighted_se_365d_shrunk" in consensus.columns:
        consensus["se_momentum"] = consensus["weighted_se_30d_shrunk"] - consensus["weighted_se_365d_shrunk"]
```

### CHANGE 7d: `compute_contested_metrics_engine()` — inside the per-mobile loop, after the existing `base.update({...})` block that sets `contested_se`, ADD:

```python
        # ── Temporal contested stats ──
        if n_total > 0 and "decision_time" in inside.columns:
            for wd in config.TEMPORAL_WINDOWS:
                cutoff = t0 - pd.Timedelta(days=wd)
                w_inside = inside[inside["decision_time"] >= cutoff]
                w_n = len(w_inside)
                w_inst = int(w_inside["installed_decision"].sum()) if w_n > 0 else 0
                w_se = round(w_inst / w_n, 4) if w_n > 0 else np.nan
                base[f"contested_installs_{wd}d"] = w_inst
                base[f"contested_total_{wd}d"] = w_n
                base[f"contested_se_{wd}d"] = w_se
        else:
            for wd in config.TEMPORAL_WINDOWS:
                base[f"contested_installs_{wd}d"] = np.nan
                base[f"contested_total_{wd}d"] = np.nan
                base[f"contested_se_{wd}d"] = np.nan
```

---

## FILE 8: `data_lib/data_fetch/get_ops_data.py`

### ADD new multi-window wrapper functions. Place right after existing `get_partner_performance()`:

```python
def get_partner_performance_multi(windows=None, end_dt=None) -> pd.DataFrame:
    """Calls get_partner_performance for each window, merges with suffixed columns."""
    if windows is None:
        windows = [30, 60, 365]
    from functools import reduce
    frames = []
    for w in windows:
        df_w = get_partner_performance(lookback_days=w, end_dt=end_dt)
        if df_w.empty:
            continue
        rename = {c: f"{c}_{w}d" for c in df_w.columns if c != "partner_id"}
        df_w = df_w.rename(columns=rename)
        frames.append(df_w)
    if not frames:
        return pd.DataFrame()
    return reduce(lambda l, r: l.merge(r, on="partner_id", how="outer"), frames)
```

### ADD right after existing `get_expected_daily_slots()`:

```python
def get_expected_daily_slots_multi(windows=None, end_dt=None) -> pd.DataFrame:
    """Calls get_expected_daily_slots for each window, merges with suffixed columns."""
    if windows is None:
        windows = [30, 60, 365]
    from functools import reduce
    frames = []
    for w in windows:
        df_w = get_expected_daily_slots(lookback_days=w, end_dt=end_dt)
        if df_w.empty:
            continue
        rename = {c: f"{c}_{w}d" for c in df_w.columns if c != "partner_id"}
        df_w = df_w.rename(columns=rename)
        frames.append(df_w)
    if not frames:
        return pd.DataFrame()
    return reduce(lambda l, r: l.merge(r, on="partner_id", how="outer"), frames)
```

### ADD right after existing `compute_reliability()`:

```python
def compute_reliability_multi(df_slots_raw: pd.DataFrame, windows=None) -> pd.DataFrame:
    """Computes reliability for each temporal window from slot confirmation data."""
    if windows is None:
        windows = [30, 60, 365]
    if df_slots_raw.empty:
        return pd.DataFrame()
    from functools import reduce
    frames = []
    for w in windows:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=w)
        df_w = df_slots_raw[df_slots_raw["slot_confirmed_time"] >= cutoff].copy()
        if df_w.empty:
            continue
        rel_w = compute_reliability(df_w)
        if rel_w.empty:
            continue
        rename = {c: f"{c}_{w}d" for c in rel_w.columns if c != "partner_id"}
        rel_w = rel_w.rename(columns=rename)
        frames.append(rel_w)
    if not frames:
        return pd.DataFrame()
    return reduce(lambda l, r: l.merge(r, on="partner_id", how="outer"), frames)
```

### ADD right after existing `compute_ticket_capacity()`:

```python
def compute_ticket_capacity_multi(df_tickets_raw: pd.DataFrame, windows=None) -> pd.DataFrame:
    """Computes ticket capacity for each temporal window."""
    if windows is None:
        windows = [30, 60, 365]
    if df_tickets_raw.empty:
        return pd.DataFrame()
    from functools import reduce
    frames = []
    for w in windows:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=w)
        df_w = df_tickets_raw[df_tickets_raw["assign_time"] >= cutoff].copy()
        if df_w.empty:
            continue
        tc_w = compute_ticket_capacity(df_w)
        if tc_w.empty:
            continue
        rename = {c: f"{c}_{w}d" for c in tc_w.columns if c != "partner_id"}
        tc_w = tc_w.rename(columns=rename)
        frames.append(tc_w)
    if not frames:
        return pd.DataFrame()
    return reduce(lambda l, r: l.merge(r, on="partner_id", how="outer"), frames)
```

### CHANGE `build_partner_ops_vector()` — ADD multi-window calls after the existing single-window calls. Find these sections and ADD after each:

After `df_perf = get_partner_performance(...)`:
```python
    print("[OPS VECTOR] Pulling multi-window partner performance...")
    df_perf_multi = get_partner_performance_multi(
        windows=config.TEMPORAL_WINDOWS, end_dt=end_dt
    )
```

After `df_slots = get_expected_daily_slots(...)`:
```python
    print("[OPS VECTOR] Pulling multi-window expected daily slots...")
    df_slots_multi = get_expected_daily_slots_multi(
        windows=config.TEMPORAL_WINDOWS, end_dt=end_dt
    )
```

After `df_reliability = compute_reliability(df_slots_raw)`:
```python
    print("[OPS VECTOR] Computing multi-window reliability...")
    df_reliability_multi = compute_reliability_multi(
        df_slots_raw, windows=config.TEMPORAL_WINDOWS
    )
```

After `df_ticket_cap = compute_ticket_capacity(df_tickets_raw)`:
```python
    print("[OPS VECTOR] Computing multi-window ticket capacity...")
    df_ticket_cap_multi = compute_ticket_capacity_multi(
        df_tickets_raw, windows=config.TEMPORAL_WINDOWS
    )
```

Then in the merge section (after `ops = df_perf.copy()`), ADD:

```python
    if not df_perf_multi.empty:
        ops = ops.merge(df_perf_multi, on="partner_id", how="left")

    if not df_slots_multi.empty:
        ops = ops.merge(df_slots_multi, on="partner_id", how="left")

    if not df_reliability_multi.empty:
        ops = ops.merge(df_reliability_multi, on="partner_id", how="left")

    if not df_ticket_cap_multi.empty:
        ops = ops.merge(df_ticket_cap_multi, on="partner_id", how="left")

    # ── Derived trend features ──
    if "se_30d_30d" in ops.columns and "se_30d_365d" in ops.columns:
        ops["partner_se_trend"] = ops["se_30d_30d"] - ops["se_30d_365d"]
    if "decline_rate_30d_30d" in ops.columns and "decline_rate_30d_365d" in ops.columns:
        ops["partner_decline_trend"] = ops["decline_rate_30d_30d"] - ops["decline_rate_30d_365d"]
    if "expected_daily_slots_30d" in ops.columns and "expected_daily_slots_365d" in ops.columns:
        ops["slot_trend"] = (
            ops["expected_daily_slots_30d"]
            / ops["expected_daily_slots_365d"].replace(0, np.nan)
        ).fillna(1.0)
    if "plan_created_rate_30d" in ops.columns and "plan_created_rate_365d" in ops.columns:
        ops["reliability_trend"] = ops["plan_created_rate_30d"] - ops["plan_created_rate_365d"]
```

### ADD `import data_lib.config as config` to imports at top (if not already present).

---

## FILE 9: `data_lib/feature/ops_features.py`

### CHANGE `compute_capacity_score()` — ADD trend component after existing parts:

```python
    # ── Capacity trend (30d vs 365d queue velocity) ──
    if "queue_velocity_30d" in df_ops.columns and "queue_velocity_365d" in df_ops.columns:
        trend = (df_ops["queue_velocity_30d"].fillna(0) - df_ops["queue_velocity_365d"].fillna(0) + 0.5).clip(0, 1)
        parts.append(trend)
```

### CHANGE `compute_reliability_score()` — ADD trend component after existing parts:

```python
    # ── Reliability trend (30d vs 365d plan rate) ──
    if "plan_created_rate_30d" in df_ops.columns and "plan_created_rate_365d" in df_ops.columns:
        trend = (df_ops["plan_created_rate_30d"].fillna(0) - df_ops["plan_created_rate_365d"].fillna(0) + 0.5).clip(0, 1)
        parts.append(trend)
```

---

## FILE 10: `data_lib/feature/composite.py`

### CHANGE `compute_spatial_components()` — ADD after existing `df["norm_color"]` block:

```python
    # ── Temporal SE signals ──
    for wd in config.TEMPORAL_WINDOWS:
        col = f"weighted_se_{wd}d_shrunk"
        if col in df.columns:
            df[f"norm_se_{wd}d"] = _safe_normalize(df[col], method="rank")

    # ── SE momentum: recent vs long-term ──
    if "se_momentum" in df.columns:
        df["norm_momentum"] = _safe_normalize(df["se_momentum"], method="rank")

    # ── Contested temporal ──
    for wd in config.TEMPORAL_WINDOWS:
        col = f"contested_se_{wd}d"
        if col in df.columns:
            df[f"norm_contested_{wd}d"] = _safe_normalize(df[col], method="rank")
```

### ADD `import data_lib.config as config` to imports (if not already present — it is).

---

## FILE 11: `steps/step1_train_maps.py`

### CHANGE `process_single_partner()` — pass `reference_date` to `compute_hexes`:

```python
    # OLD:
    hex_stats = compute_hexes(
        hexes, center_lat, center_lon, sub_df, bad_se, mid_se,
        best_size, config.METERS_PER_DEG_LAT, cos_lat, partner_id,
    )

    # NEW:
    hex_stats = compute_hexes(
        hexes, center_lat, center_lon, sub_df, bad_se, mid_se,
        best_size, config.METERS_PER_DEG_LAT, cos_lat, partner_id,
        reference_date=config.TRAIN_END_DATE,
    )
```

### CHANGE the `df_hex = pd.DataFrame(hexagons, columns=[...])` call — REMOVE explicit columns list so dicts define schema:

```python
    # OLD:
    df_hex = pd.DataFrame(
        hexagons,
        columns=[
            "partner_id", "best_size", "poly_id", "poly",
            "se", "installs", "declines", "total", "color",
        ],
    )

    # NEW:
    df_hex = pd.DataFrame(hexagons)
```

### ENSURE `decision_time` column exists in `sub_df` passed to `compute_hexes`. It already does — `df_train` has `decision_time` from `process_dataframe()`.

---

## FILE 12: `steps/step2_score_test.py`

### No structural changes needed — temporal columns flow through automatically from `process()` → `compute_hex_consensus_features()` → `scored_df`.

### OPTIONAL — ADD temporal columns to the evaluation section for visibility. After existing `results.append(evaluate_bucket(...))` blocks, ADD:

```python
    # ── Temporal SE separation check ──
    for wd in config.TEMPORAL_WINDOWS:
        col = f"weighted_se_{wd}d"
        if col in df.columns:
            valid = df[col].notna()
            if valid.any():
                p10 = df.loc[valid, col].quantile(0.1)
                p90 = df.loc[valid, col].quantile(0.9)
                print(f"  weighted_se_{wd}d — p10={p10:.4f}  p90={p90:.4f}  gap={p90-p10:.4f}")
```

### ADD temporal to the saved report — already happens automatically since `df` carries all columns through to `df.to_csv(report_path)`.

---

## FILE 13: `steps/step3_simpulate.py`

### No changes needed — temporal columns flow through `compute_process()` automatically.

---

## FILE 14: `data_lib/geometry/geometric_features.py`

### CHANGE `batch_compute_geometry()` signature — ADD `reference_date` param:

```python
# OLD:
def batch_compute_geometry(df_leads, df_history, radius_m=250):

# NEW:
def batch_compute_geometry(df_leads, df_history, radius_m=250, reference_date=None):
```

### ADD after the existing `results` loop, temporal geometry computation:

```python
    # ── Temporal geometry ──
    if reference_date is not None and "decision_time" in df_history.columns:
        ref_ts = pd.Timestamp(reference_date)
        for wd in config.TEMPORAL_WINDOWS:
            cutoff = ref_ts - pd.Timedelta(days=wd)
            hist_w = df_history[df_history["decision_time"] >= cutoff]
            if len(hist_w) < 3:
                for i in range(len(results)):
                    for k in ["local_anisotropy", "local_density", "hull_area", "spread_m"]:
                        results[i][f"{k}_{wd}d"] = np.nan
                continue

            hist_w_rad = np.radians(hist_w[["latitude", "longitude"]].values)
            tree_w = BallTree(hist_w_rad, metric="haversine")
            idx_w = tree_w.query_radius(leads_rad, r=radius_rad)

            for i, indices in enumerate(idx_w):
                if len(indices) < 3:
                    for k in ["local_anisotropy", "local_density", "hull_area", "spread_m"]:
                        results[i][f"{k}_{wd}d"] = np.nan
                    continue
                neighbors = hist_w.iloc[indices]
                m_w = compute_local_geometry(
                    df_leads.iloc[i]["latitude"],
                    df_leads.iloc[i]["longitude"],
                    neighbors, radius_m,
                )
                for k in ["local_anisotropy", "local_density", "hull_area", "spread_m"]:
                    results[i][f"{k}_{wd}d"] = m_w.get(k, np.nan)
```

### ADD import at top:

```python
import data_lib.config as config
```

### CHANGE call in `step2_score_test.py`:

```python
    # OLD:
    geom_features = batch_compute_geometry(
        df_test, df_train_geom, radius_m=int(config.GEOM_SEARCH_RADIUS_M)
    )

    # NEW:
    geom_features = batch_compute_geometry(
        df_test, df_train_geom, radius_m=int(config.GEOM_SEARCH_RADIUS_M),
        reference_date=config.TEST_START_DATE,
    )
```

---

## SUMMARY: NEW FEATURES PRODUCED

### Hex level (poly_stats_final.h5):
- `se_30d`, `se_60d`, `se_365d`
- `installs_30d/60d/365d`, `declines_30d/60d/365d`, `total_30d/60d/365d`
- `install_velocity`

### Boundary/cluster level:
- `total_installs_30d/60d/365d`, `total_obs_30d/60d/365d`, `cluster_se_30d/60d/365d`

### Hop features:
- `hop{1,2,3}_se_{30,60,365}d_wmean`
- `se_gradient_1to3_{30,60,365}d`, `se_confirmed_{30,60,365}d`

### Hex consensus (mobile level):
- `weighted_se_30d/60d/365d`, `weighted_se_30d/60d/365d_shrunk`
- `installs_30d/60d/365d`, `total_30d/60d/365d`
- `se_momentum` (30d - 365d shrunk)
- Temporal hop consensus

### Contested:
- `contested_se_30d/60d/365d`, `contested_installs_30d/60d/365d`, `contested_total_30d/60d/365d`

### Partner ops vector:
- `se_30d_{30,60,365}d`, `decline_rate_30d_{30,60,365}d`
- `median_response_min_{30,60,365}d`, `total_decisions_{30,60,365}d`
- `expected_daily_slots_{30,60,365}d`, `active_days_{30,60,365}d`
- `plan_created_rate_{30,60,365}d`, `late_arrive_median_{30,60,365}d`
- `active_tickets_{30,60,365}d`, `median_tat_min_{30,60,365}d`
- `partner_se_trend`, `partner_decline_trend`, `slot_trend`, `reliability_trend`

### Geometry:
- `local_anisotropy_{30,60,365}d`, `local_density_{30,60,365}d`
- `hull_area_{30,60,365}d`, `spread_m_{30,60,365}d`

### Composite:
- `norm_se_30d/60d/365d`, `norm_momentum`
- `norm_contested_30d/60d/365d`