"""
ForgeMind AI – Fleet Analytics Engine  (new in v2.0)

Computes fleet-level and per-asset KPIs used by the dashboard:

  OEE (Overall Equipment Effectiveness)
    = Availability × Performance × Quality  (0–100 %)

  MTBF  – Mean Time Between Failures (hours)
  MTTR  – Mean Time To Repair (hours, from maintenance history)

  Fleet Health Index  – fleet-wide weighted health average
  Fault Distribution  – count of each inferred fault type
  Cost Exposure       – aggregated estimated repair cost across fleet
  Efficiency Matrix   – per machine-type average OEE and health score
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# OEE computation
# ---------------------------------------------------------------------------

def compute_oee(row: pd.Series) -> float:
    """
    Estimate OEE (0–100) for a single machine from sensor-derived proxies.

    Availability proxy  : penalise for high failure_prob (downtime risk)
    Performance proxy   : rpm ratio vs. safe_max (operational speed ratio)
    Quality proxy       : health_score / 100

    Returns OEE as a percentage rounded to 1 dp.
    """
    # Availability: assume machine is available unless failure_prob is high
    fail_p       = float(row.get("failure_prob", 0)) / 100.0
    availability = max(0.0, 1.0 - fail_p * 0.8)    # linear availability loss

    # Performance: how close to rated RPM is the machine running?
    rpm          = float(row.get("rpm", 1000))
    # Use a reasonable rated RPM per type; fallback to 1800
    rated_rpms   = {
        "CNC Milling Machine":       2000,
        "Hydraulic Press":            800,
        "Industrial Compressor":     3000,
        "Conveyor Drive Motor":      1500,
        "Injection Moulding Unit":   1200,
        "Robotic Welding Arm":        700,
    }
    rated        = rated_rpms.get(str(row.get("machine_type", "")), 1800)
    performance  = min(rpm / rated, 1.0) if rated > 0 else 1.0
    # Slight penalty if running too fast (over-speed)
    if rpm > rated * 1.05:
        performance = max(0.0, performance - 0.05)

    # Quality: health-score proxy
    quality      = float(row.get("health_score", 100)) / 100.0

    oee          = availability * performance * quality * 100
    return round(min(max(oee, 0.0), 100.0), 1)


def add_oee(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["oee"] = df.apply(compute_oee, axis=1)
    return df


# ---------------------------------------------------------------------------
# MTBF / MTTR
# ---------------------------------------------------------------------------

def estimate_mtbf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate MTBF per machine from runtime hours and anomaly rate.

    MTBF ≈ runtime_hours / max(1, estimated_failure_events)
    Failure events estimated as: anomaly_count × failure_probability_mean
    """
    rows = []
    for mid, grp in df.groupby("machine_id"):
        runtime   = float(grp["runtime_hours"].max())
        n_anomaly = int(grp["is_anomaly"].sum())
        fp_mean   = float(grp["failure_prob"].mean()) / 100.0
        est_fails = max(1, round(n_anomaly * fp_mean))
        mtbf      = round(runtime / est_fails, 1)
        rows.append({"machine_id": mid, "mtbf_hours": mtbf})
    return pd.DataFrame(rows)


def estimate_mttr_from_history(history_df: pd.DataFrame) -> float:
    """
    Compute mean MTTR from a maintenance history DataFrame.
    Uses the 'Duration (hr)' column if available.
    """
    if "Duration (hr)" not in history_df.columns or history_df.empty:
        return 2.5    # fallback industry average
    return round(float(history_df["Duration (hr)"].mean()), 2)


# ---------------------------------------------------------------------------
# Fleet-level summary KPIs
# ---------------------------------------------------------------------------

def fleet_summary(df: pd.DataFrame) -> dict:
    """
    Return a dict of fleet-wide KPI values derived from the latest
    per-machine snapshot (df_latest).

    Keys returned:
      fleet_health_index    – weighted average health score (%)
      avg_oee               – mean OEE across fleet (%)
      total_cost_exposure   – sum of est_cost_usd for non-INFO alerts (USD)
      total_downtime_exposure – sum of est_downtime_hr for non-INFO (hr)
      critical_count        – machines with alert_priority == CRITICAL
      high_count            – CRITICAL or HIGH
      pct_anomalous         – % machines flagged as anomalous
      top_fault             – most common inferred_fault label
      machines_critical_rul – machines with rul_urgency == Critical
    """
    n = len(df)
    if n == 0:
        return {}

    summary: dict = {}

    summary["fleet_health_index"]       = round(float(df["health_score"].mean()), 1)
    summary["avg_oee"]                  = round(float(df.get("oee", pd.Series([0])).mean()), 1) if "oee" in df.columns else 0.0
    summary["total_cost_exposure"]      = int(df.loc[df["alert_priority"] != "INFO", "est_cost_usd"].sum()) if "est_cost_usd" in df.columns else 0
    summary["total_downtime_exposure"]  = round(float(df.loc[df["alert_priority"] != "INFO", "est_downtime_hr"].sum()), 1) if "est_downtime_hr" in df.columns else 0.0
    summary["critical_count"]           = int((df["alert_priority"] == "CRITICAL").sum())
    summary["high_count"]               = int(df["alert_priority"].isin(["CRITICAL", "HIGH"]).sum())
    summary["pct_anomalous"]            = round(float(df["is_anomaly"].mean()) * 100, 1)
    summary["machines_critical_rul"]    = int((df.get("rul_urgency", pd.Series([])) == "Critical").sum()) if "rul_urgency" in df.columns else 0

    if "inferred_fault" in df.columns:
        fault_counts = df[df["inferred_fault"] != "No Fault Detected"]["inferred_fault"].value_counts()
        summary["top_fault"] = fault_counts.index[0] if len(fault_counts) > 0 else "None"
    else:
        summary["top_fault"] = "N/A"

    return summary


# ---------------------------------------------------------------------------
# Efficiency matrix (per machine type)
# ---------------------------------------------------------------------------

def efficiency_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with per-machine-type aggregate statistics:
      machine_type, avg_oee, avg_health, avg_failure_prob,
      anomaly_rate (%), critical_machines
    """
    if df.empty:
        return pd.DataFrame()

    # Use named agg functions for pandas 2.x compatibility (avoids groupby.apply slowness)
    agg_cols: dict = {
        "health_score":  "mean",
        "failure_prob":  "mean",
        "is_anomaly":    "mean",
    }
    if "oee" in df.columns:
        agg_cols["oee"] = "mean"

    result = df.groupby("machine_type").agg(agg_cols).reset_index()

    # Rename
    rename_map = {
        "health_score": "avg_health",
        "failure_prob": "avg_failure_prob",
        "is_anomaly":   "anomaly_rate_pct",
    }
    if "oee" in result.columns:
        rename_map["oee"] = "avg_oee"
    result = result.rename(columns=rename_map)

    # Convert anomaly_rate to percentage
    result["anomaly_rate_pct"] = (result["anomaly_rate_pct"] * 100).round(1)
    result["avg_health"]       = result["avg_health"].round(1)
    result["avg_failure_prob"] = result["avg_failure_prob"].round(1)
    if "avg_oee" in result.columns:
        result["avg_oee"] = result["avg_oee"].round(1)

    # Critical machines count per type
    if "alert_priority" in df.columns:
        crit = (
            df[df["alert_priority"] == "CRITICAL"]
            .groupby("machine_type")
            .size()
            .reset_index(name="critical_machines")
        )
        result = result.merge(crit, on="machine_type", how="left")
        result["critical_machines"] = result["critical_machines"].fillna(0).astype(int)
    else:
        result["critical_machines"] = 0

    return result.sort_values("avg_health").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fault distribution summary
# ---------------------------------------------------------------------------

def fault_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count occurrences of each inferred fault type across the fleet.
    Returns a DataFrame with columns: fault_type, count, pct.
    """
    if "inferred_fault" not in df.columns or df.empty:
        return pd.DataFrame(columns=["fault_type", "count", "pct"])

    counts = (
        df["inferred_fault"]
        .value_counts()
        .reset_index()
    )
    counts.columns = ["fault_type", "count"]
    counts["pct"]  = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts
