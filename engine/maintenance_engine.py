"""
ForgeMind AI – Maintenance Recommendation Engine  v2.0

Improvements over v1:
  - Fault-type inference: maps anomaly_category + sensor deviations to a
    probable root-cause fault label
  - Cost & downtime estimates: each recommendation carries estimated repair
    cost (USD) and downtime (hours) so planners can prioritise
  - Severity trend column: STABLE / WORSENING / CRITICAL_ESCALATION derived
    from health_score trajectory
  - Covers all 7 sensors (humidity, rpm, oil_temp added to rule set)
  - Richer maintenance history: 12 action templates, realistic technician
    notes, parts replaced, and cost entries
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Extended rule set
# (condition_fn, priority, recommendation_text, est_cost_usd, est_downtime_hr)
# ---------------------------------------------------------------------------
RULES: list[tuple] = [
    # ── CRITICAL ────────────────────────────────────────────────────────────
    (
        lambda r: r["temperature"] > 115,
        "CRITICAL",
        "🔴 Immediate shutdown — thermal runaway risk. Inspect cooling loop and heat exchangers.",
        4500, 8,
    ),
    (
        lambda r: r["oil_temp"] > 120,
        "CRITICAL",
        "🔴 Oil temperature critically high — shutdown to prevent lubrication failure and seizure.",
        3800, 6,
    ),
    (
        lambda r: r["vibration"] > 6.5,
        "CRITICAL",
        "🔴 Severe vibration — possible shaft fracture or bearing collapse. Halt immediately.",
        6200, 12,
    ),
    (
        lambda r: r["pressure"] < 1.2,
        "CRITICAL",
        "🔴 Catastrophic pressure drop — hydraulic line rupture likely. Emergency inspection required.",
        5100, 10,
    ),
    (
        lambda r: r["power_kw"] > 105,
        "CRITICAL",
        "🔴 Power draw critically exceeded — isolate drive system, check motor insulation.",
        3200, 5,
    ),
    (
        lambda r: r["rpm"] > 4500,
        "CRITICAL",
        "🔴 Over-speed condition detected — engage mechanical stop, inspect governor/VFD.",
        2800, 4,
    ),
    # ── HIGH ────────────────────────────────────────────────────────────────
    (
        lambda r: 92 < r["temperature"] <= 115 and r.get("is_anomaly", False),
        "HIGH",
        "🟠 Cooling system degraded — inspect fans, coolant level, and heat-exchanger fouling within 4 h.",
        1800, 3,
    ),
    (
        lambda r: 90 < r["oil_temp"] <= 120,
        "HIGH",
        "🟠 Oil overheating — check oil level, filter blockage, and cooler bypass valve within 4 h.",
        1200, 2,
    ),
    (
        lambda r: 4.5 < r["vibration"] <= 6.5,
        "HIGH",
        "🟠 Elevated vibration — bearing assembly inspection and dynamic balance check within 24 h.",
        2400, 4,
    ),
    (
        lambda r: r.get("health_score", 100) < 38 and r.get("is_anomaly", False),
        "HIGH",
        "🟠 Multiple sensor anomalies with critical health score — schedule full preventive overhaul immediately.",
        5500, 16,
    ),
    (
        lambda r: r["humidity"] > 80,
        "HIGH",
        "🟠 High ambient humidity — inspect electrical enclosures for condensation; replace desiccant packs.",
        650, 1,
    ),
    (
        lambda r: r["rpm"] < 150 and r.get("is_anomaly", False),
        "HIGH",
        "🟠 Abnormal speed drop — inspect VFD, belt tension, and mechanical coupling within 24 h.",
        1500, 3,
    ),
    # ── MEDIUM ──────────────────────────────────────────────────────────────
    (
        lambda r: 38 <= r.get("health_score", 100) < 60,
        "MEDIUM",
        "🟡 Degraded health score — schedule comprehensive maintenance check within 72 h.",
        900, 4,
    ),
    (
        lambda r: r["runtime_hours"] > 8000,
        "MEDIUM",
        "🟡 High cumulative runtime (> 8 000 hr) — lubrication service, seal and gasket inspection due.",
        750, 3,
    ),
    (
        lambda r: 78 < r["temperature"] <= 92,
        "MEDIUM",
        "🟡 Elevated temperature trend — verify coolant concentration and inspect thermostat operation.",
        420, 1,
    ),
    (
        lambda r: 70 < r["oil_temp"] <= 90,
        "MEDIUM",
        "🟡 Slightly elevated oil temperature — check oil viscosity grade and last change interval.",
        280, 1,
    ),
    (
        lambda r: 60 < r["humidity"] <= 80,
        "MEDIUM",
        "🟡 Moderate humidity exceedance — improve enclosure sealing; schedule electrical inspection.",
        350, 1,
    ),
    (
        lambda r: r.get("rul_urgency", "Healthy") == "Soon",
        "MEDIUM",
        "🟡 Remaining Useful Life estimate below 500 h — plan component replacement in next maintenance window.",
        1100, 5,
    ),
    # ── LOW ─────────────────────────────────────────────────────────────────
    (
        lambda r: 2.8 < r["vibration"] <= 4.5,
        "LOW",
        "🟢 Minor vibration uptick — log reading and re-check at next scheduled interval.",
        120, 0.5,
    ),
    (
        lambda r: r["runtime_hours"] > 5000,
        "LOW",
        "🟢 Approaching mid-life runtime (> 5 000 hr) — review OEM service schedule.",
        200, 1,
    ),
    (
        lambda r: r.get("anomaly_category", "NONE") == "SENSOR_DRIFT",
        "LOW",
        "🟢 Possible sensor drift detected — calibrate affected sensors at next planned stop.",
        80, 0.5,
    ),
    # ── INFO ─────────────────────────────────────────────────────────────────
    (
        lambda r: r.get("health_score", 100) >= 70 and not r.get("is_anomaly", False),
        "INFO",
        "✅ All parameters nominal. No maintenance action required.",
        0, 0,
    ),
]

PRIORITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]


# ---------------------------------------------------------------------------
# Fault-type inference
# ---------------------------------------------------------------------------
_FAULT_INFERENCE_MAP: list[tuple[str, str]] = [
    # (anomaly_category, inferred_fault)
    ("THERMAL",       "Overheating / Cooling Failure"),
    ("MECHANICAL",    "Bearing Wear / Misalignment"),
    ("ELECTRICAL",    "Power Surge / Motor Fault"),
    ("ENVIRONMENTAL", "Humidity / Contamination Ingress"),
    ("COMPOSITE",     "Multi-System Degradation"),
    ("NONE",          "No Fault Detected"),
]

_FAULT_BY_CATEGORY = dict(_FAULT_INFERENCE_MAP)


def infer_fault_type(row: pd.Series) -> str:
    """
    Map anomaly_category (from anomaly detection engine) plus
    dominant sensor deviation to a human-readable probable fault label.
    """
    # Honour the data-generator's labelled fault type if present
    labelled = str(row.get("fault_type", "none"))
    if labelled not in ("none", "nan", ""):
        label_map = {
            "overheating":         "Overheating / Cooling Failure",
            "bearing_wear":        "Bearing Wear / Misalignment",
            "pressure_leak":       "Hydraulic Pressure Leak",
            "power_surge":         "Power Surge / Motor Fault",
            "lubrication_failure": "Lubrication Failure",
            "sensor_drift":        "Sensor Drift / Calibration",
        }
        return label_map.get(labelled, labelled.replace("_", " ").title())

    # Fall back to anomaly category
    cat = str(row.get("anomaly_category", "NONE"))
    return _FAULT_BY_CATEGORY.get(cat, "Unknown")


# ---------------------------------------------------------------------------
# Severity trend
# ---------------------------------------------------------------------------
def compute_severity_trend(machine_ts: pd.DataFrame) -> str:
    """
    Derive a severity trend label from the last 10 health_score readings.
    Returns: STABLE | WORSENING | CRITICAL_ESCALATION
    """
    if "health_score" not in machine_ts.columns or len(machine_ts) < 3:
        return "STABLE"

    recent = machine_ts["health_score"].tail(10).values.astype(float)
    if len(recent) < 2:
        return "STABLE"

    slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])

    if slope < -3.0:
        return "CRITICAL_ESCALATION"
    elif slope < -0.5:
        return "WORSENING"
    else:
        return "STABLE"


# ---------------------------------------------------------------------------
# Core recommendation logic
# ---------------------------------------------------------------------------

def get_recommendation(row: pd.Series) -> dict:
    """Return the highest-priority recommendation dict for a single machine row."""
    best = {
        "priority":        "INFO",
        "recommendation":  "✅ All parameters nominal. No maintenance action required.",
        "est_cost_usd":    0,
        "est_downtime_hr": 0,
    }

    for *rule_parts, est_cost, est_downtime in RULES:
        condition, priority, text = rule_parts
        try:
            if condition(row):
                if PRIORITY_ORDER.index(priority) < PRIORITY_ORDER.index(best["priority"]):
                    best = {
                        "priority":        priority,
                        "recommendation":  text,
                        "est_cost_usd":    est_cost,
                        "est_downtime_hr": est_downtime,
                    }
        except Exception:
            continue

    return best


def add_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns:
      alert_priority    – CRITICAL / HIGH / MEDIUM / LOW / INFO
      recommendation    – human-readable action string
      est_cost_usd      – estimated repair cost
      est_downtime_hr   – estimated downtime in hours
      inferred_fault    – probable fault label
      severity_trend    – STABLE / WORSENING / CRITICAL_ESCALATION
    """
    df = df.copy()

    # Build per-machine time-series cache for trend calculation
    ts_cache: dict[str, pd.DataFrame] = {
        mid: grp.sort_values("timestamp")
        for mid, grp in df.groupby("machine_id")
    }

    recs = df.apply(get_recommendation, axis=1)
    df["alert_priority"]    = recs.apply(lambda x: x["priority"])
    df["recommendation"]    = recs.apply(lambda x: x["recommendation"])
    df["est_cost_usd"]      = recs.apply(lambda x: x["est_cost_usd"])
    df["est_downtime_hr"]   = recs.apply(lambda x: x["est_downtime_hr"])
    df["inferred_fault"]    = df.apply(infer_fault_type, axis=1)
    df["severity_trend"]    = df["machine_id"].apply(
        lambda mid: compute_severity_trend(ts_cache.get(mid, pd.DataFrame()))
    )

    return df


# ---------------------------------------------------------------------------
# Richer maintenance history simulation
# ---------------------------------------------------------------------------
_HISTORY_ACTIONS: list[dict] = [
    {"action": "Bearing replacement",               "parts": "SKF 6308-2RS bearings (×2)",        "cost_range": (350, 900)},
    {"action": "Coolant flush and system refill",   "parts": "Coolant concentrate 20 L",           "cost_range": (120, 350)},
    {"action": "Pressure valve recalibration",      "parts": "Seal kit, pressure gauge",           "cost_range": (80, 280)},
    {"action": "Motor drive firmware update",       "parts": "None (software only)",               "cost_range": (0, 80)},
    {"action": "Full preventive maintenance",       "parts": "Filter set, oil, belts, seals",      "cost_range": (600, 1800)},
    {"action": "Hydraulic seal replacement",        "parts": "O-ring kit, hydraulic seals",        "cost_range": (200, 550)},
    {"action": "Thermal paste re-application",      "parts": "High-temp thermal compound",         "cost_range": (40, 120)},
    {"action": "Shaft alignment correction",        "parts": "Alignment shims",                    "cost_range": (180, 450)},
    {"action": "Lubrication service",               "parts": "ISO VG 46 oil 5 L, grease cartridge","cost_range": (90, 220)},
    {"action": "VFD parameter re-tuning",           "parts": "None (adjustment only)",             "cost_range": (0, 150)},
    {"action": "Sensor calibration",                "parts": "Calibration cert, reference gauge",  "cost_range": (60, 180)},
    {"action": "Humidity enclosure treatment",      "parts": "Desiccant packs, sealant",           "cost_range": (50, 160)},
]

_TECHNICIAN_NAMES = [f"Tech-{i:02d}" for i in range(1, 15)]

_NOTES_TEMPLATES = [
    "Found excessive wear; replaced and torqued to spec.",
    "System flushed; no further anomalies detected post-service.",
    "Calibrated; readings within ±0.5 % of reference.",
    "Firmware applied; drive now stable at full load.",
    "All service items completed per OEM schedule.",
    "Leak source identified at fitting; repaired and pressure-tested.",
    "Component re-seated; thermal imaging confirmed improvement.",
    "Laser-aligned to within 0.02 mm tolerance.",
    "Oil analysis sent to lab for particle count verification.",
    "Parameter set P1.04 adjusted; no over-speed recurrence.",
    "Three sensors calibrated; two replaced due to drift > 3 %.",
    "Enclosure resealed; IP65 integrity verified post-repair.",
]


def simulate_maintenance_history(machine_id: str, n: int = 8) -> pd.DataFrame:
    """
    Generate a realistic maintenance history log for a given machine.
    Includes action, parts replaced, technician, duration, cost, and notes.
    """
    rng    = np.random.default_rng(int(machine_id.split("-")[-1]) * 7)
    base   = datetime(2026, 1, 1)
    records = []

    for i in range(n):
        days_ago   = int(rng.integers(7, max(8, 200 - i * 22)))
        performed  = base - timedelta(days=days_ago)
        action_rec = _HISTORY_ACTIONS[int(rng.integers(0, len(_HISTORY_ACTIONS)))]
        cost_lo, cost_hi = action_rec["cost_range"]
        cost       = int(rng.integers(cost_lo, max(cost_lo + 1, cost_hi)))
        duration   = float(rng.choice([0.5, 1.0, 2.0, 4.0, 8.0]))
        tech       = str(rng.choice(_TECHNICIAN_NAMES))
        note       = str(rng.choice(_NOTES_TEMPLATES))

        records.append({
            "Date":           performed.strftime("%Y-%m-%d"),
            "Machine":        machine_id,
            "Action":         action_rec["action"],
            "Parts Replaced": action_rec["parts"],
            "Technician":     tech,
            "Duration (hr)":  duration,
            "Cost (USD)":     cost,
            "Notes":          note,
            "Status":         "Completed",
        })

    return pd.DataFrame(records).sort_values("Date", ascending=False).reset_index(drop=True)
