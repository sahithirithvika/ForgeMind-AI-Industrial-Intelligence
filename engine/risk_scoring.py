"""
ForgeMind AI – Risk Scoring Engine  v2.0

Improvements over v1:
  - Per-machine-type sensor thresholds (not one-size-fits-all)
  - 7 sensor channels: temperature, vibration, pressure, power_kw,
                        humidity, rpm, oil_temp
  - Per-type sensor weights (vibration matters more on a Compressor, etc.)
  - Trend-direction penalty: rising temp/vibration over the last N readings
    adds an extra degradation factor
  - Remaining Useful Life (RUL) estimate in hours based on health trajectory
  - Anomaly confidence feeds into the health score (not just a binary flag)
"""

import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Per-machine-type sensor thresholds
# Format: {sensor: (safe_min, safe_max, danger_max)}
# ---------------------------------------------------------------------------
TYPE_THRESHOLDS: dict[str, dict[str, tuple]] = {
    "CNC Milling Machine": {
        "temperature": (20,  80,  140),
        "vibration":   (0,   2.5,  10),
        "pressure":    (0,    7,   15),
        "power_kw":    (0,   35,   80),
        "humidity":    (20,  60,   90),
        "rpm":         (500, 2600, 4000),
        "oil_temp":    (20,  70,  120),
    },
    "Hydraulic Press": {
        "temperature": (20,  90,  150),
        "vibration":   (0,   4.0,  12),
        "pressure":    (0,   20,   30),
        "power_kw":    (0,   60,  100),
        "humidity":    (15,  55,   90),
        "rpm":         (200, 1400, 2500),
        "oil_temp":    (20,  85,  130),
    },
    "Industrial Compressor": {
        "temperature": (20, 100,  155),
        "vibration":   (0,   4.5,  12),
        "pressure":    (0,   15,   28),
        "power_kw":    (0,   75,  115),
        "humidity":    (10,  50,   85),
        "rpm":         (800, 3800, 5000),
        "oil_temp":    (20,  95,  135),
    },
    "Conveyor Drive Motor": {
        "temperature": (20,  70,  130),
        "vibration":   (0,   2.0,   8),
        "pressure":    (0,    5,   12),
        "power_kw":    (0,   25,   60),
        "humidity":    (20,  65,   90),
        "rpm":         (300, 2000, 3500),
        "oil_temp":    (20,  65,  115),
    },
    "Injection Moulding Unit": {
        "temperature": (20, 115,  160),
        "vibration":   (0,   3.0,  10),
        "pressure":    (0,   22,   30),
        "power_kw":    (0,   85,  115),
        "humidity":    (10,  45,   80),
        "rpm":         (200, 1600, 2800),
        "oil_temp":    (20, 100,  140),
    },
    "Robotic Welding Arm": {
        "temperature": (20,  85,  145),
        "vibration":   (0,   2.8,  10),
        "pressure":    (0,    8,   18),
        "power_kw":    (0,   50,   90),
        "humidity":    (10,  50,   85),
        "rpm":         (100, 1000, 1800),
        "oil_temp":    (20,  75,  125),
    },
}

# Default fallback thresholds
DEFAULT_THRESHOLDS = {
    "temperature": (20,  90,  150),
    "vibration":   (0,   4.0,  12),
    "pressure":    (0,   20,   30),
    "power_kw":    (0,   80,  120),
    "humidity":    (10,  65,   90),
    "rpm":         (200, 3600, 5000),
    "oil_temp":    (20,  90,  140),
}

# Per-machine-type sensor weights (must sum to 1)
TYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "CNC Milling Machine":       dict(temperature=0.22, vibration=0.28, pressure=0.15,
                                      power_kw=0.12, humidity=0.07, rpm=0.10, oil_temp=0.06),
    "Hydraulic Press":           dict(temperature=0.20, vibration=0.25, pressure=0.22,
                                      power_kw=0.12, humidity=0.05, rpm=0.08, oil_temp=0.08),
    "Industrial Compressor":     dict(temperature=0.22, vibration=0.30, pressure=0.18,
                                      power_kw=0.12, humidity=0.04, rpm=0.08, oil_temp=0.06),
    "Conveyor Drive Motor":      dict(temperature=0.18, vibration=0.32, pressure=0.12,
                                      power_kw=0.14, humidity=0.06, rpm=0.12, oil_temp=0.06),
    "Injection Moulding Unit":   dict(temperature=0.25, vibration=0.20, pressure=0.22,
                                      power_kw=0.14, humidity=0.04, rpm=0.08, oil_temp=0.07),
    "Robotic Welding Arm":       dict(temperature=0.22, vibration=0.28, pressure=0.15,
                                      power_kw=0.14, humidity=0.05, rpm=0.10, oil_temp=0.06),
}

DEFAULT_WEIGHTS = dict(temperature=0.22, vibration=0.28, pressure=0.18,
                       power_kw=0.13, humidity=0.06, rpm=0.08, oil_temp=0.05)

SENSOR_COLS = ["temperature", "vibration", "pressure",
               "power_kw", "humidity", "rpm", "oil_temp"]

# ---------------------------------------------------------------------------
# RUL parameters
# ---------------------------------------------------------------------------
MAX_RUNTIME_HOURS   = 12_000          # design life
HEALTH_FLOOR        = 10.0            # score at "end of life"
TREND_WINDOW        = 10              # readings to look back for trend penalty


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _sensor_penalty(value: float,
                    safe_min: float, safe_max: float,
                    danger_max: float) -> float:
    """0 (healthy) → 1 (critical) penalty for a single sensor reading."""
    if value <= safe_max:
        excess = max(0.0, value - safe_min)
        span   = max(safe_max - safe_min, 1e-6)
        return min(excess / span, 1.0) * 0.15    # small base stress ≤ 0.15
    else:
        ratio = (value - safe_max) / max(danger_max - safe_max, 1e-6)
        return min(0.15 + ratio * 0.85, 1.0)


def _trend_penalty(machine_ts: pd.DataFrame, sensor: str,
                   window: int = TREND_WINDOW) -> float:
    """
    Compute an upward-trend penalty (0–0.15) for a sensor by
    fitting a linear slope over the last `window` readings.
    Positive slope (rising) adds penalty; flat/falling adds 0.
    """
    series = machine_ts[sensor].tail(window)
    if len(series) < 3:
        return 0.0
    x = np.arange(len(series), dtype=float)
    y = series.values.astype(float)
    # Normalise slope by the safe_max of the sensor so the scale is comparable
    slope = float(np.polyfit(x, y, 1)[0])
    if slope <= 0:
        return 0.0
    # A slope of 1 unit/step in normalised terms → ~0.05 extra penalty
    normalised_slope = slope / (y.mean() + 1e-6)
    return float(min(normalised_slope * 2.5, 0.15))


def compute_health_score(
    row: pd.Series,
    machine_ts: Optional[pd.DataFrame] = None,
) -> float:
    """
    Returns health score in [0, 100].  100 = perfect, 0 = failure.

    Parameters
    ----------
    row        : latest sensor reading for a single machine
    machine_ts : full time-series for that machine (enables trend penalty)
    """
    mtype      = row.get("machine_type", "_default")
    thresholds = TYPE_THRESHOLDS.get(mtype, DEFAULT_THRESHOLDS)
    weights    = TYPE_WEIGHTS.get(mtype, DEFAULT_WEIGHTS)

    total_penalty = 0.0
    for sensor in SENSOR_COLS:
        if sensor not in row or sensor not in thresholds:
            continue
        s_min, s_max, d_max = thresholds[sensor]
        p = _sensor_penalty(float(row[sensor]), s_min, s_max, d_max)
        total_penalty += weights.get(sensor, 0.0) * p

    # Runtime penalty (0–0.12)
    runtime_frac   = min(float(row.get("runtime_hours", 0)) / MAX_RUNTIME_HOURS, 1.0)
    total_penalty += runtime_frac * 0.12

    # Trend penalty (0–0.10) – uses per-sensor worst-case trend
    if machine_ts is not None and len(machine_ts) >= 3:
        trend_p = max(
            _trend_penalty(machine_ts, "temperature"),
            _trend_penalty(machine_ts, "vibration"),
            _trend_penalty(machine_ts, "oil_temp"),
        )
        total_penalty = min(total_penalty + trend_p, 1.0)

    # Anomaly confidence penalty (replaces binary flag)
    conf = float(row.get("anomaly_confidence", 0.0))
    if conf > 0:
        total_penalty = min(total_penalty + (conf / 100) * 0.22, 1.0)

    health = round((1 - total_penalty) * 100, 1)
    return float(max(0.0, min(100.0, health)))


def estimate_rul(health_score: float, runtime_hours: float) -> float:
    """
    Remaining Useful Life (hours) – simple linear degradation model.

    Assumes health degrades linearly from 100 to HEALTH_FLOOR over
    MAX_RUNTIME_HOURS; current position is estimated from health_score.
    """
    if health_score <= HEALTH_FLOOR:
        return 0.0
    # Health rate of decline (per hour) based on current runtime
    if runtime_hours <= 0:
        runtime_hours = 1.0
    current_rate = (100 - health_score) / max(runtime_hours, 1.0)   # points/hr
    if current_rate <= 0:
        return float(MAX_RUNTIME_HOURS - runtime_hours)
    remaining_health = health_score - HEALTH_FLOOR
    rul = remaining_health / current_rate
    return float(min(round(rul, 0), MAX_RUNTIME_HOURS))


def label_risk(score: float) -> str:
    if score >= 70:
        return "Low"
    elif score >= 40:
        return "Medium"
    else:
        return "High"


def label_rul_urgency(rul_hours: float) -> str:
    if rul_hours < 100:
        return "Critical"
    elif rul_hours < 500:
        return "Soon"
    elif rul_hours < 2000:
        return "Planned"
    else:
        return "Healthy"


# ---------------------------------------------------------------------------
# DataFrame-level function
# ---------------------------------------------------------------------------

def add_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds to df:
      health_score   – 0–100
      risk_level     – Low / Medium / High
      failure_prob   – 0–100 %
      rul_hours      – Remaining Useful Life estimate (hours)
      rul_urgency    – Critical / Soon / Planned / Healthy
    """
    df = df.copy()

    # ── Pre-compute per-machine trend penalty (avoid O(n²) per-row lookup) ──
    trend_penalties: dict[str, float] = {}
    for mid, grp in df.groupby("machine_id"):
        ts = grp.sort_values("timestamp")
        tp = max(
            _trend_penalty(ts, "temperature"),
            _trend_penalty(ts, "vibration"),
            _trend_penalty(ts, "oil_temp") if "oil_temp" in ts.columns else 0.0,
        )
        trend_penalties[str(mid)] = tp

    def _health_for_row(row: pd.Series) -> float:
        """Vectorised-friendly per-row health scorer (no repeated ts_cache lookup)."""
        mtype      = row.get("machine_type", "_default")
        thresholds = TYPE_THRESHOLDS.get(mtype, DEFAULT_THRESHOLDS)
        weights    = TYPE_WEIGHTS.get(mtype, DEFAULT_WEIGHTS)

        total_penalty = 0.0
        for sensor in SENSOR_COLS:
            if sensor not in row or sensor not in thresholds:
                continue
            s_min, s_max, d_max = thresholds[sensor]
            p = _sensor_penalty(float(row[sensor]), s_min, s_max, d_max)
            total_penalty += weights.get(sensor, 0.0) * p

        # Runtime penalty (0–0.12)
        runtime_frac   = min(float(row.get("runtime_hours", 0)) / MAX_RUNTIME_HOURS, 1.0)
        total_penalty += runtime_frac * 0.12

        # Trend penalty – pre-computed per machine
        tp = trend_penalties.get(str(row.get("machine_id", "")), 0.0)
        total_penalty = min(total_penalty + tp, 1.0)

        # Anomaly confidence penalty
        conf = float(row.get("anomaly_confidence", 0.0))
        if conf > 0:
            total_penalty = min(total_penalty + (conf / 100) * 0.22, 1.0)

        health = round((1 - total_penalty) * 100, 1)
        return float(max(0.0, min(100.0, health)))

    df["health_score"]  = df.apply(_health_for_row, axis=1)
    df["risk_level"]    = df["health_score"].apply(label_risk)
    df["failure_prob"]  = df["health_score"].apply(
        lambda s: round((1 - s / 100) ** 1.6 * 100, 1)
    )
    df["rul_hours"]     = df.apply(
        lambda r: estimate_rul(r["health_score"], r["runtime_hours"]), axis=1
    )
    df["rul_urgency"]   = df["rul_hours"].apply(label_rul_urgency)

    return df
