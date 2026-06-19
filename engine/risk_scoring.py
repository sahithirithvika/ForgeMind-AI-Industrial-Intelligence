"""
ForgeMind AI – Risk Scoring Engine
Computes a 0–100 machine health score and maps it to risk levels.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Thresholds used to normalise each sensor to a 0–1 penalty score
# (above max_safe → full penalty; within safe range → 0 penalty)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    #               safe_min  safe_max  danger_max
    "temperature":  (20,       90,       150),
    "vibration":    (0,        4.0,      12.0),
    "pressure":     (0,        20,       30),
    "power_kw":     (0,        80,       120),
}

WEIGHTS = {
    "temperature": 0.30,
    "vibration":   0.35,
    "pressure":    0.20,
    "power_kw":    0.15,
}


def _sensor_penalty(value: float, safe_min: float,
                    safe_max: float, danger_max: float) -> float:
    """Returns 0 (healthy) → 1 (critical) for a single sensor reading."""
    if value <= safe_max:
        # Scale from safe range to 0 penalty
        excess = max(0, value - safe_min)
        return min(excess / (safe_max - safe_min), 1.0) * 0.2   # small base stress
    else:
        ratio = (value - safe_max) / max(danger_max - safe_max, 1e-6)
        return min(0.2 + ratio * 0.8, 1.0)


def compute_health_score(row: pd.Series) -> float:
    """
    Returns a health score in [0, 100].
    100 = perfect health, 0 = imminent failure.
    """
    total_penalty = 0.0
    for col, (s_min, s_max, d_max) in THRESHOLDS.items():
        penalty = _sensor_penalty(row[col], s_min, s_max, d_max)
        total_penalty += WEIGHTS[col] * penalty

    # Runtime penalty – older machines carry more stress
    runtime_factor = min(row.get("runtime_hours", 0) / 10_000, 1.0) * 0.15
    total_penalty  += runtime_factor

    # Anomaly boost
    if row.get("is_anomaly", False):
        total_penalty = min(total_penalty + 0.25, 1.0)

    health = round((1 - total_penalty) * 100, 1)
    return max(0.0, min(100.0, health))


def label_risk(score: float) -> str:
    if score >= 70:
        return "Low"
    elif score >= 40:
        return "Medium"
    else:
        return "High"


def add_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["health_score"] = df.apply(compute_health_score, axis=1)
    df["risk_level"]   = df["health_score"].apply(label_risk)
    df["failure_prob"] = df["health_score"].apply(
        lambda s: round((1 - s / 100) ** 1.8 * 100, 1)
    )
    return df
