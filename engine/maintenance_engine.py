"""
ForgeMind AI – Maintenance Recommendation Engine
Generates actionable maintenance recommendations based on sensor readings,
anomaly flags, risk scores, and machine type.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Rule-based recommendation matrix
# ---------------------------------------------------------------------------
RULES = [
    # (condition_fn, priority, recommendation_text)
    (
        lambda r: r["temperature"] > 110,
        "CRITICAL",
        "🔴 Immediate shutdown recommended — thermal runaway risk detected.",
    ),
    (
        lambda r: r["vibration"] > 6.0,
        "CRITICAL",
        "🔴 Immediate vibration analysis required — possible shaft misalignment or bearing failure.",
    ),
    (
        lambda r: r["pressure"] < 1.5,
        "CRITICAL",
        "🔴 Severe pressure drop detected — inspect for hydraulic line rupture immediately.",
    ),
    (
        lambda r: r["power_kw"] > 100,
        "CRITICAL",
        "🔴 Power consumption critically high — check motor insulation and drive system.",
    ),
    (
        lambda r: 90 < r["temperature"] <= 110 and r["is_anomaly"],
        "HIGH",
        "🟠 Inspect cooling system within 4 hours — heat dissipation degraded.",
    ),
    (
        lambda r: 4.0 < r["vibration"] <= 6.0,
        "HIGH",
        "🟠 Inspect bearing assembly within 24 hours — elevated vibration signature.",
    ),
    (
        lambda r: r["health_score"] < 40 and r["is_anomaly"],
        "HIGH",
        "🟠 Schedule immediate preventive maintenance — multiple sensor anomalies detected.",
    ),
    (
        lambda r: 40 <= r["health_score"] < 60,
        "MEDIUM",
        "🟡 Cooling system maintenance recommended — schedule within 72 hours.",
    ),
    (
        lambda r: r["runtime_hours"] > 7000,
        "MEDIUM",
        "🟡 High cumulative runtime — lubrication and seal inspection due.",
    ),
    (
        lambda r: 75 < r["temperature"] <= 90,
        "MEDIUM",
        "🟡 Elevated temperature trend detected — monitor closely and verify coolant levels.",
    ),
    (
        lambda r: 2.5 < r["vibration"] <= 4.0,
        "LOW",
        "🟢 Minor vibration increase noted — log for next scheduled maintenance.",
    ),
    (
        lambda r: r["health_score"] >= 70 and not r["is_anomaly"],
        "INFO",
        "✅ Machine operating within normal parameters. No action required.",
    ),
]


def get_recommendation(row: pd.Series) -> dict:
    """Return the highest-priority recommendation for a single machine row."""
    priority_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
    best = {"priority": "INFO",
            "recommendation": "✅ Machine operating within normal parameters."}

    for condition, priority, text in RULES:
        try:
            if condition(row):
                if priority_order.index(priority) < priority_order.index(best["priority"]):
                    best = {"priority": priority, "recommendation": text}
        except Exception:
            continue

    return best


def add_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    recs = df.apply(get_recommendation, axis=1)
    df["alert_priority"]   = recs.apply(lambda x: x["priority"])
    df["recommendation"]   = recs.apply(lambda x: x["recommendation"])
    return df


# ---------------------------------------------------------------------------
# Maintenance history simulation
# ---------------------------------------------------------------------------
HISTORY_TEMPLATES = [
    "Bearing replacement – resolved vibration anomaly",
    "Coolant flush and refill – temperature normalised",
    "Pressure valve recalibration – system stable",
    "Motor drive firmware update – power spikes resolved",
    "Full preventive maintenance service",
    "Seal replacement – pressure leak resolved",
    "Thermal paste application – overheating resolved",
    "Alignment check and correction – vibration reduced",
]


def simulate_maintenance_history(machine_id: str, n: int = 5) -> pd.DataFrame:
    """Generate a fake maintenance history log for a given machine."""
    import numpy as np
    from datetime import datetime, timedelta
    rng = np.random.default_rng(int(machine_id.split("-")[-1]))

    records = []
    base = datetime(2026, 1, 1)
    for i in range(n):
        days_ago  = rng.integers(10, 180 - i * 30)
        performed = base - timedelta(days=int(days_ago))
        records.append({
            "Date":          performed.strftime("%Y-%m-%d"),
            "Machine":       machine_id,
            "Action":        rng.choice(HISTORY_TEMPLATES),
            "Technician":    f"Tech-{rng.integers(1, 10):02d}",
            "Duration (hr)": float(rng.choice([1, 2, 4, 8])),
            "Status":        "Completed",
        })

    return pd.DataFrame(records).sort_values("Date", ascending=False)
