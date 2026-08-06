"""
ForgeMind AI – Industrial Sensor Data Generator  v2.0
Generates realistic simulated sensor telemetry for heavy industrial machinery.

Improvements over v1:
  - 30 machines (up from 20), richer diversity
  - 7 sensor channels: temperature, vibration, pressure, power_kw,
                        humidity, rpm, oil_temp
  - 6 machine types (added Robotic Welding Arm)
  - Per-machine-type operating profiles for every sensor
  - Gradual fault degradation: faults ramp up progressively, not just in
    the last 20 % of steps
  - Realistic noise model: each sensor has its own σ
  - Fault-type column retained in output for supervised learning / labelling
  - Drift simulation: slow trend injected into some healthy machines
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_MACHINES          = 30
RECORDS_PER_MACHINE   = 96          # 96 × 15-min steps = 24 hours of data
SEED                  = 42
np.random.seed(SEED)

MACHINE_TYPES = [
    "CNC Milling Machine",
    "Hydraulic Press",
    "Industrial Compressor",
    "Conveyor Drive Motor",
    "Injection Moulding Unit",
    "Robotic Welding Arm",          # NEW in v2
]

# ---------------------------------------------------------------------------
# Per-machine-type operating profiles
# Each entry: (min, max) for normal operation, noise_sigma
# Keys: temp, vib, pres, power, humidity, rpm, oil_temp
# ---------------------------------------------------------------------------
PROFILES = {
    "CNC Milling Machine": dict(
        temp     =(55,  75,  1.5),
        vib      =(0.5, 2.0, 0.10),
        pres     =(4.0, 6.0, 0.20),
        power    =(15,  30,  0.80),
        humidity =(30,  55,  1.0),
        rpm      =(800, 2400, 30),
        oil_temp =(45,  65,  1.2),
    ),
    "Hydraulic Press": dict(
        temp     =(60,  85,  2.0),
        vib      =(1.0, 3.5, 0.15),
        pres     =(10., 18., 0.30),
        power    =(30,  55,  1.0),
        humidity =(25,  50,  1.0),
        rpm      =(400, 1200, 20),
        oil_temp =(55,  80,  1.5),
    ),
    "Industrial Compressor": dict(
        temp     =(70,  95,  2.5),
        vib      =(1.5, 4.0, 0.20),
        pres     =(8.0, 14., 0.35),
        power    =(40,  70,  1.5),
        humidity =(20,  45,  0.8),
        rpm      =(1200,3600, 40),
        oil_temp =(65,  90,  2.0),
    ),
    "Conveyor Drive Motor": dict(
        temp     =(45,  65,  1.0),
        vib      =(0.3, 1.5, 0.08),
        pres     =(2.0, 4.0, 0.15),
        power    =(8,   20,  0.50),
        humidity =(35,  60,  1.2),
        rpm      =(600, 1800, 25),
        oil_temp =(40,  60,  1.0),
    ),
    "Injection Moulding Unit": dict(
        temp     =(80, 110,  3.0),
        vib      =(0.8, 2.5, 0.12),
        pres     =(12., 20., 0.40),
        power    =(50,  80,  2.0),
        humidity =(15,  40,  0.8),
        rpm      =(500, 1500, 20),
        oil_temp =(70,  95,  2.5),
    ),
    "Robotic Welding Arm": dict(
        temp     =(50,  80,  2.0),
        vib      =(0.6, 2.2, 0.12),
        pres     =(3.0, 7.0, 0.25),
        power    =(20,  45,  1.0),
        humidity =(20,  45,  0.8),
        rpm      =(300, 900,  15),
        oil_temp =(45,  70,  1.5),
    ),
}

# Fault injection: what changes and by how much at full severity
FAULT_EFFECTS = {
    "overheating": {
        "temperature": (+30, +50),
        "power_kw":    (+12, +28),
        "oil_temp":    (+15, +30),
    },
    "bearing_wear": {
        "vibration":   (+3.0, +6.5),
        "temperature": (+5,  +18),
        "rpm":         (-200, -50),
    },
    "pressure_leak": {
        "pressure":    (-3.0, -6.0),
        "vibration":   (+0.8, +3.0),
        "power_kw":    (+5,  +15),
    },
    "power_surge": {
        "power_kw":    (+22, +42),
        "temperature": (+10, +22),
        "oil_temp":    (+8,  +18),
    },
    "lubrication_failure": {     # NEW fault type
        "oil_temp":    (+20, +40),
        "vibration":   (+1.5, +4.0),
        "rpm":         (-300, -80),
    },
    "sensor_drift": {            # NEW fault type – subtle, hard to catch
        "temperature": (+3,  +10),
        "pressure":    (-1.0, -2.5),
        "humidity":    (+10, +20),
    },
}

FAULT_PROBABILITY = 0.22          # ~22 % of machines get a fault (up from 15 %)


# ---------------------------------------------------------------------------
# Helper: gradual fault ramp-up
# ---------------------------------------------------------------------------
def _ramp_severity(step: int, total_steps: int, start_frac: float = 0.55) -> float:
    """
    Returns a 0→1 severity scalar that ramps from 0 at `start_frac`*total_steps
    to 1 at total_steps.  Before start_frac it returns 0.
    """
    start = int(total_steps * start_frac)
    if step < start:
        return 0.0
    elapsed = step - start
    window  = max(total_steps - start, 1)
    return min(elapsed / window, 1.0)


def _apply_fault(row: dict, fault_type: str, severity: float,
                 rng: np.random.Generator) -> dict:
    """
    Inject a fault into `row` scaled by `severity` (0–1).
    Uses the FAULT_EFFECTS table for cleaner, data-driven application.
    delta_lo/delta_hi may be negative (e.g. pressure drop, rpm drop) —
    always sample from (min, max) to avoid ValueError from numpy.
    """
    effects = FAULT_EFFECTS.get(fault_type, {})
    for sensor, (delta_lo, delta_hi) in effects.items():
        if sensor in row:
            lo, hi = min(delta_lo, delta_hi), max(delta_lo, delta_hi)
            delta  = rng.uniform(lo, hi) * severity
            row[sensor] = row[sensor] + delta
    return row


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_dataset(
    n_machines: int = NUM_MACHINES,
    records_per_machine: int = RECORDS_PER_MACHINE,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      machine_id, machine_type, timestamp,
      temperature, vibration, pressure, power_kw,
      humidity, rpm, oil_temp,
      runtime_hours, fault_type
    """
    rng       = np.random.default_rng(seed)
    records   = []
    base_time = datetime(2026, 6, 1, 6, 0, 0)

    for m_idx in range(1, n_machines + 1):
        machine_id   = f"MCH-{m_idx:03d}"
        machine_type = MACHINE_TYPES[(m_idx - 1) % len(MACHINE_TYPES)]
        profile      = PROFILES[machine_type]
        runtime_base = float(rng.uniform(200, 9500))   # hours on the clock

        # Fault assignment
        fault_type = None
        if rng.random() < FAULT_PROBABILITY:
            fault_type = str(rng.choice(list(FAULT_EFFECTS.keys())))

        # Slow drift: small upward temp & power trend for some healthy machines
        apply_drift = (fault_type is None) and (rng.random() < 0.25)

        for step in range(records_per_machine):
            timestamp = base_time + timedelta(minutes=step * 15)

            # Sample base sensor values
            def _sample(key: str) -> float:
                lo, hi, sigma = profile[key]
                return float(rng.uniform(lo, hi) + rng.normal(0, sigma))

            row: dict = {
                "machine_id":    machine_id,
                "machine_type":  machine_type,
                "timestamp":     timestamp,
                "temperature":   _sample("temp"),
                "vibration":     _sample("vib"),
                "pressure":      _sample("pres"),
                "power_kw":      _sample("power"),
                "humidity":      _sample("humidity"),
                "rpm":           _sample("rpm"),
                "oil_temp":      _sample("oil_temp"),
                "runtime_hours": round(runtime_base + step * 0.25, 2),
                "fault_type":    fault_type if fault_type else "none",
            }

            # Gradual fault injection
            if fault_type:
                severity = _ramp_severity(step, records_per_machine, start_frac=0.50)
                if severity > 0:
                    row = _apply_fault(row, fault_type, severity, rng)

            # Drift simulation on healthy machines
            if apply_drift:
                drift_frac  = step / records_per_machine
                row["temperature"] += drift_frac * rng.uniform(3, 8)
                row["power_kw"]    += drift_frac * rng.uniform(2, 5)

            records.append(row)

    df = pd.DataFrame(records)

    # Physical clipping
    df["temperature"] = df["temperature"].clip(lower=20,   upper=180)
    df["vibration"]   = df["vibration"].clip(lower=0,     upper=15)
    df["pressure"]    = df["pressure"].clip(lower=0,      upper=30)
    df["power_kw"]    = df["power_kw"].clip(lower=0,      upper=120)
    df["humidity"]    = df["humidity"].clip(lower=5,      upper=95)
    df["rpm"]         = df["rpm"].clip(lower=50,          upper=5000)
    df["oil_temp"]    = df["oil_temp"].clip(lower=20,     upper=140)

    # Rounding
    df["temperature"] = df["temperature"].round(2)
    df["vibration"]   = df["vibration"].round(3)
    df["pressure"]    = df["pressure"].round(2)
    df["power_kw"]    = df["power_kw"].round(2)
    df["humidity"]    = df["humidity"].round(1)
    df["rpm"]         = df["rpm"].round(0).astype(int)
    df["oil_temp"]    = df["oil_temp"].round(2)

    return df.reset_index(drop=True)


def save_dataset(path: str = "data/sensor_data.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = generate_dataset()
    df.to_csv(path, index=False)
    print(f"[DataGen] Saved {len(df)} records → {path}")
    return df


if __name__ == "__main__":
    df = save_dataset()
    print(df.describe())
    print("\nFault distribution:\n", df["fault_type"].value_counts())
