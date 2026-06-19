"""
ForgeMind AI – Industrial Sensor Data Generator
Generates realistic simulated sensor data for heavy industrial machinery.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_MACHINES = 20
RECORDS_PER_MACHINE = 60          # 60 time-steps per machine = 1 200 total records
SEED = 42
np.random.seed(SEED)

MACHINE_TYPES = [
    "CNC Milling Machine",
    "Hydraulic Press",
    "Industrial Compressor",
    "Conveyor Drive Motor",
    "Injection Moulding Unit",
]

# Normal operating ranges per machine type
PROFILES = {
    "CNC Milling Machine":       dict(temp=(55,  75),  vib=(0.5, 2.0), pres=(4.0,  6.0),  power=(15, 30)),
    "Hydraulic Press":           dict(temp=(60,  85),  vib=(1.0, 3.5), pres=(10.0, 18.0), power=(30, 55)),
    "Industrial Compressor":     dict(temp=(70,  95),  vib=(1.5, 4.0), pres=(8.0,  14.0), power=(40, 70)),
    "Conveyor Drive Motor":      dict(temp=(45,  65),  vib=(0.3, 1.5), pres=(2.0,  4.0),  power=(8,  20)),
    "Injection Moulding Unit":   dict(temp=(80, 110),  vib=(0.8, 2.5), pres=(12.0, 20.0), power=(50, 80)),
}


def _apply_fault(row: dict, fault_type: str) -> dict:
    """Inject a specific fault pattern into a sensor row."""
    if fault_type == "overheating":
        row["temperature"]   += np.random.uniform(20, 45)
        row["power_kw"]      += np.random.uniform(10, 25)
    elif fault_type == "bearing_wear":
        row["vibration"]     += np.random.uniform(3.0, 6.0)
        row["temperature"]   += np.random.uniform(5, 15)
    elif fault_type == "pressure_leak":
        row["pressure"]      -= np.random.uniform(2.0, 5.0)
        row["vibration"]     += np.random.uniform(1.0, 3.0)
    elif fault_type == "power_surge":
        row["power_kw"]      += np.random.uniform(20, 40)
        row["temperature"]   += np.random.uniform(10, 20)
    return row


def generate_dataset(n_machines: int = NUM_MACHINES,
                     records_per_machine: int = RECORDS_PER_MACHINE) -> pd.DataFrame:
    records = []
    base_time = datetime(2026, 6, 1, 6, 0, 0)

    for m_idx in range(1, n_machines + 1):
        machine_id   = f"MCH-{m_idx:03d}"
        machine_type = MACHINE_TYPES[(m_idx - 1) % len(MACHINE_TYPES)]
        profile      = PROFILES[machine_type]
        runtime_base = np.random.uniform(200, 8000)   # hours on the clock

        # ~15 % of machines have some fault injected
        fault = None
        fault_probability = np.random.rand()
        if fault_probability > 0.85:
            fault = np.random.choice(["overheating", "bearing_wear",
                                      "pressure_leak", "power_surge"])

        for step in range(records_per_machine):
            timestamp = base_time + timedelta(minutes=step * 15)

            row = {
                "machine_id":    machine_id,
                "machine_type":  machine_type,
                "timestamp":     timestamp,
                "temperature":   np.random.uniform(*profile["temp"])
                                 + np.random.normal(0, 1.5),
                "vibration":     np.random.uniform(*profile["vib"])
                                 + np.random.normal(0, 0.15),
                "pressure":      np.random.uniform(*profile["pres"])
                                 + np.random.normal(0, 0.3),
                "power_kw":      np.random.uniform(*profile["power"])
                                 + np.random.normal(0, 1.0),
                "runtime_hours": round(runtime_base + step * 0.25, 2),
            }

            # Inject fault only on last 20 % of steps for that machine
            if fault and step >= int(records_per_machine * 0.8):
                row = _apply_fault(row, fault)

            records.append(row)

    df = pd.DataFrame(records)

    # Clip to physical limits
    df["temperature"] = df["temperature"].clip(lower=20, upper=160)
    df["vibration"]   = df["vibration"].clip(lower=0,   upper=15)
    df["pressure"]    = df["pressure"].clip(lower=0,    upper=30)
    df["power_kw"]    = df["power_kw"].clip(lower=0,    upper=120)

    # Round sensor values
    df["temperature"] = df["temperature"].round(2)
    df["vibration"]   = df["vibration"].round(3)
    df["pressure"]    = df["pressure"].round(2)
    df["power_kw"]    = df["power_kw"].round(2)

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
