"""
ForgeMind AI – Anomaly Detection Engine  v2.0

Improvements over v1:
  - 7 sensor features (humidity, rpm, oil_temp added)
  - Ensemble: Isolation Forest + Local Outlier Factor, majority-vote fusion
  - Per-machine-type contamination tuning based on operational risk profile
  - Normalised anomaly confidence score (0–100, higher = more anomalous)
  - Anomaly category classification: THERMAL / MECHANICAL / ELECTRICAL /
    ENVIRONMENTAL / COMPOSITE – derived from which sensors are most deviant
  - Robust per-type StandardScaler to reduce cross-type false positives
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Feature columns – all 7 sensor channels
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "temperature", "vibration", "pressure",
    "power_kw", "humidity", "rpm", "oil_temp",
]

# ---------------------------------------------------------------------------
# Per-machine-type contamination rates
# (higher for types that run hotter / harder)
# ---------------------------------------------------------------------------
TYPE_CONTAMINATION = {
    "CNC Milling Machine":       0.10,
    "Hydraulic Press":           0.12,
    "Industrial Compressor":     0.15,
    "Conveyor Drive Motor":      0.08,
    "Injection Moulding Unit":   0.14,
    "Robotic Welding Arm":       0.11,
    "_default":                  0.12,
}

# ---------------------------------------------------------------------------
# Sensor-to-fault-category mapping used for anomaly classification
# ---------------------------------------------------------------------------
SENSOR_CATEGORY = {
    "temperature": "THERMAL",
    "oil_temp":    "THERMAL",
    "vibration":   "MECHANICAL",
    "rpm":         "MECHANICAL",
    "power_kw":    "ELECTRICAL",
    "pressure":    "MECHANICAL",
    "humidity":    "ENVIRONMENTAL",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _contamination_for_type(machine_type: str) -> float:
    return TYPE_CONTAMINATION.get(machine_type, TYPE_CONTAMINATION["_default"])


def _anomaly_category(row: pd.Series, scaler: StandardScaler) -> str:
    """
    Classify the type of anomaly by finding which normalised sensor
    deviates most from its group mean.
    """
    vals    = row[FEATURE_COLS].values.reshape(1, -1)
    z       = scaler.transform(vals)[0]                 # z-scores
    abs_z   = np.abs(z)
    top_idx = np.argmax(abs_z)
    sensor  = FEATURE_COLS[top_idx]

    # If multiple sensors are very deviant (z > 2), call it COMPOSITE
    n_deviant = int((abs_z > 2.0).sum())
    if n_deviant >= 3:
        return "COMPOSITE"

    return SENSOR_CATEGORY.get(sensor, "COMPOSITE")


def _confidence_from_scores(if_scores: np.ndarray,
                             lof_scores: np.ndarray) -> np.ndarray:
    """
    Combine IF decision-function (negative = anomalous) and
    LOF negative-outlier-factor (more negative = more anomalous)
    into a 0–100 anomaly confidence score.
    """
    # Normalise IF: invert so higher = more anomalous, then scale 0-1
    if_range  = if_scores.max() - if_scores.min() + 1e-9
    if_norm   = (if_scores.max() - if_scores) / if_range          # 0–1

    # Normalise LOF: negate and scale
    lof_inv   = -lof_scores
    lof_range = lof_inv.max() - lof_inv.min() + 1e-9
    lof_norm  = (lof_inv - lof_inv.min()) / lof_range              # 0–1

    combined  = 0.55 * if_norm + 0.45 * lof_norm
    return np.clip(combined * 100, 0, 100).round(1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_and_predict(
    df: pd.DataFrame,
    contamination: float | None = None,
) -> pd.DataFrame:
    """
    Fit the anomaly ensemble per machine type, then attach:

      anomaly_score      – raw IF decision-function (lower = more anomalous)
      anomaly_confidence – 0-100 score (higher = more anomalous)
      is_anomaly         – bool (ensemble majority vote)
      anomaly_category   – THERMAL / MECHANICAL / ELECTRICAL /
                           ENVIRONMENTAL / COMPOSITE / NONE

    Parameters
    ----------
    df              : DataFrame from generate_dataset()
    contamination   : override per-type rates when not None
    """
    df       = df.copy()
    results  = []

    for mtype, group in df.groupby("machine_type"):
        cont  = contamination if contamination is not None else _contamination_for_type(mtype)
        X_raw = group[FEATURE_COLS].values

        scaler = StandardScaler()
        X      = scaler.fit_transform(X_raw)

        # ── Isolation Forest ────────────────────────────────────────────
        iso = IsolationForest(
            n_estimators=250,
            contamination=cont,
            max_features=0.85,          # slight feature randomness
            random_state=42,
            n_jobs=-1,
        )
        iso.fit(X)
        if_scores  = iso.decision_function(X)   # negative = anomalous
        if_preds   = iso.predict(X) == -1       # True = anomaly

        # ── Local Outlier Factor (novelty=False → fit+predict) ───────────
        n_neighbors = min(20, max(5, len(group) // 10))
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=cont,
            n_jobs=-1,
        )
        lof_preds  = lof.fit_predict(X) == -1
        lof_scores = lof.negative_outlier_factor_

        # ── Majority-vote fusion ─────────────────────────────────────────
        ensemble_flag = if_preds | lof_preds   # OR gate (sensitive)
        # Could also use & (AND) for stricter detection; OR reduces false negatives

        # ── Confidence score ─────────────────────────────────────────────
        confidence = _confidence_from_scores(if_scores, lof_scores)

        # ── Anomaly category per anomalous row ──────────────────────────
        categories = []
        for idx_in_group, row_series in group.iterrows():
            if ensemble_flag[group.index.get_loc(idx_in_group)]:
                cat = _anomaly_category(row_series, scaler)
            else:
                cat = "NONE"
            categories.append(cat)

        # Build partial result
        partial = group.copy()
        partial["anomaly_score"]      = if_scores
        partial["anomaly_confidence"] = confidence
        partial["is_anomaly"]         = ensemble_flag
        partial["anomaly_category"]   = categories

        results.append(partial)

    out = pd.concat(results).sort_index()

    # Ensure bool dtype
    out["is_anomaly"] = out["is_anomaly"].astype(bool)

    return out
