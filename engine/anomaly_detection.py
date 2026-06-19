"""
ForgeMind AI – Anomaly Detection Engine
Uses Isolation Forest to flag abnormal machine behaviour.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["temperature", "vibration", "pressure", "power_kw"]
CONTAMINATION = 0.12          # expected anomaly fraction (~12 %)


def train_and_predict(df: pd.DataFrame,
                      contamination: float = CONTAMINATION) -> pd.DataFrame:
    """
    Fit an Isolation Forest on FEATURE_COLS and attach anomaly flags.

    Returns a copy of `df` with two new columns:
      - anomaly_score : raw decision-function score (lower = more anomalous)
      - is_anomaly    : bool flag (True = anomalous)
    """
    df = df.copy()

    scaler  = StandardScaler()
    X       = scaler.fit_transform(df[FEATURE_COLS])

    model   = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    df["anomaly_score"] = model.decision_function(X)   # negative = anomalous
    df["is_anomaly"]    = model.predict(X) == -1       # True when predicted as anomaly

    return df
