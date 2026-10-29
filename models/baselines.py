"""Sadə baseline-lar. Hamısı xüsusiyyət cədvəlindəki (sızmasız) lag sütunlarından hesablanır."""
from __future__ import annotations

import pandas as pd

BASELINES = {
    "naive (son saat)": "lag_last",
    "seasonal naive 24s": "q_lag24",
    "seasonal naive 168s": "q_lag168",
    "MA: eyni saat, 7 gün": "same_hour_mean7",
}


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, col in BASELINES.items():
        df[f"bl_{col}"] = df[col]
    return df


def baseline_cols() -> dict[str, str]:
    return {name: f"bl_{col}" for name, col in BASELINES.items()}
