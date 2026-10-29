import os
import sys
from pathlib import Path

os.environ["DATABASE_URL"] = ""  # testlər heç vaxt real bazaya yazmır (dotenv mövcud dəyişəni əvəz etmir)
os.environ.pop("API_TOKEN", None)

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def synthetic_hourly():
    """İki həftəlik sintetik 'təmizlənmiş' saatlıq sıra (turkan məntəqəsi)."""
    ts = pd.date_range("2023-01-01", periods=24 * 21, freq="h")
    q = 3 + np.sin(2 * np.pi * ts.hour / 24) + 0.01 * np.arange(len(ts))
    df = pd.DataFrame({"station": "turkan", "ts": ts, "q_clean": q, "q_filled": q, "q_raw": q,
                       "train_ok": True, "f_event": False, "p_kpa": 500.0, "f_dp_saturated": False})
    return df


@pytest.fixture
def synthetic_weather():
    ts = pd.date_range("2022-12-25", periods=24 * 40, freq="h")
    cols = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "wind_speed_10m",
            "shortwave_radiation", "cloud_cover", "precipitation"]
    w = pd.DataFrame({c: 5.0 for c in cols}, index=ts)
    w["temperature_2m"] = 5 + 3 * np.sin(2 * np.pi * ts.hour / 24)
    return w
