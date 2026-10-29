"""Hava əsaslı törəmə xüsusiyyətlər: HDD (kalibrə olunmuş baza), istilik ətaləti, kəskin soyuma."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER_COLS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "wind_speed_10m",
                "shortwave_radiation", "cloud_cover", "precipitation"]


def calibrate_hdd_base(q_hourly: pd.Series, temp_hourly: pd.Series, grid=np.arange(12, 24.01, 0.5)) -> dict:
    """Gündəlik sərf = a + b·max(0, Tb − T_gün) modelində ən yaxşı Tb (R² üzrə)."""
    d = pd.concat([q_hourly.resample("D").sum(min_count=20).rename("q"),
                   temp_hourly.resample("D").mean().rename("t")], axis=1).dropna()
    best = {"tb": None, "r2": -1.0}
    for tb in grid:
        x = np.clip(tb - d["t"].to_numpy(), 0, None)
        if x.std() == 0:
            continue
        r2 = float(np.corrcoef(x, d["q"])[0, 1] ** 2)
        if r2 > best["r2"]:
            best = {"tb": float(tb), "r2": round(r2, 4)}
    best["days"] = int(len(d))
    return best


def weather_features(w: pd.DataFrame, hdd_base: float) -> pd.DataFrame:
    """w: interval başlanğıcı ilə indekslənmiş saatlıq hava (bir məntəqə)."""
    t = w["temperature_2m"]
    out = pd.DataFrame(index=w.index)
    out["temp"] = t
    out["app_temp"] = w["apparent_temperature"]
    out["humidity"] = w["relative_humidity_2m"]
    out["wind"] = w["wind_speed_10m"]
    out["radiation"] = w["shortwave_radiation"]
    out["cloud"] = w["cloud_cover"]
    out["precip"] = w["precipitation"]
    out["hdd"] = np.clip(hdd_base - t, 0, None)
    # Binaların istilik ətaləti: son 24/48/72 saatın ortalaması
    for n in (24, 48, 72):
        out[f"t_roll{n}"] = t.rolling(n, min_periods=n // 2).mean()
    out["hdd_roll24"] = np.clip(hdd_base - out["t_roll24"], 0, None)
    out["hdd_roll72"] = np.clip(hdd_base - out["t_roll72"], 0, None)
    # Kəskin soyuma: son 24 saatın ortası 3 gün əvvəlkindən nə qədər aşağıdır
    out["cold_snap"] = out["t_roll24"] - out["t_roll24"].shift(72)
    out["wind_chill_load"] = out["wind"] * out["hdd"]
    return out
