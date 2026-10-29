"""Hava məlumatları — Open-Meteo (pulsuz, açarsız; CC BY 4.0).

Mənbələr:
  archive        — faktiki (reanaliz, ERA5 əsaslı) saatlıq hava: təlim üçün
  previous_day1  — Previous Runs API: hədəf saat üçün ~1 gün ƏVVƏL verilmiş proqnoz (2021-03-dən).
                   Məlumat sızmasının qarşısını almaq üçün test/backtest və demo-replay rejimində istifadə olunur.
  forecast       — Forecast API: canlı rejim

Vaxt düzülüşü: Open-Meteo dəyərləri HH:00 anına aiddir (temperatur və s. ani, radiasiya/yağıntı
əvvəlki saatın cəmi/ortası). Bizim interval [t, t+1h) olduğundan:
  ani dəyişənlər → (v(t) + v(t+1)) / 2;   axın dəyişənləri → v(t+1).

İşə salma: python -m external.weather   (bütün məntəqələr üçün cədvəlləri yaradır/yeniləyir)
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from core.config import load_config, stations
from core.storage import has_table, read_table, write_table
from external.common import cache_dir, http_get, set_status

log = logging.getLogger(__name__)
FLUX_VARS = {"shortwave_radiation", "precipitation"}
SOURCES = ("archive", "previous_day1", "forecast")


def _wcfg() -> dict:
    return load_config()["weather"]


def _request(source: str, lat: float, lon: float, start: str | None, end: str | None) -> pd.DataFrame:
    w = _wcfg()
    tz = load_config()["project"]["timezone"]
    vars_ = w["variables"]
    params = {"latitude": lat, "longitude": lon, "timezone": tz}
    if source == "archive":
        url = w["base_urls"]["archive"]
        params |= {"hourly": ",".join(vars_), "start_date": start, "end_date": end}
    elif source == "previous_day1":
        url = w["base_urls"]["previous_runs"]
        params |= {"hourly": ",".join(f"{v}_previous_day1" for v in vars_), "start_date": start, "end_date": end}
    elif source == "forecast":
        url = w["base_urls"]["forecast"]
        params |= {"hourly": ",".join(vars_), "past_days": 3, "forecast_days": 3}
    else:
        raise ValueError(source)
    js = http_get(url, params, source=f"open-meteo:{source}").json()
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").rename(columns=lambda c: c.replace("_previous_day1", ""))
    return df.reindex(columns=vars_).astype(float)


def point_weather(source: str, lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp,
                  offline: bool = False) -> pd.DataFrame:
    """Bir nöqtə üçün saatlıq hava; illər üzrə Parquet keşi ilə."""
    d = cache_dir(f"weather/{source}")
    key = f"{lat:.4f}_{lon:.4f}"
    if source == "forecast":
        f = d / f"{key}.parquet"
        try:
            if offline:
                raise ConnectionError("offline rejim")
            df = _request(source, lat, lon, None, None)
            df.to_parquet(f)
            set_status("open-meteo:forecast", True)
        except Exception as e:  # API əlçatmazdır → keş
            set_status("open-meteo:forecast", False, f"Forecast API əlçatmazdır, keş istifadə olunur: {e}")
            if not f.exists():
                raise
            df = pd.read_parquet(f)
        return df.loc[start:end]

    parts = []
    today = pd.Timestamp(datetime.now().date())
    for y in range(start.year, end.year + 1):
        f = d / f"{key}_{y}.parquet"
        y0, y1 = pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31, 23)
        fresh_needed = y1 >= today - timedelta(days=7)  # cari il: tez-tez yenilənir
        if f.exists() and (not fresh_needed or offline or
                           datetime.fromtimestamp(f.stat().st_mtime) > datetime.now() - timedelta(hours=12)):
            parts.append(pd.read_parquet(f))
            continue
        s = max(y0, pd.Timestamp(_wcfg()["history_start"] if source == "archive" else _wcfg()["previous_runs_start"]))
        e = min(y1, today - timedelta(days=1 if source == "previous_day1" else 6))
        if s > e:
            continue
        try:
            if offline:
                raise ConnectionError("offline rejim")
            df = _request(source, lat, lon, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"))
            df.to_parquet(f)
            set_status(f"open-meteo:{source}", True)
            parts.append(df)
        except Exception as ex:
            set_status(f"open-meteo:{source}", False, f"{y} ili alınmadı: {ex}")
            if f.exists():
                parts.append(pd.read_parquet(f))
            else:
                log.error("%s %s %s: keş də yoxdur", source, key, y)
    if not parts:
        return pd.DataFrame(columns=_wcfg()["variables"], dtype=float)
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.loc[start:end]


def align_to_interval(df: pd.DataFrame) -> pd.DataFrame:
    """HH:00 ani dəyərləri [t, t+1h) intervalına çevirir."""
    nxt = df.shift(-1, freq="h").reindex(df.index)
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        out[c] = nxt[c] if c in FLUX_VARS else (df[c] + nxt[c]) / 2
    return out.dropna(how="all")


def station_weather(station_id: str, source: str, start, end, offline: bool = False) -> pd.DataFrame:
    """Məntəqənin xidmət ərazisi üzrə nöqtələrin çəkili ortalaması, interval başlanğıcı ilə indekslənmiş."""
    pts = stations()[station_id]["weather_points"]
    start, end = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta("1h")
    acc, wsum = None, None
    for p in pts:
        df = point_weather(source, p["lat"], p["lon"], start, end, offline=offline)
        wt = df.notna().astype(float) * p["weight"]
        val = df.fillna(0) * p["weight"]
        acc = val if acc is None else acc.add(val, fill_value=0)
        wsum = wt if wsum is None else wsum.add(wt, fill_value=0)
    out = acc / wsum.replace(0, np.nan)
    return align_to_interval(out).loc[start:end - pd.Timedelta("1h")]


TABLES = (("archive", "weather_actual"), ("previous_day1", "weather_fc_d1"))


def build_tables(start: str = "2018-01-01", end: str | None = None, offline: bool = False) -> dict:
    """Bütün məntəqələr üçün faktiki və 'əvvəlki gün proqnozu' cədvəlləri (saxlama qatına)."""
    end = end or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    res = {}
    for source, name in TABLES:
        frames = []
        for sid in stations():
            df = station_weather(sid, source, start, end, offline=offline)
            frames.append(df.assign(station=sid).rename_axis("ts").reset_index())
        out = pd.concat(frames, ignore_index=True)
        write_table(name, out)
        res[source] = out
    return res


def update_tables(end: str, offline: bool = False) -> None:
    """Mövcud cədvəlləri yalnız çatışmayan dövr üçün uzadır (bulud: lokal keş yoxdur → az API çağırışı)."""
    for source, name in TABLES:
        if not has_table(name):
            build_tables(end=end, offline=offline)
            return
        cur = read_table(name)
        cur["ts"] = pd.to_datetime(cur["ts"])
        start = cur["ts"].max() - pd.Timedelta("3D")
        if start >= pd.Timestamp(end):
            continue
        frames = [station_weather(sid, source, start, end, offline=offline)
                  .assign(station=sid).rename_axis("ts").reset_index() for sid in stations()]
        new = pd.concat([cur[cur.ts < start], *frames], ignore_index=True)
        write_table(name, new.drop_duplicates(["station", "ts"], keep="last"))


def validate_against_meteostat(actual: pd.DataFrame) -> dict:
    """Open-Meteo faktiki temperaturunu verilmiş Meteostat (Sumqayıt, gündəlik) faylı ilə müqayisə."""
    from core.config import path
    f = path("raw_dir") / "sumqayit hava 2018-2024.csv"
    if not f.exists():
        return {}
    ms = pd.read_csv(f, encoding="utf-8-sig", parse_dates=["date"]).set_index("date")["tavg"]
    om = actual[actual.station == "sumqayit4"].set_index("ts")["temperature_2m"].resample("D").mean()
    j = pd.concat([ms.rename("meteostat"), om.rename("open_meteo")], axis=1).dropna()
    return {"days": len(j), "corr": round(float(j.corr().iloc[0, 1]), 4),
            "bias_c": round(float((j.open_meteo - j.meteostat).mean()), 3),
            "mae_c": round(float((j.open_meteo - j.meteostat).abs().mean()), 3)}


def forecast_skill(actual: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """'Əvvəlki gün proqnozu'nun faktiki havaya nisbətən xətası (hesabat üçün)."""
    j = actual.merge(fc, on=["station", "ts"], suffixes=("_act", "_fc"))
    rows = []
    for v in _wcfg()["variables"]:
        a, b = j[f"{v}_act"], j[f"{v}_fc"]
        m = a.notna() & b.notna()
        if m.sum():
            rows.append({"variable": v, "n": int(m.sum()), "mae": float((a[m] - b[m]).abs().mean()),
                         "bias": float((b[m] - a[m]).mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    end = sys.argv[1] if len(sys.argv) > 1 else "2024-08-31"
    t = build_tables(end=end)
    print("Meteostat müqayisəsi:", validate_against_meteostat(t["archive"]))
    print(forecast_skill(t["archive"], t["previous_day1"]).round(3).to_string())
