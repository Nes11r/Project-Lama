"""Xüsusiyyət kontekstinin (data + hava + kalibrasiya + klasterlər) yığılması."""
from __future__ import annotations

import logging

import pandas as pd

from core.config import load_config, stations
from core.storage import read_table
from data.pipeline import load_hourly
from external.calendar import build_calendar
from features.build import FeatureContext
from features.profiles import ProfileClusterer
from features.weather_features import WEATHER_COLS, calibrate_hdd_base

log = logging.getLogger(__name__)


def _read(name: str, start=None, end=None) -> pd.DataFrame:
    df = read_table(name, start=start, end=end)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def weather_table(kind: str, start=None, end=None) -> dict[str, pd.DataFrame]:
    """kind:
    actual   — faktiki hava (təlim; optimist qiymətləndirmə)
    forecast — 1 gün əvvəlki proqnoz; proqnozu olmayan dəyişən/dövr faktiki ilə tamamlanır
               (temperatur 2021-03-dən, digərləri 2024-01-dən proqnozdur)
    start/end — yalnız bu aralıq (API: "indi" ətrafındakı pəncərə)
    """
    act = _read("weather_actual", start, end)
    fc = _read("weather_fc_d1", start, end) if kind != "actual" else None
    out = {}
    for sid in stations():
        a = act[act.station == sid].set_index("ts")[WEATHER_COLS]
        if kind == "actual":
            out[sid] = a
            continue
        f = fc[fc.station == sid].set_index("ts")[WEATHER_COLS].reindex(a.index)
        out[sid] = f.combine_first(a)
    return out


def forecast_coverage() -> pd.DataFrame:
    """Hansı dəyişən hansı dövrdən real proqnozdur."""
    fc = _read("weather_fc_d1")
    rows = []
    for v in WEATHER_COLS:
        s = fc.loc[fc[v].notna(), "ts"]
        rows.append({"variable": v, "forecast_from": s.min(), "forecast_to": s.max()})
    return pd.DataFrame(rows)


def build_context(weather_kind: str = "actual", cutoff: pd.Timestamp | None = None,
                  with_clusters: bool = True, with_static: bool = True,
                  hourly: pd.DataFrame | None = None) -> FeatureContext:
    hourly = load_hourly() if hourly is None else hourly
    cutoff = cutoff or hourly["ts"].max() + pd.Timedelta("1h")
    weather = weather_table(weather_kind)
    act = weather if weather_kind == "actual" else weather_table("actual")

    hdd = {}
    for sid in stations():
        g = hourly[(hourly.station == sid) & (hourly.ts < cutoff) & hourly.train_ok].set_index("ts")["q_clean"]
        cal = calibrate_hdd_base(g, act[sid]["temperature_2m"].reindex(g.index))
        hdd[sid] = cal["tb"] or load_config()["weather"]["hdd_base_default"]

    static = {}
    if with_static:
        from external.osm import static_features
        static = static_features()

    clusterer = None
    if with_clusters:
        cal = build_calendar(str(hourly.ts.min().date()), str(hourly.ts.max().date()))
        clusterer = ProfileClusterer(k=6).fit(hourly, cutoff, cal)
    return FeatureContext(hourly=hourly, weather=weather, hdd_base=hdd, static=static, clusterer=clusterer)
