"""Təqvim: bayramlar, iş/qeyri-iş günləri, köçürmələr, məktəb tətilləri, Ramazan, xüsusi rejimlər,
gün çıxma/batma (istehlakçıların evdə olma vaxtı üçün proksi göstəricilər).

Mənbələr:
  - `holidays` Python kitabxanası (AZ dəstəklənir; köçürülmüş günlər "... ilə əvəz edilmişdir" adı ilə gəlir)
  - config/calendar_overrides.yaml (əl ilə düzəlişlər)
  - Nager.Date API Azərbaycanı dəstəkləmir (yoxlanıb: HTTP 204) → ehtiyat mənbə kimi istifadə edilmir
  - `astral` — gün çıxma/batma
"""
from __future__ import annotations

import re
from datetime import date
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd
from astral import Observer
from astral.sun import sun

from core.config import load_config, load_yaml, stations

TRANSFER_RE = re.compile(r"(\d\d\.\d\d\.\d{4}) ilə əvəz")
KINDS = [("novruz", "Novruz"), ("ramazan", "Ramazan"), ("qurban", "Qurban"), ("new_year", "Yeni il")]


def _kind(name: str) -> str:
    for k, key in KINDS:
        if key.lower() in name.lower():
            return k
    if "İstirahət günü" in name:
        return "transfer"
    return "other"


@lru_cache(maxsize=8)
def build_calendar(start: str, end: str) -> pd.DataFrame:
    """Gündəlik təqvim cədvəli [start, end]."""
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    years = range(days.min().year - 1, days.max().year + 2)
    lib = holidays.country_holidays("AZ", years=years)
    ov = load_yaml("calendar_overrides")

    hol, working_weekend, eid_ramazan = {}, set(), []
    for d, name in lib.items():
        hol[pd.Timestamp(d)] = name
        m = TRANSFER_RE.search(name)
        if m:
            working_weekend.add(pd.Timestamp(pd.to_datetime(m.group(1), format="%d.%m.%Y")))
        if "Ramazan" in name and "müşahidə" not in name:
            eid_ramazan.append(pd.Timestamp(d))
    for x in ov.get("extra_nonworking_days", []) or []:
        hol[pd.Timestamp(x["date"])] = x.get("name", "Əlavə qeyri-iş günü")
    for x in ov.get("remove_nonworking_days", []) or []:
        hol.pop(pd.Timestamp(x["date"] if isinstance(x, dict) else x), None)
    for x in ov.get("working_weekend_days", []) or []:
        working_weekend.add(pd.Timestamp(x["date"]))

    cal = pd.DataFrame(index=pd.date_range(days.min() - pd.Timedelta("20D"), days.max() + pd.Timedelta("20D")))
    cal["dow"] = cal.index.dayofweek
    cal["is_weekend"] = cal["dow"] >= 5
    names = pd.Series(hol)
    cal["holiday_name"] = names.reindex(cal.index).fillna("")
    cal["is_holiday"] = cal["holiday_name"] != ""
    cal["holiday_kind"] = cal["holiday_name"].map(lambda n: _kind(n) if n else "")
    cal["is_working_weekend"] = cal.index.isin(list(working_weekend))
    cal["is_nonworking"] = (cal["is_weekend"] & ~cal["is_working_weekend"]) | cal["is_holiday"]

    # Qeyri-iş blokları (bayram + həftəsonu birləşməsi)
    nw = cal["is_nonworking"]
    grp = nw.ne(nw.shift()).cumsum()
    cal["block_len"] = nw.groupby(grp).transform("size").where(nw, 0)
    cal["block_pos"] = nw.groupby(grp).cumcount().add(1).where(nw, 0)
    cal["long_block"] = cal["block_len"] >= 3
    cal["pre_nonworking"] = ~nw & nw.shift(-1, fill_value=False)
    cal["post_nonworking"] = ~nw & nw.shift(1, fill_value=False)
    hd = cal["is_holiday"] & (cal["holiday_kind"] != "transfer")
    cal["pre_holiday"] = ~hd & hd.shift(-1, fill_value=False)
    cal["post_holiday"] = ~hd & hd.shift(1, fill_value=False)

    # Ramazan ayı ≈ Ramazan bayramından əvvəlki 30 gün (hicri təqvim)
    cal["ramadan"] = False
    for e in eid_ramazan:
        cal.loc[e - pd.Timedelta("30D"): e - pd.Timedelta("1D"), "ramadan"] = True

    # Məktəb tətilləri
    cal["school_holiday"] = False
    sh = ov.get("school_holidays", {}) or {}
    explicit = sh.get("periods", []) or []
    explicit_years = {pd.Timestamp(p["start"]).year for p in explicit}
    for y in range(cal.index.min().year - 1, cal.index.max().year + 1):
        if y in explicit_years:
            continue
        for r in sh.get("default_rules", []) or []:
            s = pd.Timestamp(f"{y}-{r['start']}")
            e = pd.Timestamp(f"{y}-{r['end']}")
            if e < s:
                e = pd.Timestamp(f"{y + 1}-{r['end']}")
            cal.loc[s:e, "school_holiday"] = True
    for p in explicit:
        cal.loc[pd.Timestamp(p["start"]):pd.Timestamp(p["end"]), "school_holiday"] = True

    # Xüsusi rejimlər (karantin və s.): maksimal səviyyə
    cal["special_level"] = 0
    for p in ov.get("special_periods", []) or []:
        sl = slice(pd.Timestamp(p["start"]), pd.Timestamp(p["end"]))
        cal.loc[sl, "special_level"] = np.maximum(cal.loc[sl, "special_level"], int(p.get("level", 1)))

    return cal.loc[days.min():days.max()].rename_axis("date")


@lru_cache(maxsize=16)
def sun_table(station_id: str, start: str, end: str) -> pd.DataFrame:
    a = stations()[station_id]["area"]
    obs = Observer(latitude=a["lat"], longitude=a["lon"])
    tz = load_config()["project"]["timezone"]
    import zoneinfo
    tzinfo = zoneinfo.ZoneInfo(tz)
    rows = []
    for d in pd.date_range(start, end, freq="D"):
        s = sun(obs, date=d.date(), tzinfo=tzinfo)
        rise = s["sunrise"].hour + s["sunrise"].minute / 60
        sset = s["sunset"].hour + s["sunset"].minute / 60
        rows.append((d, rise, sset))
    return pd.DataFrame(rows, columns=["date", "sunrise_h", "sunset_h"]).set_index("date")


def hourly_calendar_features(ts: pd.DatetimeIndex, station_id: str) -> pd.DataFrame:
    """Saatlıq təqvim/günəş xüsusiyyətləri (ts = interval başlanğıcı)."""
    start, end = ts.min().strftime("%Y-%m-%d"), ts.max().strftime("%Y-%m-%d")
    cal = build_calendar(start, end)
    sn = sun_table(station_id, start, end)
    d = ts.normalize()
    out = pd.DataFrame(index=ts)
    hour = ts.hour + 0.5  # intervalın ortası
    out["hour"] = ts.hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    doy = ts.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    c = cal.reindex(d)
    for col in ["dow", "is_nonworking", "is_holiday", "is_working_weekend", "block_len", "block_pos",
                "long_block", "pre_nonworking", "post_nonworking", "pre_holiday", "post_holiday",
                "ramadan", "school_holiday", "special_level"]:
        out[col] = c[col].to_numpy()
    out["holiday_kind"] = c["holiday_kind"].to_numpy()
    s = sn.reindex(d)
    rise, sset = s["sunrise_h"].to_numpy(), s["sunset_h"].to_numpy()
    out["is_dark"] = (hour < rise) | (hour > sset)
    out["h_from_sunset"] = hour - sset          # mənfi: qüruba qədər; müsbət: qürubdan sonra
    out["h_from_sunrise"] = hour - rise
    # Ramazan: iftar (qürub) ətrafında yemək bişirmə piki
    out["iftar_window"] = out["ramadan"].astype(bool) & (np.abs(out["h_from_sunset"]) <= 1.5)
    bool_cols = out.select_dtypes(bool).columns
    out[bool_cols] = out[bool_cols].astype(np.int8)
    return out
