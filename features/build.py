"""Birbaşa çox-üfüqlü (direct multi-horizon) xüsusiyyət cədvəli.

Proqnoz başlanğıcı T = son məlum saatın interval başlanğıcı. Hədəflər t = T+h, h = 1..H.
SIZMAYA QARŞI QAYDA:
  - "başlanğıc" xüsusiyyətləri yalnız ≤ T sərf məlumatından hesablanır və h saat sürüşdürülür;
  - hədəf-indeksli sərf lag-ları yalnız t−24·k (k ≥ 1) istifadə edir; h ≤ 24 olduğundan t−24 ≤ T;
  - hava hədəf saatı üçün: təlimdə faktiki, qiymətləndirmə/canlı rejimdə PROQNOZ cədvəli verilir.
Eyni funksiya həm təlim, həm də canlı proqnoz üçün istifadə olunur (tutarlılıq).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.config import stations
from external.calendar import hourly_calendar_features
from features.profiles import ProfileClusterer
from features.weather_features import weather_features

HOLIDAY_KINDS = ["", "novruz", "ramazan", "qurban", "new_year", "transfer", "other"]

FEATURE_GROUPS: dict[str, list[str]] = {
    # Daxili məlumat: sərf tarixçəsi + saat/gün/il dövrü
    "base": ["station_code", "h", "lag_last", "mean_24", "mean_168", "trend_24", "min_24",
             "q_lag24", "q_lag48", "q_lag168", "same_hour_mean7", "hour", "hour_sin", "hour_cos",
             "dow", "doy_sin", "doy_cos"],
    "weather": ["temp", "app_temp", "humidity", "wind", "radiation", "cloud", "precip", "hdd",
                "t_roll24", "t_roll48", "t_roll72", "hdd_roll24", "hdd_roll72", "cold_snap",
                "wind_chill_load", "temp_now", "t_roll24_now", "d_t24"],
    "calendar": ["is_nonworking", "is_holiday", "holiday_kind_code", "is_working_weekend", "block_len",
                 "block_pos", "long_block", "pre_nonworking", "post_nonworking", "pre_holiday",
                 "post_holiday", "ramadan", "iftar_window", "school_holiday", "special_level",
                 "is_dark", "h_from_sunset", "h_from_sunrise"],
    "profile": ["profile_cluster_prev"],
    "population": ["pop_total", "pop_density", "pop_inner_share", "osm_buildings_total",
                   "osm_apartment_share", "osm_industrial_share", "osm_commercial_share"],
}
CATEGORICAL = ["station_code", "holiday_kind_code", "profile_cluster_prev"]


def station_code(sid: str) -> int:
    return list(stations()).index(sid)


@dataclass
class FeatureContext:
    hourly: pd.DataFrame                              # təmizlənmiş saatlıq data (bütün məntəqələr)
    weather: dict[str, pd.DataFrame]                  # məntəqə → hava cədvəli (ts indeksli)
    hdd_base: dict[str, float]
    static: dict[str, dict] = field(default_factory=dict)
    clusterer: ProfileClusterer | None = None
    _clusters: pd.DataFrame | None = None

    def clusters(self) -> pd.DataFrame | None:
        if self.clusterer is None:
            return None
        if self._clusters is None:
            self._clusters = self.clusterer.transform(self.hourly)
        return self._clusters


def _origin_features(q: pd.Series, wf: pd.DataFrame) -> pd.DataFrame:
    o = pd.DataFrame(index=q.index)
    o["lag_last"] = q
    o["mean_24"] = q.rolling(24, min_periods=12).mean()
    o["mean_168"] = q.rolling(168, min_periods=84).mean()
    prev24 = q.shift(24).rolling(24, min_periods=12).mean()
    o["trend_24"] = o["mean_24"] / prev24.replace(0, np.nan)
    o["min_24"] = q.rolling(24, min_periods=12).min()
    o["temp_now"] = wf["temp"].reindex(q.index)
    o["t_roll24_now"] = wf["t_roll24"].reindex(q.index)
    return o


def _target_features(ctx: FeatureContext, sid: str, g: pd.DataFrame) -> pd.DataFrame:
    q = g["q_filled"]
    t = pd.DataFrame(index=g.index)
    t["q_lag24"] = q.shift(24)
    t["q_lag48"] = q.shift(48)
    t["q_lag168"] = q.shift(168)
    t["same_hour_mean7"] = sum(q.shift(24 * k) for k in range(1, 8)) / 7
    cal = hourly_calendar_features(g.index, sid)
    cal["holiday_kind_code"] = cal.pop("holiday_kind").map({k: i for i, k in enumerate(HOLIDAY_KINDS)}).fillna(6)
    t = t.join(cal)
    for k, v in (ctx.static.get(sid) or {}).items():
        if k in FEATURE_GROUPS["population"]:
            t[k] = np.nan if v is None else float(v)  # None → object tipi LightGBM-i sındırır
    t["station_code"] = station_code(sid)
    return t


def make_rows(ctx: FeatureContext, sid: str, horizon: int = 24, origin_step: int = 1,
              origins: pd.DatetimeIndex | None = None, until: pd.Timestamp | None = None) -> pd.DataFrame:
    """Bir məntəqə üçün (origin, h) sətirləri.

    origins verilərsə yalnız həmin başlanğıclar; əks halda T.hour % origin_step == 0 olan bütün T.
    until: canlı proqnoz üçün şəbəkəni bu saata qədər uzat (gələcək saatlar, q=NaN).
    """
    g = ctx.hourly[ctx.hourly["station"] == sid].set_index("ts").sort_index()
    end = g.index.max()
    if until is not None and until > end:
        g = g.reindex(pd.date_range(g.index.min(), until, freq="h", name="ts"))
        g["station"] = sid
    w = ctx.weather[sid].reindex(g.index)
    wf = weather_features(w, ctx.hdd_base.get(sid, 18.0))

    q = g["q_filled"]
    known = q.copy()
    if origins is not None:
        # canlı rejim: başlanğıcdan sonrakı sərf məlum deyil
        known = known.where(known.index <= origins.max())
    o = _origin_features(known, wf)
    tf = _target_features(ctx, sid, g.assign(q_filled=known)).join(wf)

    cl = ctx.clusters()
    if cl is not None:
        c = cl[cl["station"] == sid].set_index("date")["profile_cluster"]
        # başlanğıc gününün bir gün əvvəli (həmişə tam gündür)
        prev_day = (o.index.normalize() - pd.Timedelta("1D"))
        o["profile_cluster_prev"] = c.reindex(prev_day).to_numpy()

    y = np.log1p(g["q_clean"])
    parts = []
    for h in range(1, horizon + 1):
        oh = o.shift(h)  # t indeksində dəyər = o[t − h]
        df = tf.join(oh)
        df["h"] = h
        df["origin"] = df.index - pd.Timedelta(hours=h)
        if origins is not None:
            df = df[df["origin"].isin(origins)]
        else:
            df = df[df["origin"].dt.hour % origin_step == 0]
        parts.append(df)
    out = pd.concat(parts)
    out["d_t24"] = out["t_roll24"] - out["t_roll24_now"]
    out["y"] = y.reindex(out.index).to_numpy()
    out["q_actual"] = g["q_clean"].reindex(out.index).to_numpy()
    out["train_ok"] = g["train_ok"].reindex(out.index).fillna(False).astype(bool).to_numpy()
    out["event"] = g["f_event"].reindex(out.index).fillna(False).astype(bool).to_numpy()
    out["station"] = sid
    out = out.rename_axis("ts").reset_index()
    # Başlanğıcın özü data daxilində olmalıdır
    out = out[out["origin"] >= g.index.min() + pd.Timedelta("8D")]
    return out


def make_dataset(ctx: FeatureContext, horizon: int = 24, origin_step: int = 3) -> pd.DataFrame:
    df = pd.concat([make_rows(ctx, sid, horizon, origin_step) for sid in stations()], ignore_index=True)
    num = df.select_dtypes("float64").columns
    df[num] = df[num].astype("float32")
    return df


def feature_list(groups: list[str]) -> list[str]:
    cols: list[str] = []
    for gname in groups:
        cols += FEATURE_GROUPS[gname]
    return cols
