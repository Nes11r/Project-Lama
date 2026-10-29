"""Bir məntəqənin saatlıq sırasının təmizlənməsi.

Hər addım sənədləşdirilib (docs/data_pipeline.md) və hesabata yazılır. Çıxış sütunları:
  q_raw      — ROC-dan gələn saatlıq sərf (min m³), dəyişməmiş
  q_clean    — etibarlı dəyərlər (+ "Cəmi" ilə bərpa olunanlar); etibarsızlar NaN
  q_filled   — xüsusiyyətlər (lag-lar) üçün boşluqsuz sıra
  train_ok   — modelin təlimi/qiymətləndirilməsi üçün yararlı saat
  f_*        — keyfiyyət bayraqları
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ["sg", "dp_kpa", "p_kpa", "t_c", "q_hour"]


def _runs(mask: pd.Series) -> pd.Series:
    """Hər True elementi üçün aid olduğu ardıcıl True seqmentinin uzunluğu."""
    grp = mask.ne(mask.shift()).cumsum()
    size = mask.groupby(grp).transform("size")
    return size.where(mask, 0)


def robust_z_same_hour(q: pd.Series, days: int = 15) -> tuple[pd.Series, pd.Series]:
    """Hər saatı günün eyni saatının mərkəzləşdirilmiş `days` günlük median/MAD-ı ilə müqayisə edir.

    Qaytarır: (z, median)."""
    hour = q.index.hour
    med = q.groupby(hour).transform(lambda s: s.rolling(days, center=True, min_periods=5).median())
    dev = (q - med).abs()
    mad = dev.groupby(hour).transform(lambda s: s.rolling(days, center=True, min_periods=5).median())
    # MAD çox kiçik olduqda (stabil yay rejimi) mənasız böyük z-lərin qarşısını al
    floor = 0.05 * med.abs()
    return (q - med) / (1.4826 * np.maximum(mad, floor)).replace(0, np.nan), med


def _dedupe(rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows = rows.copy()
    # Eyni saat üçün: sıfır/boş sətir yox, real ölçmə; bərabərdirsə sonrakı fayl
    rows["_score"] = (rows["q_hour"].fillna(0) > 0).astype(int) + (rows["p_kpa"].fillna(0) > 0).astype(int)
    n0 = len(rows)
    rows = rows.sort_values(["ts", "_score", "file_order"]).drop_duplicates("ts", keep="last")
    return rows.drop(columns="_score"), n0 - len(rows)


def reconstruct_from_totals(q: pd.Series, missing: pd.Series, totals: pd.DataFrame,
                            max_missing: int, bounds: tuple[float, float]) -> tuple[pd.Series, pd.Series]:
    """İxracda itən saatları ROC gündəlik cəmi ilə bərpa edir.

    itən_cəm = Cəmi − Σ(mövcud saatlar); itən saatlar arasında interpolasiya formasına uyğun paylanır.
    Nəticə interpolasiyaya nisbətən `bounds` daxilində deyilsə, bərpa edilmir.
    """
    q = q.copy()
    flag = pd.Series(False, index=q.index)
    if totals.empty:
        return q, flag
    shape = q.interpolate(method="time", limit_area="inside")
    day = q.index.normalize()
    tot = totals.set_index("gas_day")["total"]
    df = pd.DataFrame({"q": q, "miss": missing, "shape": shape, "day": day})
    for d, g in df[df["day"].isin(tot.index)].groupby("day"):
        k = int(g["miss"].sum())
        if k == 0 or k > max_missing or len(g) != 24:
            continue
        present = g.loc[~g["miss"], "q"]
        if present.isna().any():
            continue
        residual = tot[d] - present.sum()
        sh = g.loc[g["miss"], "shape"]
        if sh.isna().any() or sh.sum() <= 0:
            continue
        ratio = residual / sh.sum()
        if bounds[0] <= ratio <= bounds[1]:
            q.loc[sh.index] = sh * ratio
            flag.loc[sh.index] = True
    return q, flag


def clean_station(station_id: str, rows: pd.DataFrame, totals: pd.DataFrame,
                  ccfg: dict, events: list[dict]) -> tuple[pd.DataFrame, dict]:
    rep: dict = {"station": station_id, "raw_rows": int(len(rows)), "daily_totals": int(len(totals))}

    # 1. Tam saata düşməyən damğalar (logger yenidən başlaması → natamam saat)
    off = ~rows["on_hour"]
    rep["dropped_off_hour_stamps"] = int(off.sum())
    rows = rows[~off]

    # 2. Təkrarlar
    rows, ndup = _dedupe(rows)
    rep["dropped_duplicates"] = ndup

    # 3. Tam saatlıq şəbəkə
    idx = pd.date_range(rows["ts"].min(), rows["ts"].max(), freq="h", name="ts")
    df = rows.set_index("ts")[CHANNELS].reindex(idx)
    df["f_missing_raw"] = df["q_hour"].isna()
    rep["hours_total"] = int(len(df))
    rep["hours_missing_raw"] = int(df["f_missing_raw"].sum())
    df["q_raw"] = df["q_hour"]

    # 4. Fiziki hüdud yoxlaması (sensor xətaları)
    p_raw = df["p_kpa"].copy()
    rep["out_of_range"] = {}
    for c, (lo, hi) in ccfg["valid_ranges"].items():
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= df[c] < lo
        if hi is not None:
            bad |= df[c] > hi
        df[f"f_bad_{c}"] = bad
        rep["out_of_range"][c] = int(bad.sum())
        df.loc[bad, c] = np.nan

    # 5. Xəttin boşaldılması (mütləq təzyiq ≈ atmosfer) və ΔP saturasiyası
    df["f_depressurized"] = p_raw < ccfg["depressurized_p_kpa"]
    df["f_dp_saturated"] = df["dp_kpa"] >= ccfg["dp_saturation_kpa"]
    rep["depressurized_hours"] = int(df["f_depressurized"].sum())
    rep["dp_saturated_hours"] = int(df["f_dp_saturated"].sum())

    q = df["q_hour"]

    # 6. Donmuş sensor: ≥N saat eyni dəyər
    same = q.notna() & q.eq(q.shift())
    same = same | same.shift(-1, fill_value=False)
    flat = _runs(same) >= ccfg["flatline_hours"]
    df["f_outage"] = flat & (q <= 1e-6) & df["f_depressurized"].rolling(48, center=True, min_periods=1).max().astype(bool)
    df["f_flatline"] = flat & ~df["f_outage"]
    rep["flatline_hours"] = int(df["f_flatline"].sum())
    rep["outage_hours"] = int(df["f_outage"].sum())
    q = q.mask(df["f_flatline"])

    # 7. Kəskin sıçrayışlar: robust z, günün EYNİ SAATI üzrə ±N/2 günlük median (gündəlik dövr nəzərə alınır)
    z, med = robust_z_same_hour(q, days=max(3, ccfg["spike_window_h"] // 24 * 2 + 1))
    lo, hi = ccfg["spike_ratio_bounds"]  # statistik VƏ kəmiyyətcə əhəmiyyətli sapma
    extreme = (z.abs() > ccfg["spike_z"]) & ((q < lo * med) | (q > hi * med))
    run = _runs(extreme)
    df["f_spike"] = extreme & (run <= ccfg["spike_max_run_h"])
    df["f_anomaly_long"] = extreme & (run > ccfg["spike_max_run_h"])
    rep["spike_hours"] = int(df["f_spike"].sum())
    rep["long_anomaly_hours"] = int(df["f_anomaly_long"].sum())
    q = q.mask(df["f_spike"])

    # 8. İtən saatların "Cəmi" ilə bərpası
    q, rec = reconstruct_from_totals(q, df["f_missing_raw"], totals,
                                     ccfg["reconstruct_max_missing_h"], tuple(ccfg["reconstruct_ratio_bounds"]))
    df["f_reconstructed"] = rec
    rep["reconstructed_hours"] = int(rec.sum())
    df["q_clean"] = q

    # 9. Hadisələr (planlı təmir və s.)
    df["f_event"] = False
    df["f_event_excluded"] = False
    for ev in events:
        if ev.get("station") not in (station_id, "all"):
            continue
        s = pd.Timestamp(ev["start"])  # divar saatı = interval başlanğıcı
        e = pd.Timestamp(ev["end"]) if ev.get("end") else df.index.max() + pd.Timedelta("100D")
        m = (df.index >= s) & (df.index < e)
        df.loc[m, "f_event"] = True
        if ev.get("exclude_from_training"):
            df.loc[m, "f_event_excluded"] = True
    rep["event_hours"] = int(df["f_event"].sum())

    # 10. Xüsusiyyətlər üçün doldurulmuş sıra
    filled = q.interpolate(method="time", limit=ccfg["interp_max_gap_h"], limit_area="inside")
    df["f_interpolated"] = q.isna() & filled.notna()
    still = filled.isna()
    for lag in (24, 168):  # uzun boşluq: əvvəlki gün/həftənin eyni saatı
        for _ in range(60):
            prev = filled.shift(lag)
            m = filled.isna() & prev.notna()
            if not m.any():
                break
            filled[m] = prev[m]
    filled = filled.bfill().ffill()
    df["f_profile_filled"] = still & filled.notna()
    df["q_filled"] = filled
    rep["interpolated_hours"] = int(df["f_interpolated"].sum())
    rep["profile_filled_hours"] = int(df["f_profile_filled"].sum())

    # Digər kanallar: qısa boşluqlar
    for c in ["sg", "dp_kpa", "p_kpa", "t_c"]:
        df[c] = df[c].interpolate(method="time", limit=ccfg["interp_max_gap_h"], limit_area="inside")

    # 11. Təlim üçün yararlılıq
    df["train_ok"] = (df["q_clean"].notna() & ~df["f_depressurized"] & ~df["f_outage"]
                      & ~df["f_anomaly_long"] & ~df["f_event_excluded"])
    rep["train_ok_hours"] = int(df["train_ok"].sum())
    rep["train_ok_share"] = round(float(df["train_ok"].mean()), 4)
    rep["period"] = [str(df.index.min()), str(df.index.max())]

    df = df.drop(columns=["q_hour"])
    df.insert(0, "station", station_id)
    return df.reset_index(), rep
