"""Xəbərdarlıqlar (yalnız MƏLUMAT/TÖVSİYƏ — heç bir idarəetmə əmri göndərilmir).

Növlər:
  peak_forecast      — proqnoz konfiqurasiya olunan pik limitlərini keçir
  low_pressure       — cari təzyiq minimumdan aşağıdır / pik + aşağı təzyiq ehtiyatı riski
  deviation_high     — faktiki proqnozdan sistemli YUXARI (mümkün sızma, qeydə alınmamış istehlak, sensor)
  deviation_low      — faktiki proqnozdan sistemli AŞAĞI (təchizat məhdudiyyəti, sensor nasazlığı)
  night_excess       — gecə minimum sərfi gözləniləndən yüksək (klassik sızma göstəricisi)
  sensor_flatline    — sərf dəyişmir (donmuş sensor)
  data_stale         — yeni data gəlmir
  dp_saturation      — ΔP ötürücüsü diapazon sonunda (sərf aşağı ölçülə bilər)
  event_active       — qeydiyyatlı hadisə (planlı təmir); müvafiq xəbərdarlıqlar susdurulur
  external_source    — xarici API əlçatmazdır, keş istifadə olunur
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from core.config import load_config, stations
from core.storage import kv_get, kv_set
from external.common import read_status

SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}


def limits() -> dict:
    """config.yaml limitləri + operatorun paneldən etdiyi dəyişikliklər (runtime/limits.json)."""
    lim = {sid: dict(s["limits"]) for sid, s in stations().items()}
    for sid, v in (kv_get("limits", {}) or {}).items():
        lim.setdefault(sid, {}).update(v)
    return lim


def save_limits(sid: str, values: dict) -> dict:
    cur = kv_get("limits", {}) or {}
    cur.setdefault(sid, {}).update({k: float(v) for k, v in values.items() if v is not None})
    kv_set("limits", cur)
    return limits()[sid]


def _alert(sid, ts, severity, type_, title, message, reason, suppressed=False) -> dict:
    aid = hashlib.md5(f"{sid}|{type_}|{pd.Timestamp(ts).floor('D')}".encode()).hexdigest()[:12]
    return {"id": aid, "station": sid, "station_name": stations()[sid]["name"], "ts": str(ts),
            "severity": severity, "type": type_, "title": title, "message": message, "reason": reason,
            "suppressed": suppressed}


def _consecutive(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def evaluate_alerts(sid: str, now: pd.Timestamp, fc: dict, recent: pd.DataFrame, hourly_st: pd.DataFrame) -> list[dict]:
    cfg = load_config()["alerts"]
    lim = limits()[sid]
    out: list[dict] = []
    f = fc["forecast"]
    events = fc.get("events", [])
    suppress = any(e.get("suppress_alerts") for e in events)

    # --- Qeydiyyatlı hadisə
    for ev in events:
        out.append(_alert(sid, now, "info", "event_active", f"Aktiv hadisə: {ev.get('title', ev['id'])}",
                          ev.get("description", ""),
                          "Hadisə config/events.yaml-da qeydə alınıb. Sapma xəbərdarlıqları susdurulur"
                          + (f"; proqnoz faktiki səviyyəyə uyğunlaşdırılıb (×{fc['event_scale']:.2f})"
                             if fc.get("event_scale") else "") + "."))

    # --- Pik tələbat
    peak = f.loc[f.q50.idxmax()]
    peak90 = f.q90.max()
    if peak.q50 >= lim["peak_flow_critical"]:
        sev = "critical"
    elif peak.q50 >= lim["peak_flow_warning"] or peak90 >= lim["peak_flow_critical"]:
        sev = "warning"
    else:
        sev = None
    if sev:
        out.append(_alert(
            sid, peak.ts, sev, "peak_forecast",
            f"Pik tələbat gözlənilir: {pd.Timestamp(peak.ts):%d.%m %H:00}",
            f"Proqnoz {peak.q50:.2f} min m³/saat (90% yuxarı hədd {peak90:.2f}); "
            f"xəbərdarlıq həddi {lim['peak_flow_warning']}, kritik hədd {lim['peak_flow_critical']}.",
            "Median proqnoz və ya yuxarı kvantil konfiqurasiya olunan limiti keçir."))

    # --- Təzyiq
    last = hourly_st[hourly_st.ts <= now].tail(24)
    p_now = last["p_kpa"].dropna().iloc[-1] if last["p_kpa"].notna().any() else np.nan
    p_min24 = last["p_kpa"].min()
    if np.isfinite(p_now) and p_now < lim["min_pressure_kpa"]:
        out.append(_alert(sid, now, "critical", "low_pressure", "Aşağı təzyiq",
                          f"Cari təzyiq {p_now:.0f} kPa < minimum {lim['min_pressure_kpa']:.0f} kPa.",
                          "Ölçülən mütləq təzyiq konfiqurasiya olunan minimumdan aşağıdır.", suppress))
    elif np.isfinite(p_min24) and p_min24 < 1.1 * lim["min_pressure_kpa"] and peak.q50 >= 0.9 * lim["peak_flow_warning"]:
        out.append(_alert(sid, peak.ts, "warning", "low_pressure", "Pik saatlarda aşağı təzyiq riski",
                          f"Son 24 saatın minimum təzyiqi {p_min24:.0f} kPa (minimumdan ≤10% yuxarı), "
                          f"növbəti pik {peak.q50:.2f} min m³/saat (xəbərdarlıq həddinin ≥90%-i).",
                          "Yüksək tələbat + az təzyiq ehtiyatı birlikdə.", suppress))

    # --- Faktiki ↔ proqnoz sapması
    if not recent.empty:
        r = recent.dropna(subset=["actual"]).tail(48)
        sigma = ((r.q90 - r.q10) / 2.563).clip(lower=1e-3)
        z = (r.actual - r.q50) / sigma
        hi_run = _consecutive((z > cfg["deviation_z"]).to_numpy()[-12:])
        lo_run = _consecutive((z < -cfg["deviation_z"]).to_numpy()[-12:])
        if hi_run >= cfg["deviation_min_hours"]:
            out.append(_alert(sid, now, "warning", "deviation_high", "Faktiki sərf proqnozdan yüksəkdir",
                              f"Son 12 saatda {hi_run} ardıcıl saat faktiki dəyər 90% intervalından xeyli yuxarıdır.",
                              "Mümkün səbəblər: sızma, qeydə alınmamış istehlak, sayğac/sensor xətası. "
                              "Sahədə yoxlama tövsiyə olunur.", suppress))
        if lo_run >= cfg["deviation_min_hours"]:
            out.append(_alert(sid, now, "warning", "deviation_low", "Faktiki sərf proqnozdan aşağıdır",
                              f"Son 12 saatda {lo_run} ardıcıl saat faktiki dəyər intervalın xeyli altındadır.",
                              "Mümkün səbəblər: təchizat məhdudiyyəti/kəsinti, klapan vəziyyəti, sensor nasazlığı.",
                              suppress))
        # Gecə minimum sərfi (sızma göstəricisi)
        h0, h1 = cfg["night_excess_hours"]
        night = r[(r.ts.dt.hour >= h0) & (r.ts.dt.hour <= h1)].tail(h1 - h0 + 1)
        if len(night) >= 3 and night.q50.sum() > 0:
            ratio = night.actual.sum() / night.q50.sum()
            if ratio > cfg["night_excess_ratio"]:
                out.append(_alert(sid, now, "warning", "night_excess", "Gecə minimum sərfi yüksəkdir",
                                  f"Son gecə {h0:02d}:00–{h1 + 1:02d}:00 faktiki sərf proqnozdan {100 * (ratio - 1):.0f}% çoxdur.",
                                  "Gecə istehlak minimaldır; davamlı artıq sərf sızmanın klassik əlamətidir.",
                                  suppress))

    # --- Sensor / data keyfiyyəti
    tail = hourly_st[hourly_st.ts <= now].tail(cfg["flatline_hours"])
    if len(tail) == cfg["flatline_hours"] and tail.q_raw.notna().all() and tail.q_raw.nunique() == 1:
        out.append(_alert(sid, now, "warning", "sensor_flatline", "Sərf göstəricisi dəyişmir",
                          f"Son {cfg['flatline_hours']} saat eyni dəyər: {tail.q_raw.iloc[0]:.3f}.",
                          "Donmuş sensor və ya rabitə problemi ola bilər."))
    data_end = hourly_st.ts.max()
    if now - data_end > pd.Timedelta("3h"):
        out.append(_alert(sid, now, "warning", "data_stale", "Yeni data gəlmir",
                          f"Son ölçmə: {data_end:%d.%m.%Y %H:00} ({(now - data_end) / pd.Timedelta('1h'):.0f} saat əvvəl).",
                          "Proqnoz son məlum dəyərlərə əsaslanır; dəqiqlik azalır."))
    if last["f_dp_saturated"].any():
        out.append(_alert(sid, now, "info", "dp_saturation", "ΔP ötürücüsü diapazon sonunda",
                          "Son 24 saatda ΔP ≥ saturasiya həddi.",
                          "Bu saatlarda sərf aşağı ölçülə bilər; diafraqma/ötürücü diapazonunu yoxlayın."))

    # --- Xarici mənbələr
    for src, st in read_status().items():
        if not st.get("ok", True):
            out.append(_alert(sid, now, "info", "external_source", f"Xarici mənbə əlçatmazdır: {src}",
                              st.get("message", ""), "Son keşlənmiş məlumat istifadə olunur."))
    return sorted(out, key=lambda a: (SEV_ORDER[a["severity"]], a["ts"]))
