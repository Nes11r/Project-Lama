"""Proqnoz servisi: növbəti 24 saat, qeyri-müəyyənlik intervalı və sadə dildə izah.

Rejimlər (config.yaml → demo.mode):
  replay — tarixi data "real vaxt" kimi oynadılır. "İndi" = saat (runtime/clock.json).
           Hava: indiyə qədər faktiki, sonrası 1 gün əvvəlki proqnoz (sızma yoxdur).
  live   — "indi" = sistem saatı. Hava: Open-Meteo Forecast API (əlçatmazsa keş + xəbərdarlıq).
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime

import numpy as np
import pandas as pd

from core.config import load_config, load_yaml
from core.storage import kv_get, kv_set
from data.pipeline import load_hourly
from features.build import FEATURE_GROUPS, FeatureContext, make_rows
from features.context import weather_table
from features.weather_features import WEATHER_COLS
from models import registry

log = logging.getLogger(__name__)

# İzah üçün xüsusiyyətlərin operatora başa düşülən qruplara bölünməsi
EXPLAIN_GROUPS = {
    "Hava (temperatur, külək və s.)": FEATURE_GROUPS["weather"],
    "Təqvim (bayram, iş/qeyri-iş günü, tətil)": FEATURE_GROUPS["calendar"] + ["dow"],
    "Son sərf trendi": ["lag_last", "mean_24", "mean_168", "trend_24", "min_24", "q_lag24", "q_lag48",
                        "q_lag168", "same_hour_mean7", "profile_cluster_prev"],
    "Mövsüm / günün saatı": ["hour", "hour_sin", "hour_cos", "doy_sin", "doy_cos"],
}

# Yaddaşa qənaət: servis bütün tarixçəni yox, yalnız "indi" ətrafındakı pəncərəni saxlayır.
# Ən uzun geriyə baxış: tarixçə səhifəsi (31 gün) + 1 gün lead + xüsusiyyətlər (168 saat + 8 gün qoruma).
WINDOW_BACK = pd.Timedelta("60D")
WINDOW_FWD = pd.Timedelta("5D")
NEED_BACK = pd.Timedelta("55D")
NEED_FWD = pd.Timedelta("2D")


class Clock:
    """Demo (replay) rejimində "indiki an" (saxlama qatında: bütün instansiyalar eyni saatı görür)."""

    def now(self) -> pd.Timestamp:
        cfg = load_config()["demo"]
        if cfg["mode"] == "live":
            return pd.Timestamp(datetime.now()).floor("h")
        v = kv_get("clock")
        return pd.Timestamp(v["now"]) if v else pd.Timestamp(cfg["now"])

    def set(self, ts) -> pd.Timestamp:
        ts = pd.Timestamp(ts).floor("h")
        kv_set("clock", {"now": str(ts)})
        return ts


class ForecastService:
    def __init__(self):
        self.lock = threading.RLock()
        self.clock = Clock()
        self.reload()

    # ------------------------------------------------------------------ yükləmə
    def reload(self):
        with self.lock:
            self.model, self.meta, self.clusterer = registry.load()
            self._load_window(self.clock.now())
            self.events = load_yaml("events").get("events", []) or []
            log.info("Servis yükləndi: model %s", self.meta["version"])

    def _load_window(self, now: pd.Timestamp):
        lo = now - WINDOW_BACK
        hi = None if load_config()["demo"]["mode"] == "live" else now + WINDOW_FWD
        self.hourly = None  # köhnə pəncərə yenisi yüklənməzdən əvvəl buraxılır
        self.hourly = load_hourly(lo, hi)
        self.w_actual = weather_table("actual", lo, hi)
        try:
            self.w_fc = weather_table("forecast", lo, hi)
        except FileNotFoundError:
            self.w_fc = self.w_actual
        self._clusters = self.clusterer.transform(self.hourly) if self.clusterer else None
        self.window = (lo, hi)
        log.info("Data pəncərəsi: %s – %s (%d sətir)", lo, hi or "son", len(self.hourly))

    def ensure_window(self) -> pd.Timestamp:
        """Demo saatı pəncərədən çıxıbsa datanı yenidən yüklə. "İndi"ni qaytarır."""
        now = self.clock.now()
        with self.lock:
            lo, hi = self.window
            if now - NEED_BACK < lo or (hi is not None and now + NEED_FWD > hi):
                self._load_window(now)
        return now

    def data_end(self, sid: str) -> pd.Timestamp:
        return self.hourly.loc[self.hourly.station == sid, "ts"].max()

    def origin_for(self, sid: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
        """Proqnoz başlanğıcı: "indi"dən əvvəl tamamlanmış son saat (və data sonundan gec olmayan)."""
        now = now or self.clock.now()
        return min(now - pd.Timedelta("1h"), self.data_end(sid))

    # ------------------------------------------------------------------ hava
    def _weather(self, sid: str, origin: pd.Timestamp) -> tuple[pd.DataFrame, str]:
        mode = load_config()["demo"]["mode"]
        a = self.w_actual[sid]
        if mode == "live":
            from external.weather import station_weather
            try:
                f = station_weather(sid, "forecast", origin - pd.Timedelta("3D"), origin + pd.Timedelta("2D"))
                src = "Open-Meteo Forecast API"
            except Exception as e:  # keş də yoxdursa — faktiki son dəyərlər
                log.error("Forecast API: %s", e)
                f, src = a.iloc[0:0], "ƏLÇATMAZ — keş/faktiki"
            w = f.combine_first(a)
        else:
            f = self.w_fc[sid]
            w = pd.concat([a[a.index <= origin], f[f.index > origin]])
            src = "Open-Meteo Previous Runs (1 gün əvvəlki proqnoz)"
        return w[WEATHER_COLS], src

    def _ctx(self, sid: str, origin: pd.Timestamp) -> tuple[FeatureContext, str]:
        w, src = self._weather(sid, origin)
        hourly = self.hourly[(self.hourly.station == sid) & (self.hourly.ts <= origin)]
        ctx = FeatureContext(hourly=hourly, weather={sid: w}, hdd_base=self.meta["hdd_base"],
                             static=self.meta.get("static", {}), clusterer=self.clusterer)
        ctx._clusters = self._clusters
        return ctx, src

    def _rows(self, sid: str, origins: pd.DatetimeIndex) -> tuple[pd.DataFrame, str]:
        ctx, src = self._ctx(sid, origins.max())
        h = load_config()["forecast"]["horizon_h"]
        return make_rows(ctx, sid, h, origins=origins, until=origins.max() + pd.Timedelta(hours=h)), src

    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        lg = self.model.predict_log(X)
        k = float(self.meta.get("interval_scale", 1.0))
        lg["q10"] = lg["q50"] - k * (lg["q50"] - lg["q10"])
        lg["q90"] = lg["q50"] + k * (lg["q90"] - lg["q50"])
        return np.expm1(lg).clip(lower=0)

    # ------------------------------------------------------------------ hadisələr
    def active_events(self, sid: str, at: pd.Timestamp) -> list[dict]:
        out = []
        for ev in self.events:
            if ev.get("station") not in (sid, "all"):
                continue
            s = pd.Timestamp(ev["start"])
            e = pd.Timestamp(ev["end"]) if ev.get("end") else pd.Timestamp.max
            if s <= at < e:
                out.append(ev)
        return out

    def _event_scale(self, sid: str, origin: pd.Timestamp, events: list[dict]) -> float | None:
        ev = next((e for e in events if e.get("adaptive_scaling")), None)
        if not ev:
            return None
        start = max(pd.Timestamp(ev["start"]), origin - pd.Timedelta("72h"))
        origins = pd.date_range(start.ceil("6h"), origin - pd.Timedelta("6h"), freq="6h")
        if len(origins) == 0:
            return None
        X, _ = self._rows(sid, origins)
        X = X[X.ts <= origin]
        p = self._predict(X)["q50"]
        act = self.hourly[self.hourly.station == sid].set_index("ts")["q_clean"].reindex(X.ts).to_numpy()
        r = act / p.to_numpy()
        r = r[np.isfinite(r)]
        return float(np.clip(np.median(r), 0.05, 2.0)) if len(r) else None

    # ------------------------------------------------------------------ əsas API
    def forecast(self, sid: str, origin: pd.Timestamp | None = None, explain: bool = True) -> dict:
        with self.lock:
            origin = pd.Timestamp(origin) if origin is not None else self.origin_for(sid)
            X, wsrc = self._rows(sid, pd.DatetimeIndex([origin]))
            X = X.sort_values("h")
            P = self._predict(X)
            events = self.active_events(sid, origin)
            scale = self._event_scale(sid, origin, events)
            if scale is not None:
                P = P * scale
            df = pd.DataFrame({"ts": X.ts.to_numpy(), "h": X.h.to_numpy(),
                               "q10": P.q10.to_numpy(), "q50": P.q50.to_numpy(), "q90": P.q90.to_numpy(),
                               "temp": X.temp.to_numpy(), "actual": X.q_actual.to_numpy()})
            res = {"station": sid, "origin": origin, "issued_for": origin + pd.Timedelta("1h"),
                   "model_version": self.meta["version"], "weather_source": wsrc,
                   "event_scale": scale, "events": events, "forecast": df}
            if explain:
                res["explanation"] = self.explain(sid, origin, X, P)
            return res

    def explain(self, sid: str, origin: pd.Timestamp, X: pd.DataFrame, P: pd.DataFrame) -> dict:
        """Növbəti 24 saatın proqnozunu son 24 saatın modelləşdirilmiş səviyyəsi ilə müqayisə edir.

        Δlog(proqnoz) = Σ_qrup ΔSHAP_qrup  → hər qrupun töhfəsi faizlə.
        """
        ref_origin = origin - pd.Timedelta("24h")
        R, _ = self._rows(sid, pd.DatetimeIndex([ref_origin]))
        R = R.sort_values("h")
        booster = self.model.models[0.5]
        feats = self.model.features
        # LightGBM daxili TreeSHAP: son sütun baza dəyəridir
        sv_f = pd.DataFrame(booster.predict(self.model.matrix(X), pred_contrib=True)[:, :-1], columns=feats).mean()
        sv_r = pd.DataFrame(booster.predict(self.model.matrix(R), pred_contrib=True)[:, :-1], columns=feats).mean()
        delta = sv_f - sv_r
        contrib = {}
        for g, fs in EXPLAIN_GROUPS.items():
            d = float(delta[[f for f in fs if f in delta.index]].sum())
            contrib[g] = d
        other = float(delta.sum() - sum(contrib.values()))
        contrib["Digər"] = other
        tot_f = float(P.q50.sum())
        ref_act = self.hourly[(self.hourly.station == sid) & (self.hourly.ts > ref_origin)
                              & (self.hourly.ts <= origin)]["q_filled"].sum()
        t_next, t_prev = float(X.temp.mean()), float(R.temp.mean())
        items = []
        for g, d in sorted(contrib.items(), key=lambda kv: -abs(kv[1])):
            pct = 100 * (np.exp(d) - 1)
            if abs(pct) < 1:
                continue
            text = f"{g}: {pct:+.0f}%"
            if g.startswith("Hava"):
                trend = "soyuma" if t_next < t_prev - 0.5 else ("istiləşmə" if t_next > t_prev + 0.5 else "sabit hava")
                consistent = (trend == "soyuma" and pct > 0) or (trend == "istiləşmə" and pct < 0)
                if consistent:
                    text = (f"Gözlənilən {trend} (orta temperatur {t_prev:.1f}°C → {t_next:.1f}°C) "
                            f"səbəbindən sərfiyyat {pct:+.0f}%")
                else:  # temperatur istiqaməti ilə uyğun deyil → digər hava amilləri (radiasiya, külək) üstündür
                    text = (f"Hava amilləri (orta temperatur {t_prev:.1f}°C → {t_next:.1f}°C, "
                            f"külək/günəş radiasiyası daxil): {pct:+.0f}%")
            elif g.startswith("Təqvim"):
                nw = int(X.is_nonworking.max()), int(R.is_nonworking.max())
                hol = X.loc[X.is_holiday == 1, "ts"]
                why = "bayram günü" if len(hol) else ("qeyri-iş günü" if nw[0] else "iş günü")
                text = f"Təqvim amili ({why}): {pct:+.0f}%"
            items.append({"group": g, "pct": round(pct, 1), "text": text})
        return {"next24_total": round(tot_f, 2), "prev24_actual_total": round(float(ref_act), 2),
                "change_pct": round(100 * (tot_f / ref_act - 1), 1) if ref_act > 0 else None,
                "temp_next24": round(t_next, 1), "temp_prev24": round(t_prev, 1), "factors": items}

    def recent_forecast_vs_actual(self, sid: str, hours: int = 48, lead: int = 24) -> pd.DataFrame:
        """Son N saat üçün `lead` saat əvvəl verilmiş proqnozlar və faktiki (sapma aşkarlanması üçün)."""
        with self.lock:
            origin = self.origin_for(sid)
            targets_start = origin - pd.Timedelta(hours=hours - 1)
            origins = pd.date_range((targets_start - pd.Timedelta(hours=lead)).floor("h"),
                                    origin - pd.Timedelta(hours=1), freq=f"{lead}h")
            if len(origins) == 0:
                return pd.DataFrame()
            X, _ = self._rows(sid, origins)
            X = X[(X.ts > X.origin) & (X.ts <= origin) & (X.ts >= targets_start)]
            P = self._predict(X)
            out = pd.DataFrame({"ts": X.ts.to_numpy(), "h": X.h.to_numpy(), "q10": P.q10.to_numpy(),
                                "q50": P.q50.to_numpy(), "q90": P.q90.to_numpy()})
            g = self.hourly[self.hourly.station == sid].set_index("ts")
            out["actual"] = g["q_raw"].reindex(out.ts).to_numpy()
            out["pressure"] = g["p_kpa"].reindex(out.ts).to_numpy()
            out["event"] = g["f_event"].reindex(out.ts).fillna(False).to_numpy()
            return out.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)


_service: ForecastService | None = None
_service_lock = threading.Lock()


def service_ready() -> bool:
    return _service is not None


def get_service() -> ForecastService:
    global _service
    if _service is None:
        with _service_lock:  # eyni anda iki yükləmə yaddaşı ikiqat artırardı
            if _service is None:
                _service = ForecastService()
    _service.ensure_window()
    return _service
