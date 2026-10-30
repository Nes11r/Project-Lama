"""REST API — proqnoz, tarixçə, dəqiqlik, xəbərdarlıqlar, tənzimləmələr.

İşə salma:  uvicorn api.main:app --port 8000
Sənədləşmə: http://localhost:8000/docs

TƏHLÜKƏSİZLİK: API yalnız oxuyur və tövsiyə verir. ROC/SCADA-ya yazma yoxdur.
/api/v1/control/* yalnız gələcək avtomatika üçün İNTERFEYS təsviridir və deaktivdir (HTTP 403).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import os

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.config import load_config, path, stations
from core.storage import blob_put, describe, has_table, kv_get, kv_set, read_table, using_db
from data.pipeline import load_hourly
from decision.alerts import evaluate_alerts, limits, save_limits
from decision.recommendations import recommendations
from external.common import read_status
from models.predict import get_service, service_ready

log = logging.getLogger("api")


def require_token(x_api_key: str | None = Header(default=None)):
    """Yazma əməliyyatları: API_TOKEN təyin olunubsa X-API-Key başlığı tələb olunur."""
    token = os.environ.get("API_TOKEN")
    if token and x_api_key != token:
        raise HTTPException(401, "Bu əməliyyat üçün icazə yoxdur (X-API-Key)")


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%dT%H:%M")
    df = df.replace([np.inf, -np.inf], np.nan).astype(object).where(df.notna(), None)
    return df.to_dict(orient="records")


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _records(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.strftime("%Y-%m-%dT%H:%M")
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------- qovluq izləyicisi
def _watch_incoming(stop: threading.Event, interval_s: int = 60):
    """dataset/incoming/ qovluğunda yeni/dəyişmiş CSV → pipeline + servis yenilənməsi."""
    last = None
    while not stop.wait(interval_s):
        try:
            files = list(path("incoming_dir").glob("*.csv"))
            sig = max((f.stat().st_mtime for f in files), default=0), len(files)
            if last is not None and sig != last:
                log.info("Yeni fayl aşkarlandı → pipeline")
                _ingest()
            last = sig
        except Exception as e:  # izləyici heç vaxt API-ni dayandırmamalıdır
            log.error("İzləyici xətası: %s", e)


def _ingest() -> dict:
    from data.pipeline import run as run_pipeline
    hourly = run_pipeline()
    try:
        from external.weather import update_tables
        update_tables(end=str(hourly.ts.max().date()))
    except Exception as e:
        log.warning("Hava yenilənmədi (keş istifadə olunur): %s", e)
    get_service().reload()
    return {"rows": int(len(hourly)), "data_end": {s: str(g.ts.max()) for s, g in hourly.groupby("station")}}


_LOAD = {"busy": False, "error": None, "failed_at": 0.0}
_LOAD_LOCK = threading.Lock()
RETRY_AFTER_FAIL_S = 30


def _warm_up():
    """Model və data fonda yüklənir: port dərhal açılır, Render proxy-si 502 qaytarmır."""
    with _LOAD_LOCK:
        if _LOAD["busy"]:
            return
        _LOAD["busy"] = True
    t0 = time.monotonic()
    try:
        get_service()
        _LOAD["error"] = None
        log.info("Servis hazırdır (%.0f s)", time.monotonic() - t0)
    except Exception as e:
        _LOAD.update(error=str(e), failed_at=time.monotonic())
        log.exception("Servis yüklənmədi")
    finally:
        _LOAD["busy"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    stop = threading.Event()
    t = threading.Thread(target=_watch_incoming, args=(stop,), daemon=True)
    t.start()
    yield
    stop.set()


app = FastAPI(title="Qaz sərfiyyatı proqnozu — Operator API", version="0.1.0", lifespan=lifespan)
V1 = "/api/v1"
NEEDS_NO_SERVICE = (f"{V1}/health", f"{V1}/control/")  # control: model olmadan da həmişə 403


@app.middleware("http")
async def _wait_for_service(request: Request, call_next):
    """Model hələ yüklənirsə data sorğularına 503 (panel bunu gözləyib təkrarlayır)."""
    p = request.url.path
    if p.startswith(V1) and not p.startswith(NEEDS_NO_SERVICE) and not service_ready():
        err = _LOAD["error"]
        if err and not _LOAD["busy"] and time.monotonic() - _LOAD["failed_at"] > RETRY_AFTER_FAIL_S:
            threading.Thread(target=_warm_up, daemon=True).start()
        detail = f"Model yüklənmədi: {err}" if err else "Model yüklənir, bir neçə saniyədən sonra yenidən cəhd edin"
        return JSONResponse({"detail": detail}, status_code=503, headers={"Retry-After": "5"})
    return await call_next(request)


def _check(sid: str):
    if sid not in stations():
        raise HTTPException(404, f"Naməlum məntəqə: {sid}")


# ---------------------------------------------------------------- ümumi
@app.get(f"{V1}/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat(timespec="seconds")}


@app.get(f"{V1}/status")
def status():
    svc = get_service()
    now = svc.clock.now()
    return _clean({
        "mode": load_config()["demo"]["mode"], "now": now, "model_version": svc.meta["version"],
        "model_trained_until": svc.meta.get("train_end"), "automation_enabled": False,
        "storage": describe(),
        "data_end": {sid: svc.data_end(sid) for sid in stations()},
        "external_sources": read_status(),
    })


@app.get(f"{V1}/stations")
def list_stations():
    lim = limits()
    return [{"id": sid, "name": s["name"], "area": s["area"], "limits": lim[sid]} for sid, s in stations().items()]


class ClockIn(BaseModel):
    now: str | None = None
    advance_hours: int | None = None


@app.get(f"{V1}/clock")
def get_clock():
    return _clean({"mode": load_config()["demo"]["mode"], "now": get_service().clock.now()})


@app.put(f"{V1}/clock", dependencies=[Depends(require_token)])
def set_clock(body: ClockIn):
    svc = get_service()
    if load_config()["demo"]["mode"] != "replay":
        raise HTTPException(400, "Saat yalnız replay (demo) rejimində dəyişdirilə bilər")
    now = pd.Timestamp(body.now) if body.now else svc.clock.now()
    if body.advance_hours:
        now += pd.Timedelta(hours=body.advance_hours)
    return _clean({"now": svc.clock.set(now)})


# ---------------------------------------------------------------- proqnoz
@app.get(f"{V1}/forecast/{{sid}}")
def forecast(sid: str, origin: str | None = None):
    _check(sid)
    svc = get_service()
    fc = svc.forecast(sid, pd.Timestamp(origin) if origin else None)
    return _clean(fc)


@app.get(f"{V1}/dashboard/{{sid}}")
def dashboard(sid: str):
    """Əsas panel üçün hər şey bir sorğuda: cari vəziyyət + 24 saat proqnoz + xəbərdarlıq + tövsiyə."""
    _check(sid)
    svc = get_service()
    now = svc.clock.now()
    fc = svc.forecast(sid)
    recent = svc.recent_forecast_vs_actual(sid, hours=48)
    st = svc.hourly[svc.hourly.station == sid]
    alerts = evaluate_alerts(sid, now, fc, recent, st)
    _store_alerts(alerts)
    cur = st[st.ts <= now].tail(1)
    last24 = st[(st.ts <= now) & (st.ts > now - pd.Timedelta("24h"))]
    current = {}
    if len(cur):
        c = cur.iloc[0]
        current = {"ts": c.ts, "flow": c.q_raw, "pressure": c.p_kpa, "gas_temp": c.t_c, "dp": c.dp_kpa,
                   "sg": c.sg, "flow_24h_total": last24.q_raw.sum(), "pressure_min_24h": last24.p_kpa.min()}
    return _clean({"station": sid, "station_name": stations()[sid]["name"], "now": now, "current": current,
                   "limits": limits()[sid], "forecast": fc, "recent": recent,
                   "alerts": alerts, "recommendations": recommendations(sid, fc, alerts)})


# ---------------------------------------------------------------- tarixçə və dəqiqlik
@app.get(f"{V1}/history/{{sid}}")
def history(sid: str, start: str | None = None, end: str | None = None, with_forecast: bool = True):
    _check(sid)
    svc = get_service()
    now = svc.clock.now()
    end_ts = min(pd.Timestamp(end) if end else now, now)
    start_ts = pd.Timestamp(start) if start else end_ts - pd.Timedelta("7D")
    lo, hi = svc.window
    src = svc.hourly if start_ts >= lo and (hi is None or end_ts <= hi) else load_hourly(start_ts, end_ts)
    st = src[(src.station == sid) & (src.ts >= start_ts) & (src.ts <= end_ts)]
    cols = ["ts", "q_raw", "q_clean", "p_kpa", "t_c", "dp_kpa", "f_reconstructed", "f_spike", "f_event", "train_ok"]
    out = {"station": sid, "actual": st[cols]}
    if with_forecast:
        hours = int((end_ts - start_ts) / pd.Timedelta("1h")) + 1
        if hours <= 24 * 31 and start_ts >= lo + pd.Timedelta("22D"):
            out["day_ahead"] = svc.recent_forecast_vs_actual(sid, hours=hours)
    return _clean(out)


@app.get(f"{V1}/accuracy/{{sid}}")
def accuracy(sid: str):
    _check(sid)
    acc = kv_get("accuracy")
    if acc is None and not has_table("backtest"):
        raise HTTPException(404, "Backtest yoxdur: python -m models.evaluate")
    if acc is None and using_db():  # bulud: 440k sətirlik backtest yaddaşa sığmır
        raise HTTPException(404, "Dəqiqlik xülasəsi yoxdur: lokal maşında python scripts/publish.py --no-train")
    if acc is None:  # lokal köhnə qurulum: xülasə bir dəfə hesablanır və saxlanılır
        from models.metrics import accuracy_summary
        acc = _clean(accuracy_summary(read_table(
            "backtest", ["station", "fold", "origin", "h", "q_actual", "q10", "q50", "q90", "bl_q_lag24"])))
        kv_set("accuracy", acc)
    if sid not in acc:
        raise HTTPException(404, f"Bu məntəqə üçün backtest yoxdur: {sid}")
    return _clean({"station": sid, **acc[sid], "global": kv_get("model_metrics", {}) or {}})


# ---------------------------------------------------------------- xəbərdarlıqlar
def _load_store() -> dict:
    return kv_get("alerts", {}) or {}


def _store_alerts(alerts: list[dict]):
    store = _load_store()
    now = datetime.now().isoformat(timespec="seconds")
    for a in alerts:
        s = store.get(a["id"], {"first_seen": now, "acknowledged": False})
        s.update({**a, "last_seen": now})
        store[a["id"]] = s
    kv_set("alerts", _clean(store))


@app.get(f"{V1}/alerts")
def alerts(station: str | None = None, refresh: bool = True, include_ack: bool = True):
    svc = get_service()
    if refresh:
        now = svc.clock.now()
        for sid in ([station] if station else list(stations())):
            fc = svc.forecast(sid, explain=False)
            recent = svc.recent_forecast_vs_actual(sid, hours=48)
            _store_alerts(evaluate_alerts(sid, now, fc, recent, svc.hourly[svc.hourly.station == sid]))
    items = list(_load_store().values())
    if station:
        items = [a for a in items if a["station"] == station]
    if not include_ack:
        items = [a for a in items if not a.get("acknowledged")]
    return sorted(items, key=lambda a: a.get("last_seen", ""), reverse=True)


@app.post(f"{V1}/alerts/{{aid}}/ack", dependencies=[Depends(require_token)])
def ack(aid: str, operator: str = Form("operator")):
    store = _load_store()
    if aid not in store:
        raise HTTPException(404, "Xəbərdarlıq tapılmadı")
    store[aid].update({"acknowledged": True, "ack_by": operator, "ack_at": datetime.now().isoformat(timespec="seconds")})
    kv_set("alerts", store)
    return store[aid]


# ---------------------------------------------------------------- tənzimləmələr
class LimitsIn(BaseModel):
    peak_flow_warning: float | None = None
    peak_flow_critical: float | None = None
    min_pressure_kpa: float | None = None


@app.get(f"{V1}/settings/limits")
def get_limits():
    return limits()


@app.put(f"{V1}/settings/limits/{{sid}}", dependencies=[Depends(require_token)])
def put_limits(sid: str, body: LimitsIn):
    _check(sid)
    v = body.model_dump()
    if v["peak_flow_warning"] and v["peak_flow_critical"] and v["peak_flow_warning"] > v["peak_flow_critical"]:
        raise HTTPException(422, "Xəbərdarlıq həddi kritik həddən böyük ola bilməz")
    return save_limits(sid, v)


@app.get(f"{V1}/models")
def models():
    from models.registry import list_versions
    return _clean(list_versions())


# ---------------------------------------------------------------- data yükləmə
@app.post(f"{V1}/upload", dependencies=[Depends(require_token)])
async def upload(station: str = Form(...), file: UploadFile = File(...)):
    _check(station)
    from data.loader import parse_roc_file
    raw = await file.read()
    tmp = path("runtime") / f"upload_{int(time.time())}.csv"
    tmp.write_bytes(raw)
    try:
        rows, totals = parse_roc_file(tmp)
        if rows.empty:
            raise ValueError("Faylda saatlıq sətir tapılmadı")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(422, f"Format uyğun deyil (ROC ixracı gözlənilir): {e}")
    dest = path("incoming_dir") / f"{station}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    tmp.replace(dest)
    blob_put(f"raw/incoming/{dest.name}", raw)  # bulud: disk müvəqqətidir → xam fayl bazada saxlanır
    global _BT
    _BT = None
    res = _ingest()
    return _clean({"saved_as": dest.name, "rows_in_file": len(rows), "daily_totals": len(totals),
                   "period": [rows.ts.min(), rows.ts.max()], **res})


# ---------------------------------------------------------------- avtomatika (DEAKTİV)
@app.get(f"{V1}/control/spec")
def control_spec():
    """Gələcək avtomatika üçün nəzərdə tutulan interfeys (yalnız təsvir)."""
    return {
        "enabled": False,
        "policy": "Sistem yalnız tövsiyə rejimində işləyir. ROC/SCADA-ya yazma qadağandır.",
        "planned_endpoints": [
            {"method": "POST", "path": "/api/v1/control/setpoint",
             "body": {"station": "str", "parameter": "outlet_pressure_kpa", "value": "float",
                      "reason": "str", "requested_by": "str", "approved_by": "str (məcburi ikinci şəxs)"}},
        ],
        "requirements_before_enabling": ["İki mərhələli təsdiq (operator + dispetçer)", "Audit jurnalı",
                                          "Təhlükəsizlik hüdudları (min/max) və avtomatik geri qaytarma",
                                          "SCADA tərəfində ayrıca icazə və sınaq mühiti"],
    }


@app.post(f"{V1}/control/{{action}}")
def control_disabled(action: str):
    raise HTTPException(403, "Avtomatik idarəetmə deaktivdir: sistem yalnız tövsiyə rejimində işləyir.")
