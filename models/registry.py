"""Model versiyalama.

Hər versiya: kvantil modellər + klaster modeli + meta.json → zip arxivi saxlama qatına
("models/<versiya>.zip" faylı, "model/meta/<versiya>" və "model/latest" açarları).
Lokal nüsxə storage/models/<versiya>/ qovluğunda keşlənir (bulud konteynerində açılışda endirilir).
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import joblib

from core.config import path
from core.storage import blob_get, blob_put, kv_get, kv_set
from models.lgbm import QuantileForecaster


def new_version() -> str:
    return datetime.now().strftime("v%Y%m%d-%H%M%S")


def _zip_dir(d: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in d.iterdir():
            z.write(f, f.name)
    return buf.getvalue()


def save(forecaster: QuantileForecaster, meta: dict, clusterer=None) -> Path:
    v = meta.setdefault("version", new_version())
    d = path("models") / v
    forecaster.save(d)
    if clusterer is not None:
        joblib.dump(clusterer, d / "clusterer.joblib")
    meta["created_at"] = datetime.now().isoformat(timespec="seconds")
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    blob_put(f"models/{v}.zip", _zip_dir(d))
    meta_js = json.loads(json.dumps(meta, default=str))
    kv_set(f"model/meta/{v}", meta_js)
    versions = kv_get("model/versions", []) or []
    kv_set("model/versions", [v] + [x for x in versions if x != v])
    kv_set("model/latest", v)  # aktivləşdirmə ən sonda
    return d


def latest_version() -> str | None:
    return kv_get("model/latest")


def _ensure_local(v: str) -> Path:
    d = path("models") / v
    if (d / "meta.json").exists():
        return d
    data = blob_get(f"models/{v}.zip")
    if data is None:
        raise FileNotFoundError(f"Model {v} saxlama qatında tapılmadı")
    d.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(d)
    return d


def load(version: str | None = None):
    v = version or latest_version()
    if not v:
        raise FileNotFoundError("Model tapılmadı. Əvvəlcə: python -m models.train")
    d = _ensure_local(v)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    clusterer = joblib.load(d / "clusterer.joblib") if (d / "clusterer.joblib").exists() else None
    return QuantileForecaster.load(d), meta, clusterer


def activate(version: str) -> None:
    """Köhnə versiyaya geri qayıtmaq üçün."""
    _ensure_local(version)
    kv_set("model/latest", version)


def list_versions() -> list[dict]:
    out = []
    latest = latest_version()
    for v in kv_get("model/versions", []) or []:
        m = kv_get(f"model/meta/{v}") or {}
        out.append({"version": v, "active": v == latest,
                    **{k: m.get(k) for k in ("created_at", "train_end", "n_train_rows", "interval_scale")}})
    return out
