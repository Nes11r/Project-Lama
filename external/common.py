"""Xarici API-lər üçün ümumi köməkçilər: HTTP (təkrar cəhd + rate limit), keş, mənbə statusu.

Prinsip: API əlçatmaz olduqda sistem dayanmır — son keşlənmiş məlumat istifadə olunur və
status faylına xəbərdarlıq yazılır (operator panelində göstərilir).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import requests

from core.config import load_config, path

log = logging.getLogger(__name__)
_last_call: dict[str, float] = {}


def cache_dir(sub: str) -> Path:
    d = path("cache") / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def http_get(url: str, params: dict | None = None, *, data: dict | None = None,
             retries: int = 3, source: str = "http") -> requests.Response:
    cfg = load_config()
    pause = cfg["weather"]["request_pause_s"]
    headers = {"User-Agent": cfg["population"]["user_agent"]}
    host = url.split("/")[2]
    for attempt in range(1, retries + 1):
        wait = pause - (time.time() - _last_call.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.time()
        try:
            if data is not None:
                r = requests.post(url, data=data, headers=headers, timeout=cfg["weather"]["timeout_s"])
            else:
                r = requests.get(url, params=params, headers=headers, timeout=cfg["weather"]["timeout_s"])
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            log.warning("%s cəhd %d/%d uğursuz: %s", source, attempt, retries, e)
            if attempt == retries:
                raise
            # 429 = rate limit (Open-Meteo dəqiqəlik limit) → bir dəqiqədən çox gözlə
            time.sleep(65 if "429" in str(e) else 2 ** attempt)
    raise RuntimeError("unreachable")


def read_status() -> dict:
    from core.storage import kv_get
    try:
        return kv_get("external_status", {}) or {}
    except Exception:  # baza əlçatmazdırsa status oxunmasın, sistem dayanmasın
        return {}


def set_status(source: str, ok: bool, message: str = "") -> None:
    from core.storage import kv_set
    st = read_status()
    entry = st.get(source, {})
    now = datetime.now().isoformat(timespec="seconds")
    entry.update({"ok": ok, "checked_at": now, "message": message})
    if ok:
        entry["last_success"] = now
    st[source] = entry
    try:
        kv_set("external_status", st)
    except Exception as e:
        log.warning("Status yazılmadı: %s", e)
