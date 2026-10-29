"""Təmizləmə pipeline-ı: xam ROC CSV → "hourly" cədvəli (Neon və ya storage/processed/hourly.parquet)

İşə salma:  python -m data.pipeline
Yeni CSV `dataset/incoming/`-ə atıldıqda eyni əmr yenidən işə salınır (idempotentdir).
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime

import pandas as pd

from core.config import load_config, load_yaml, path, stations
from core.storage import has_table, kv_set, read_table, write_table
from data.cleaning import clean_station
from data.loader import load_station_raw

log = logging.getLogger(__name__)


def run() -> pd.DataFrame:
    cfg = load_config()
    events = load_yaml("events").get("events", []) or []
    frames, reports, totals_all = [], [], []
    for sid in stations():
        rows, totals = load_station_raw(sid)
        df, rep = clean_station(sid, rows, totals, cfg["cleaning"], events)
        frames.append(df)
        reports.append(rep)
        totals_all.append(totals.assign(station=sid))
        log.info("%s: %s saat, train_ok=%.1f%%", sid, rep["hours_total"], 100 * rep["train_ok_share"])

    hourly = pd.concat(frames, ignore_index=True)
    write_table("hourly", hourly)
    write_table("daily_totals", pd.concat(totals_all, ignore_index=True))

    rep_dir = path("reports")
    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"), "stations": reports}
    (rep_dir / "cleaning_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    kv_set("cleaning_report", payload)
    return hourly


def load_hourly(start=None, end=None) -> pd.DataFrame:
    if not has_table("hourly"):
        df = run()
        return df if start is None and end is None else df[df.ts.between(start or df.ts.min(), end or df.ts.max())]
    df = read_table("hourly", start=start, end=end)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    run()
    print((path("reports") / "cleaning_report.json").read_text(encoding="utf-8"))
