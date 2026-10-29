"""ROC flow computer CSV ixraclarının oxunması.

Fayl formatı (Mərhələ 1-də müəyyən edilib):
  TARİX, XÜSUSİ ÇƏKİ (kq/m3), TƏZYİQLƏR FƏRQİ (kPa), TƏZYİQ (kPa), TEMPERATUR (C),
  SAATLIQ SƏRF (min m3), SƏRF (min m3)  [= saatlıq × 24, törəmə sütun]
- Damğa "dd-mm-YYYY HH:MM", saat-sonu: "01:00" = 00:00–01:00 intervalı.
- Faylın bir hissəsi gündəlik hesabatlardan ibarətdir: hər günün sonunda "Cəmi: X" sətri
  (ROC gündəlik akkumulyatoru, min m³) gəlir.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import pandas as pd

from core.config import path, stations

RAW_COLS = ["ts_raw", "sg", "dp_kpa", "p_kpa", "t_c", "q_hour", "q_day_rate"]
NUM_COLS = RAW_COLS[1:]


def parse_roc_file(file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bir ROC CSV faylını oxuyur.

    Qaytarır:
      rows   — ts (interval başlanğıcı), ts_end (ROC damğası), on_hour, ədədi sütunlar, source
      totals — gas_day, total (ROC gündəlik cəmi, min m³)
    """
    df = pd.read_csv(file, header=0, names=RAW_COLS, dtype=str, encoding="utf-8-sig")
    df["ts_raw"] = df["ts_raw"].str.strip()
    is_tot = df["ts_raw"].str.startswith("Cəmi", na=False)
    ts_end = pd.to_datetime(df["ts_raw"], format="%d-%m-%Y %H:%M", errors="coerce")

    # "Cəmi" özündən əvvəlki son damğanın qaz gününə aiddir
    prev_end = ts_end.where(~is_tot).ffill().shift().where(is_tot)
    totals = pd.DataFrame({
        "gas_day": (prev_end[is_tot] - pd.Timedelta("1min")).dt.normalize(),
        "total": pd.to_numeric(df.loc[is_tot, "ts_raw"].str.extract(r"([-\d.]+)")[0], errors="coerce"),
    }).dropna()

    rows = df.loc[~is_tot].copy()
    rows["ts_end"] = ts_end[~is_tot]
    rows = rows.dropna(subset=["ts_end"])
    for c in NUM_COLS:
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    rows["on_hour"] = rows["ts_end"].dt.minute.eq(0)
    rows["ts"] = rows["ts_end"] - pd.Timedelta("1h")  # interval başlanğıcı
    rows["source"] = file.name
    return rows.drop(columns=["ts_raw", "q_day_rate"]).reset_index(drop=True), totals


def _blob_raw_files() -> list[Path]:
    """Saxlama qatındakı xam CSV-lər ("raw/..." açarları) — bulud konteynerində dataset/ qovluğu yoxdur."""
    from core.storage import LOCAL, blob_get, blob_list, using_db
    if not using_db():
        return []
    out = []
    for b in blob_list("raw/"):
        dest = LOCAL / "raw_cache" / b["key"]
        if not dest.exists() or dest.stat().st_size != b["size"]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob_get(b["key"]))
        out.append(dest)
    return out


def station_files(station_id: str) -> list[Path]:
    """Əsas fayllar əvvəl, incoming sonra (sonrakılar üstünlük alır). Eyni adlı fayl bir dəfə götürülür."""
    cfg = stations()[station_id]
    base, incoming = {}, {}
    for f in [*sorted(path("raw_dir").glob("*.csv")),
              *[p for p in _blob_raw_files() if p.parent.name != "incoming"]]:
        base.setdefault(f.name, f)
    for f in [*sorted(path("incoming_dir").glob("*.csv")),
              *[p for p in _blob_raw_files() if p.parent.name == "incoming"]]:
        incoming.setdefault(f.name, f)
    files = [*base.values(), *sorted(incoming.values(), key=lambda p: p.name)]
    return [f for f in files if any(fnmatch.fnmatch(f.name, pat) for pat in cfg["file_patterns"])]


def load_station_raw(station_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Məntəqənin bütün fayllarını (əsas + incoming) birləşdirir. Sonrakı fayllar üstünlük alır."""
    parts, tots = [], []
    for i, f in enumerate(station_files(station_id)):
        r, t = parse_roc_file(f)
        r["file_order"] = i
        t["file_order"] = i
        parts.append(r)
        tots.append(t)
    if not parts:
        raise FileNotFoundError(f"{station_id}: heç bir fayl tapılmadı")
    rows = pd.concat(parts, ignore_index=True)
    totals = (pd.concat(tots, ignore_index=True)
              .sort_values(["gas_day", "file_order"])
              .drop_duplicates("gas_day", keep="last")
              .drop(columns="file_order"))
    return rows, totals.reset_index(drop=True)
