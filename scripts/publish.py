"""Lokal maşında hazırlanmış hər şeyi saxlama qatına (Neon) yükləyir.

İşə salma (DATABASE_URL .env-də olmalıdır):
  python scripts/publish.py              # xam CSV + pipeline + hava + backtest + təlim
  python scripts/publish.py --eval       # əvvəlcə tam walk-forward qiymətləndirmə də (≈30 dəq)
  python scripts/publish.py --no-train   # yalnız data

Bulud konteyneri (Render) ağır iş görmür: yalnız bu skriptin yazdıqlarını oxuyur.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from core.config import path  # noqa: E402
from core.storage import blob_put, describe, has_table, kv_set, using_db, write_table  # noqa: E402

log = logging.getLogger("publish")


def main(run_eval: bool, train: bool, offline_weather: bool):
    if not using_db():
        sys.exit("DATABASE_URL təyin olunmayıb (.env). Lokal rejimdə bu skriptə ehtiyac yoxdur.")
    log.info("Hədəf: %s", describe())

    # 1. Xam ROC CSV-lər (bulud konteynerində dataset/ qovluğu yoxdur)
    for sub, d in (("raw", path("raw_dir")), ("raw/incoming", path("incoming_dir"))):
        for f in sorted(d.glob("*.csv")):
            blob_put(f"{sub}/{f.name}", f.read_bytes())
            log.info("Fayl yükləndi: %s/%s (%.1f MB)", sub, f.name, f.stat().st_size / 1e6)

    # 2. Təmizləmə → "hourly", "daily_totals"
    from data.pipeline import run as run_pipeline
    hourly = run_pipeline()

    # 3. Hava cədvəlləri (lokal keşdən; çatışmayan illər API-dən)
    from external.weather import build_tables
    build_tables(end=str(hourly.ts.max().date()), offline=offline_weather)

    # 4. Statik xüsusiyyətlər keşi lokal qalır; modelin meta.json-una yazılır

    # 5. Backtest
    if run_eval:
        from models.evaluate import run as run_evaluate
        run_evaluate(quick=False)
    else:
        bt = path("processed") / "backtest.parquet"
        if bt.exists():
            from models.metrics import accuracy_summary
            df = pd.read_parquet(bt)
            if not has_table("backtest"):
                write_table("backtest", df)
            kv_set("accuracy", accuracy_summary(df))  # API backtest cədvəlini yox, bu xülasəni oxuyur
        mf = path("reports") / "model" / "metrics.json"
        if mf.exists():
            kv_set("model_metrics", json.loads(mf.read_text(encoding="utf-8")))

    # 6. Yekun model
    if train:
        from models.train import main as train_main
        v = train_main(skip_data=True)
        log.info("Aktiv model: %s", v)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--offline-weather", action="store_true")
    a = ap.parse_args()
    main(a.eval, not a.no_train, a.offline_weather)
