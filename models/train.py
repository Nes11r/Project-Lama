"""Yekun modelin (yenidən) öyrədilməsi və versiyalanması.

İşə salma:
  python -m models.train                 # pipeline + hava yeniləməsi + təlim
  python -m models.train --skip-data     # mövcud emal olunmuş data ilə
  python -m models.train --offline       # API çağırmadan (yalnız keş)

Addımlar: təmizləmə → hava cədvəlləri → kontekst (HDD kalibrasiyası, klasterlər, statik) →
bütün yararlı data ilə kvantil LightGBM → intervalın konformal kalibrasiyası (backtest əsasında) → saxla.
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from core.config import load_config
from core.storage import has_table, kv_get, read_table
from features.build import feature_list, make_dataset
from features.context import build_context
from models import registry
from models.evaluate import ABLATION
from models.lgbm import QuantileForecaster

log = logging.getLogger(__name__)


def interval_scale_from_backtest(target: float = 0.8) -> float:
    """10–90% intervalını log-miqyasda median ətrafında genişləndirən əmsal (backtest örtüyü = hədəf)."""
    if not has_table("backtest"):
        return 1.0
    bt = read_table("backtest", ["q_actual", "q10", "q50", "q90"])
    y, m = np.log1p(bt.q_actual), np.log1p(bt.q50)
    lo, hi = np.log1p(bt.q10), np.log1p(bt.q90)
    best = 1.0
    for k in np.arange(0.8, 2.51, 0.02):
        cov = np.mean((y >= m - k * (m - lo)) & (y <= m + k * (hi - m)))
        best = k
        if cov >= target:
            break
    return round(float(best), 3)


def main(skip_data: bool = False, offline: bool = False, groups: list[str] | None = None) -> str:
    cfg = load_config()
    fc = cfg["forecast"]
    if not skip_data:
        from data.pipeline import run as run_pipeline
        from external.weather import build_tables
        run_pipeline()
        from data.pipeline import load_hourly
        build_tables(end=str(load_hourly().ts.max().date()), offline=offline)

    ctx = build_context("actual")
    ds = make_dataset(ctx, fc["horizon_h"], fc["train_origin_step_h"])
    tr = ds[ds.train_ok & ds.y.notna()]
    feats = feature_list(groups or ABLATION[-1])
    log.info("Təlim: %d sətir, %d xüsusiyyət", len(tr), len(feats))
    model = QuantileForecaster(feats, fc["quantiles"], fc["lightgbm"]).fit(tr, tr.y.to_numpy())

    cv = {}
    js = kv_get("model_metrics")
    if js:
        cv = {r["model"]: {k: round(r[k], 3) for k in ("MAE", "WAPE%", "RMSE")} for r in js.get("overall", [])}
    meta = {
        "train_start": str(tr.ts.min()), "train_end": str(tr.ts.max()), "n_train_rows": int(len(tr)),
        "feature_groups": groups or ABLATION[-1], "features": feats, "hdd_base": ctx.hdd_base,
        "static": ctx.static, "interval_scale": interval_scale_from_backtest(),
        "cluster_names": ctx.clusterer.names if ctx.clusterer else {}, "cv_summary": cv,
        "weather_training": "Open-Meteo archive (faktiki)",
    }
    d = registry.save(model, meta, ctx.clusterer)
    log.info("Model saxlanıldı: %s", d)
    return meta["version"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    print(main(skip_data=a.skip_data, offline=a.offline))
