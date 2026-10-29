"""Dəqiqlik metrikləri (orijinal miqyasda, min m³/saat)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    e = p - y
    nz = np.abs(y) > 1e-6
    return {
        "n": int(len(y)),
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAPE%": float(100 * np.mean(np.abs(e[nz]) / np.abs(y[nz]))),
        "sMAPE%": float(100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p) + 1e-9))),
        "WAPE%": float(100 * np.sum(np.abs(e)) / np.sum(np.abs(y))),
        "bias%": float(100 * np.sum(e) / np.sum(np.abs(y))),
    }


def interval_metrics(y, lo, hi, alpha: float = 0.2) -> dict:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[m], lo[m], hi[m]
    cover = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    # Winkler / interval score
    score = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return {"coverage%": 100 * cover, "mean_width": width, "interval_score": float(np.mean(score))}


def pinball(y, q, tau: float) -> float:
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.nanmean(np.maximum(tau * d, (tau - 1) * d)))


def table(df: pd.DataFrame, pred_cols: dict[str, str], by: list[str] | None = None) -> pd.DataFrame:
    """pred_cols: ad → sütun. Hər model üçün metrikləri (qrup üzrə) qaytarır."""
    rows = []
    groups = [((), df)] if not by else df.groupby(by)
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for name, col in pred_cols.items():
            r = {"model": name, **metrics(g["q_actual"], g[col])}
            for k, v in zip(by or [], key):
                r[k] = v
            rows.append(r)
    out = pd.DataFrame(rows)
    return out[(by or []) + [c for c in out.columns if c not in (by or [])]]


def accuracy_summary(bt: pd.DataFrame) -> dict:
    """Backtest → məntəqə üzrə panel cədvəlləri (fold, üfüq, ay). API bunu kv-dən oxuyur ki,
    bulud konteyneri böyük backtest cədvəlini yaddaşa yükləməsin."""
    bt = bt.assign(origin=pd.to_datetime(bt["origin"]))
    out = {}
    for sid, s in bt.groupby("station"):
        folds = []
        for fold, g in s.groupby("fold"):
            folds.append({"fold": int(fold), "period": f"{g.origin.min():%Y-%m} – {g.origin.max():%Y-%m}",
                          **{f"lgbm_{k}": v for k, v in metrics(g.q_actual, g.q50).items()},
                          **{f"snaive_{k}": v for k, v in metrics(g.q_actual, g.bl_q_lag24).items()},
                          **interval_metrics(g.q_actual, g.q10, g.q90)})
        by_h = [{"h": int(h), "MAE": m["MAE"], "WAPE%": m["WAPE%"]}
                for h, g in s.groupby("h") for m in [metrics(g.q_actual, g.q50)]]
        by_month = [{"m": int(mo), "WAPE%": metrics(g.q_actual, g.q50)["WAPE%"]}
                    for mo, g in s.groupby(s.origin.dt.month)]
        out[str(sid)] = {"folds": folds, "by_horizon": by_h, "by_month": by_month}
    return _finite(out)


def _finite(o):
    """JSONB NaN qəbul etmir → None."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o
