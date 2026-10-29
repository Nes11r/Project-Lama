"""Kvantil LightGBM: hər kvantil üçün ayrıca model, hədəf log1p(q) miqyasında.

Monoton çevirmə kvantilləri saxlayır: Q_τ(q) = expm1(Q_τ(log1p q)).
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.build import CATEGORICAL


class QuantileForecaster:
    def __init__(self, features: list[str], quantiles=(0.1, 0.5, 0.9), params: dict | None = None):
        self.features = list(features)
        self.quantiles = list(quantiles)
        self.params = dict(params or {})
        self.models: dict[float, lgb.Booster] = {}

    def _ds(self, X: pd.DataFrame, y=None):
        cats = [c for c in CATEGORICAL if c in self.features]
        return lgb.Dataset(X[self.features], label=y, categorical_feature=cats, free_raw_data=False)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "QuantileForecaster":
        p = dict(self.params)
        n_est = p.pop("n_estimators", 500)  # qalan sklearn-üslublu adlar LightGBM alias-larıdır
        ds = self._ds(X, y)
        for q in self.quantiles:
            params = {"objective": "quantile", "alpha": q, "seed": 42, "num_threads": 0, **p}
            self.models[q] = lgb.train(params, ds, num_boost_round=n_est)
        return self

    def matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Model girişi: bütün sütunlar ədədi (object → float, NaN saxlanılır)."""
        M = X[self.features]
        obj = M.columns[M.dtypes == object]
        if len(obj):
            M = M.copy()
            M[obj] = M[obj].apply(pd.to_numeric, errors="coerce")
        return M

    def predict_log(self, X: pd.DataFrame) -> pd.DataFrame:
        M = self.matrix(X)
        out = pd.DataFrame({f"q{int(q * 100)}": m.predict(M) for q, m in self.models.items()}, index=X.index)
        # Kvantil kəsişməsinin qarşısını al
        out[:] = np.sort(out.to_numpy(), axis=1)
        return out

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.expm1(self.predict_log(X)).clip(lower=0)

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        for q, m in self.models.items():
            m.save_model(str(d / f"lgbm_q{int(q * 100)}.txt"))
        (d / "forecaster.json").write_text(json.dumps(
            {"features": self.features, "quantiles": self.quantiles, "params": self.params}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "QuantileForecaster":
        meta = json.loads((d / "forecaster.json").read_text(encoding="utf-8"))
        f = cls(meta["features"], meta["quantiles"], meta["params"])
        for q in f.quantiles:
            f.models[q] = lgb.Booster(model_file=str(d / f"lgbm_q{int(q * 100)}.txt"))
        return f
