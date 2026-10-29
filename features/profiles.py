"""Gündəlik yük profillərinin klasterləşdirilməsi ("iş günü qış", "bayram", "yay həftəsonu" və s.).

Forma = saatlıq sərf / gündəlik orta (24 ölçülü vektor) → KMeans. Klaster yalnız TƏLİM dövrü üzrə
öyrədilir. Proqnoz üçün xüsusiyyət: başlanğıcdan əvvəlki TAM günün klasteri (sızma yoxdur).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SEASON = {12: "qış", 1: "qış", 2: "qış", 3: "yaz", 4: "yaz", 5: "yaz",
          6: "yay", 7: "yay", 8: "yay", 9: "payız", 10: "payız", 11: "payız"}


def daily_shapes(hourly: pd.DataFrame) -> pd.DataFrame:
    """hourly: station, ts, q_filled → (station, date) × 24 forma matrisi."""
    h = hourly[["station", "ts", "q_filled"]].copy()
    h["date"] = h["ts"].dt.normalize()
    h["hour"] = h["ts"].dt.hour
    m = h.pivot_table(index=["station", "date"], columns="hour", values="q_filled")
    m = m.dropna()
    mean = m.mean(axis=1)
    m = m[mean > 0]
    return m.div(m.mean(axis=1), axis=0)


class ProfileClusterer:
    def __init__(self, k: int = 6, seed: int = 0):
        self.k = k
        self.km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        self.names: dict[int, str] = {}

    def fit(self, hourly: pd.DataFrame, cutoff: pd.Timestamp, calendar: pd.DataFrame | None = None):
        s = daily_shapes(hourly[hourly["ts"] < cutoff])
        self.km.fit(s.to_numpy())
        lab = pd.Series(self.km.labels_, index=s.index)
        # Ad: klasterdəki günlərin ən çox rast gəlinən mövsümü + gün tipi
        dates = lab.index.get_level_values("date")
        season = pd.Series(dates.month.map(SEASON), index=lab.index)
        if calendar is not None:
            nw = calendar["is_nonworking"].reindex(dates).fillna(False).to_numpy()
        else:
            nw = dates.dayofweek >= 5
        dtype = pd.Series(np.where(nw, "qeyri-iş günü", "iş günü"), index=lab.index)
        for c in range(self.k):
            m = lab == c
            se = season[m].value_counts(normalize=True)
            dt = dtype[m].value_counts(normalize=True)
            self.names[c] = f"{se.index[0]} ({se.iloc[0]:.0%}), {dt.index[0]} ({dt.iloc[0]:.0%})"
        return self

    def transform(self, hourly: pd.DataFrame) -> pd.DataFrame:
        s = daily_shapes(hourly)
        lab = pd.Series(self.km.predict(s.to_numpy()), index=s.index, name="profile_cluster")
        return lab.reset_index()

    def describe(self) -> pd.DataFrame:
        c = pd.DataFrame(self.km.cluster_centers_)
        c.insert(0, "ad", [self.names.get(i, "") for i in range(self.k)])
        c["pik/min"] = c.iloc[:, 1:25].max(axis=1) / c.iloc[:, 1:25].min(axis=1)
        c["axşam_piki_saat"] = c.iloc[:, 1:25].idxmax(axis=1)
        return c[["ad", "pik/min", "axşam_piki_saat"]]
