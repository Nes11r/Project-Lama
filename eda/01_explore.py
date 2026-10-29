"""Mərhələ 1 — Data kəşfi (EDA).

İşə salma:  python eda/01_explore.py
Nəticələr:  reports/eda/  (mətn hesabatı + PNG qrafiklər)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
OUT = ROOT / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

STATIONS = {
    "merdekan": "Mərdəkan (01.01.2018 tarixdən).csv",
    "sumqayit4": "Sumqayıt Şəhər-4 (01.01.2018 tarixdən).csv",
    "turkan": "Türkan qəsəbəsi 01.01.2018 tarixdən.csv",
}
COLS = ["ts_raw", "sg", "dp_kpa", "p_kpa", "t_c", "q_hour", "q_day_rate"]

log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def load_station(fname: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / fname, header=0, names=COLS, dtype=str)
    is_tot = df["ts_raw"].str.startswith("Cəmi", na=False)
    # "Cəmi" sətri özündən əvvəlki günün ROC gündəlik cəmidir (min m³)
    df["ts"] = pd.to_datetime(df["ts_raw"], format="%d-%m-%Y %H:%M", errors="coerce")
    prev_ts = df["ts"].ffill().shift()
    totals = pd.DataFrame({
        "last_ts": prev_ts[is_tot],
        "total": df.loc[is_tot, "ts_raw"].str.extract(r"([\d.]+)")[0].astype(float),
    })
    df = df[~is_tot].copy()
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["on_hour"] = (df["ts"].dt.minute == 0)
    df.attrs["totals"] = totals
    return df


def describe_station(name: str, df: pd.DataFrame) -> pd.DataFrame:
    p(f"\n{'=' * 70}\nMƏNTƏQƏ: {name}\n{'=' * 70}")
    tot = df.attrs["totals"]
    p(f"Sətir sayı: {len(df)} | 'Cəmi' (gündəlik cəm) sətirləri: {len(tot)}; ilki {tot['last_ts'].min()}")
    off = df[~df["on_hour"] & df["ts"].notna()]
    p(f"Tam saata düşməyən damğalar: {len(off)}  nümunə: {list(off['ts_raw'].head(5))}")
    # Gündəlik cəm vs saatlıq cəm
    h = df[df["on_hour"]].copy()
    h["gas_day"] = (h["ts"] - pd.Timedelta("1h")).dt.normalize()
    agg = h.groupby("gas_day")["q_hour"].agg(["sum", "count"])
    tot = tot.assign(gas_day=(tot["last_ts"] - pd.Timedelta("1h")).dt.normalize()).set_index("gas_day")
    j = tot.join(agg, how="inner")
    full = j[j["count"] == 24]
    p(f"Tam günlərdə (24 saat) |Cəmi − Σsaatlıq|: median={np.median(np.abs(full['total']-full['sum'])):.3f}, "
      f"max={np.abs(full['total']-full['sum']).max() if len(full) else float('nan'):.3f}, n={len(full)}")
    part = j[j["count"] < 24]
    p(f"Natamam günlər: {len(part)}; Cəmi/Σ orta nisbəti={np.nanmean(part['total']/part['sum']):.3f}")
    p("Gündə saat sayı paylanması (saat-sonu konvensiyası ilə):\n" + agg["count"].value_counts().head(8).to_string())
    p(f"Parse olunmayan tarix: {df['ts'].isna().sum()}")
    p(f"Dövr: {df['ts'].min()}  →  {df['ts'].max()}")

    # Monotonluq və təkrarlar
    dup = df["ts"].duplicated(keep=False)
    p(f"Təkrar zaman damğası sətirləri: {dup.sum()}")
    if dup.any():
        p(df.loc[dup, ["ts_raw", "q_hour", "p_kpa"]].head(12).to_string())
    non_mono = (df["ts"].diff() <= pd.Timedelta(0)).sum()
    p(f"Geriyə gedən/eyni addımlar: {non_mono}")

    # Addım paylanması
    steps = df["ts"].diff().value_counts().head(10)
    p("Addım (diff) paylanması:\n" + steps.to_string())

    # Boşluqlar
    full = pd.date_range(df["ts"].min().floor("h"), df["ts"].max(), freq="h")
    missing = full.difference(df.loc[df["on_hour"], "ts"].dropna())
    p(f"Gözlənilən saat: {len(full)} | Çatışmayan saat: {len(missing)} ({len(missing) / len(full):.2%})")
    if len(missing):
        gaps = pd.Series(missing).diff().ne(pd.Timedelta("1h")).cumsum()
        g = pd.Series(missing).groupby(gaps.values).agg(["min", "max", "count"])
        g = g.sort_values("count", ascending=False)
        p("Ən böyük boşluqlar (saat):\n" + g.head(15).to_string())
        # DST yoxlaması: Azərbaycan 2016-dan DST istifadə etmir; mart/okt son bazar günləri
        p("Mart/Oktyabr son bazar günü ətrafındakı çatışmayan saatlar:")
        for y in range(2018, 2025):
            for m in (3, 10):
                d = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
                d = d - pd.Timedelta(days=(d.weekday() + 1) % 7)
                win = missing[(missing >= d) & (missing < d + pd.Timedelta(days=1))]
                if len(win):
                    p(f"   {d.date()}: {list(win.strftime('%H:%M'))}")

    # Hər sütun üzrə NaN
    p("NaN sayı (sütunlar üzrə):\n" + df[COLS[1:]].isna().sum().to_string())
    p("Statistik xülasə:\n" + df[COLS[1:]].describe(percentiles=[.01, .05, .5, .95, .99]).T.round(3).to_string())

    # Vahid əlaqəsi: q_day_rate == q_hour*24 ?
    ratio = (df["q_day_rate"] / df["q_hour"]).replace([np.inf, -np.inf], np.nan)
    p(f"q_day_rate/q_hour: median={ratio.median():.3f}, 1%={ratio.quantile(.01):.3f}, 99%={ratio.quantile(.99):.3f}")

    # Sıfır/mənfi və sabit (donmuş) dəyərlər
    p(f"q_hour == 0: {(df['q_hour'] == 0).sum()} | < 0: {(df['q_hour'] < 0).sum()}")
    p(f"dp_kpa <= 0: {(df['dp_kpa'] <= 0).sum()} | p_kpa <= 0: {(df['p_kpa'] <= 0).sum()}")
    for c in ["q_hour", "p_kpa", "dp_kpa", "t_c"]:
        s = df[c]
        run = s.ne(s.shift()).cumsum()
        runs = s.groupby(run).agg(["first", "size"])
        long = runs[runs["size"] >= 6]
        p(f"{c}: ≥6 saat eyni qalan (donmuş) seqment sayı={len(long)}, cəm saat={long['size'].sum()}")
        if len(long):
            p("   ən uzunları: " + ", ".join(f"{v:.3f}×{n}" for v, n in long.sort_values('size', ascending=False).head(5).values))

    # Statistik outlier (rolling median ± 6·MAD)
    s = df[df["on_hour"]].drop_duplicates("ts").set_index("ts")["q_hour"]
    med = s.rolling(24 * 7, center=True, min_periods=24).median()
    mad = (s - med).abs().rolling(24 * 7, center=True, min_periods=24).median()
    z = (s - med) / (1.4826 * mad.replace(0, np.nan))
    out = z.abs() > 6
    p(f"q_hour robust-outlier (|z|>6, həftəlik pəncərə): {out.sum()}")
    if out.any():
        p(s[out].sort_values().head(8).to_string())
        p(s[out].sort_values().tail(8).to_string())
    # Kompensasiya fiziki yoxlaması: Q ~ k*sqrt(dP*P/T)
    phys = np.sqrt(df["dp_kpa"].clip(lower=0) * df["p_kpa"] / (df["t_c"] + 273.15))
    corr = df["q_hour"].corr(phys)
    p(f"corr(q_hour, sqrt(dP·P/T)) = {corr:.4f}  (orifice tənliyi ilə uyğunluq)")
    return df


def plots(name: str, df: pd.DataFrame, wx: pd.DataFrame):
    s = df[df["on_hour"]].drop_duplicates("ts").set_index("ts").asfreq("h")
    fig, ax = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    for a, (c, lbl) in zip(ax, [("q_hour", "Saatlıq sərf, min m³"), ("p_kpa", "Təzyiq, kPa"),
                                ("dp_kpa", "ΔP, kPa"), ("t_c", "Qaz temperaturu, °C"), ("sg", "Xüsusi çəki, kq/m³")]):
        a.plot(s.index, s[c], lw=0.3, color="#1f4e79")
        a.plot(s[c].rolling(24 * 7, min_periods=24).mean(), lw=1.2, color="#c0504d")
        a.set_ylabel(lbl, fontsize=8)
        a.grid(alpha=.3)
    ax[0].set_title(f"{name}: saatlıq xam sıralar (qırmızı — 7 günlük ortalama)")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_01_timeseries.png", dpi=110)
    plt.close(fig)

    q = s["q_hour"]
    # Gündəlik profil mövsüm üzrə
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    season = q.index.month.map({12: "Qış", 1: "Qış", 2: "Qış", 3: "Yaz", 4: "Yaz", 5: "Yaz",
                                6: "Yay", 7: "Yay", 8: "Yay", 9: "Payız", 10: "Payız", 11: "Payız"})
    prof = q.groupby([season, q.index.hour]).mean().unstack(0)
    prof.plot(ax=ax[0], marker=".")
    ax[0].set_title("Gündəlik profil (saat üzrə orta)")
    ax[0].set_xlabel("Saat")
    # Həftəlik
    wk = q.groupby([season, q.index.dayofweek]).mean().unstack(0)
    wk.index = ["B.e", "Ç.a", "Ç", "C.a", "C", "Ş", "B"]
    wk.plot(ax=ax[1], marker="o")
    ax[1].set_title("Həftənin günü üzrə orta")
    # İllik
    d = q.resample("D").sum(min_count=20)
    for y, g in d.groupby(d.index.year):
        ax[2].plot(g.index.dayofyear, g.rolling(7, min_periods=3).mean(), lw=1, label=str(y))
    ax[2].set_title("Gündəlik cəm (7 günlük ortalama) — illər üzrə")
    ax[2].set_xlabel("İlin günü")
    ax[2].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_02_seasonality.png", dpi=110)
    plt.close(fig)

    # Heatmap saat × ay
    hm = q.groupby([q.index.month, q.index.hour]).mean().unstack()
    fig, a = plt.subplots(figsize=(10, 4.5))
    im = a.imshow(hm.values, aspect="auto", cmap="viridis")
    a.set_yticks(range(12), [str(m) for m in hm.index])
    a.set_xlabel("Saat")
    a.set_ylabel("Ay")
    a.set_title(f"{name}: orta saatlıq sərf (ay × saat)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_03_heatmap.png", dpi=110)
    plt.close(fig)

    # Hava ilə əlaqə (günlük)
    dd = pd.DataFrame({"q": d}).join(wx[["tavg"]], how="inner").dropna()
    fig, a = plt.subplots(figsize=(7, 5))
    sc = a.scatter(dd["tavg"], dd["q"], s=4, c=dd.index.year, cmap="tab10", alpha=.6)
    a.set_xlabel("Hava temperaturu (gündəlik orta), °C")
    a.set_ylabel("Gündəlik sərf, min m³")
    a.set_title(f"{name}: sərf vs hava temperaturu  (r={dd['q'].corr(dd['tavg']):.2f})")
    fig.colorbar(sc, label="il")
    a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_04_vs_temp.png", dpi=110)
    plt.close(fig)
    return d


def hdd_breakpoint(d: pd.Series, wx: pd.DataFrame) -> tuple[float, float]:
    """Gündəlik sərf = a + b·max(0, Tb − T): ən yaxşı Tb-ni axtar."""
    dd = pd.DataFrame({"q": d}).join(wx[["tavg"]], how="inner").dropna()
    best = (None, -1)
    for tb in np.arange(10, 24.5, 0.5):
        x = np.clip(tb - dd["tavg"], 0, None)
        r2 = np.corrcoef(x, dd["q"])[0, 1] ** 2
        if r2 > best[1]:
            best = (tb, r2)
    return best


def main():
    wx = pd.read_csv(DATA / "sumqayit hava 2018-2024.csv", encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
    p("=" * 70 + "\nHAVA FAYLI (Meteostat formatı, günlük)\n" + "=" * 70)
    p(f"Dövr: {wx.index.min().date()} → {wx.index.max().date()} | sətir: {len(wx)}")
    full = pd.date_range(wx.index.min(), wx.index.max(), freq="D")
    p(f"Çatışmayan gün: {len(full.difference(wx.index))} | təkrar: {wx.index.duplicated().sum()}")
    p("NaN payı:\n" + wx.isna().mean().round(3).to_string())
    p(wx.describe().T.round(2).to_string())

    daily = {}
    frames = {}
    for name, f in STATIONS.items():
        df = load_station(f)
        frames[name] = describe_station(name, df)
        daily[name] = plots(name, df, wx)
        tb, r2 = hdd_breakpoint(daily[name], wx)
        p(f"HDD baza temperaturu (ən yaxşı uyğunluq): Tb={tb}°C, R²={r2:.3f}")

    # Mərdəkan vs Türkan: eyni SG və təzyiq?
    u = {k: v[v["on_hour"]].drop_duplicates("ts").set_index("ts") for k, v in frames.items()}
    a, b = u["merdekan"], u["turkan"]
    j = a.join(b, lsuffix="_m", rsuffix="_t", how="inner")
    p("\nMərdəkan ↔ Türkan müqayisəsi:")
    p(f"  SG eyni olan sətir payı: {(j['sg_m'] == j['sg_t']).mean():.3f}")
    p(f"  corr(P): {j['p_kpa_m'].corr(j['p_kpa_t']):.3f} | corr(q): {j['q_hour_m'].corr(j['q_hour_t']):.3f}")
    p(f"  corr(q, sumqayit4): M={u['merdekan']['q_hour'].corr(u['sumqayit4']['q_hour']):.3f}")

    # Ümumi müqayisə qrafiki
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for n, d in daily.items():
        ax[0].plot(d.rolling(7, min_periods=3).mean(), lw=1, label=n)
    ax[0].set_ylabel("Gündəlik sərf, min m³ (7g ort.)")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot(wx["tavg"].rolling(7, min_periods=3).mean(), color="#c0504d")
    ax[1].set_ylabel("Hava tavg, °C (7g ort.)")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "all_stations_daily.png", dpi=110)
    plt.close(fig)

    (OUT / "eda_report.txt").write_text(log.getvalue(), encoding="utf-8")
    p(f"\nHesabat: {OUT / 'eda_report.txt'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
