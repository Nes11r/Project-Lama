"""Walk-forward qiymətləndirmə, baseline müqayisəsi, ablation, sızma testi və SHAP.

İşə salma:  python -m models.evaluate            (tam)
            python -m models.evaluate --quick    (ablation-sız)
Nəticələr:  reports/model/  və  storage/processed/backtest.parquet

Sxem (sızmaya qarşı):
  - HDD bazası və profil klasterləri yalnız İLK fold başlanğıcından əvvəlki data ilə öyrədilir;
  - hər fold-da model yalnız hədəf vaxtı fold başlanğıcından əvvəl olan sətirlərlə öyrədilir;
  - test sətirlərində hava = 1 gün əvvəl verilmiş PROQNOZ (Open-Meteo Previous Runs);
    müqayisə üçün faktiki hava ilə də hesablanır (optimist yuxarı hədd).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import load_config, path
from core.storage import kv_set, write_table
from features.build import FEATURE_GROUPS, feature_list, make_dataset
from features.context import build_context, weather_table
from models.baselines import add_baselines, baseline_cols
from models.lgbm import QuantileForecaster
from models.metrics import interval_metrics, metrics, table

log = logging.getLogger(__name__)
ABLATION = [["base"], ["base", "weather"], ["base", "weather", "calendar"],
            ["base", "weather", "calendar", "profile"],
            ["base", "weather", "calendar", "profile", "population"]]
GROUP_LABELS = {"base": "Sərf tarixçəsi + saat/gün", "weather": "+ Hava", "calendar": "+ Təqvim/bayram/günəş",
                "profile": "+ Profil klasterləri", "population": "+ Əhali/OSM"}


def datasets(cutoff: pd.Timestamp, step: int):
    ctx_a = build_context("actual", cutoff=cutoff)
    ds_a = make_dataset(ctx_a, 24, step)
    ctx_f = build_context("actual", cutoff=cutoff, hourly=ctx_a.hourly, with_clusters=False, with_static=False)
    ctx_f.weather = weather_table("forecast")
    ctx_f.hdd_base, ctx_f.static, ctx_f.clusterer = ctx_a.hdd_base, ctx_a.static, ctx_a.clusterer
    ctx_f._clusters = ctx_a.clusters()
    ds_f = make_dataset(ctx_f, 24, step)
    return ctx_a, ds_a, ds_f


def run(quick: bool = False) -> dict:
    cfg = load_config()
    fc = cfg["forecast"]
    folds = [(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta("1D")) for a, b in fc["cv_folds"]]
    out_dir = path("reports") / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ctx, ds_a, ds_f = datasets(folds[0][0], fc["train_origin_step_h"])
    log.info("Dataset: %d sətir (%.0fs)", len(ds_a), time.time() - t0)
    full = feature_list(ABLATION[-1])
    params = fc["lightgbm"]

    preds, ablation_rows, models = [], [], {}
    for i, (start, end) in enumerate(folds, 1):
        tr = ds_a[(ds_a.ts < start) & ds_a.train_ok & ds_a.y.notna()]
        te_mask = lambda d: (d.origin >= start) & (d.origin < end) & d.train_ok & ~d.event & d.y.notna()
        te_f, te_a = ds_f[te_mask(ds_f)], ds_a[te_mask(ds_a)]
        t1 = time.time()
        m = QuantileForecaster(full, fc["quantiles"], params).fit(tr, tr.y.to_numpy())
        models[i] = m
        pf, pa = m.predict(te_f), m.predict(te_a)
        res = add_baselines(te_f[["station", "ts", "origin", "h", "q_actual", "lag_last", "q_lag24",
                                  "q_lag168", "same_hour_mean7"]])
        res = res.join(pf).join(pa["q50"].rename("q50_actual_wx"))
        res["fold"] = i
        preds.append(res)
        log.info("Fold %d [%s → %s]: train=%d test=%d, %.0fs", i, start.date(), end.date(), len(tr), len(te_f),
                 time.time() - t1)

        if not quick:
            for groups in ABLATION[:-1]:
                mm = QuantileForecaster(feature_list(groups), [0.5], params).fit(tr, tr.y.to_numpy())
                p = mm.predict(te_f)["q50"]
                ablation_rows.append({"fold": i, "set": groups[-1], **metrics(te_f.q_actual, p)})
            ablation_rows.append({"fold": i, "set": "population", **metrics(te_f.q_actual, pf["q50"])})

    bt = pd.concat(preds, ignore_index=True)
    write_table("backtest", bt)

    # --- Cədvəllər
    cols = {**baseline_cols(), "LightGBM (proqnoz hava)": "q50", "LightGBM (faktiki hava)": "q50_actual_wx"}
    overall = table(bt, cols)
    by_station = table(bt, cols, by=["station"])
    by_h = bt.assign(hb=pd.cut(bt.h, [0, 1, 3, 6, 12, 24], labels=["1", "2-3", "4-6", "7-12", "13-24"])) \
        .groupby("hb", observed=True).apply(lambda g: pd.Series({
            "MAE_lgbm": metrics(g.q_actual, g.q50)["MAE"], "MAE_snaive24": metrics(g.q_actual, g.bl_q_lag24)["MAE"],
            "WAPE_lgbm%": metrics(g.q_actual, g.q50)["WAPE%"]}))
    by_fold = table(bt, {"LightGBM": "q50", "seasonal naive 24s": "bl_q_lag24"}, by=["fold", "station"])
    intervals = pd.DataFrame([{"station": s, **interval_metrics(g.q_actual, g.q10, g.q90)}
                              for s, g in bt.groupby("station")])

    # Növbəti 24 saatın CƏMİ (operator üçün): tam pəncərəli başlanğıclar
    w = bt.groupby(["station", "fold", "origin"]).agg(n=("h", "size"), act=("q_actual", "sum"),
                                                      lgbm=("q50", "sum"), sn=("bl_q_lag24", "sum"))
    w = w[w.n == 24]
    day_tot = pd.DataFrame([{"station": s, "LightGBM WAPE%": metrics(g.act, g.lgbm)["WAPE%"],
                             "seasonal naive WAPE%": metrics(g.act, g.sn)["WAPE%"], "n": len(g)}
                            for s, g in w.reset_index().groupby("station")])

    # Sızma testi: 2024 fold-unda bütün hava dəyişənləri real proqnozdur
    last = bt[bt.fold == len(folds)]
    leak = pd.DataFrame([{"hava": "faktiki (sızma, optimist)", **metrics(last.q_actual, last.q50_actual_wx)},
                         {"hava": "1 gün əvvəlki proqnoz (real şərait)", **metrics(last.q_actual, last.q50)}])

    result = {"overall": overall, "by_station": by_station, "by_horizon": by_h.reset_index(),
              "by_fold": by_fold, "intervals": intervals, "next24_total": day_tot, "leakage_2024": leak}

    if ablation_rows:
        ab = pd.DataFrame(ablation_rows).groupby("set", sort=False)[["MAE", "RMSE", "WAPE%", "sMAPE%"]].mean()
        ab.index = [GROUP_LABELS[s] for s in ab.index]
        ab["ΔWAPE vs əvvəlki"] = ab["WAPE%"].diff()
        result["ablation"] = ab.reset_index(names="xüsusiyyət dəsti")

    # --- SHAP (son fold modeli; LightGBM daxili TreeSHAP = pred_contrib)
    try:
        m = models[len(folds)]
        te = ds_f[(ds_f.origin >= folds[-1][0]) & ds_f.train_ok & ds_f.y.notna()].sample(4000, random_state=0)
        sv = m.models[0.5].predict(te[m.features], pred_contrib=True)[:, :-1]
        imp = pd.Series(np.abs(sv).mean(0), index=m.features).sort_values(ascending=False)
        grp = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
        gimp = imp.groupby(imp.index.map(grp)).sum().sort_values(ascending=False)
        result["shap_features"] = imp.head(20).rename("mean|SHAP| (log miqyas)").reset_index(names="xüsusiyyət")
        result["shap_groups"] = (100 * gimp / gimp.sum()).round(1).rename("pay %").reset_index(names="qrup")
        fig, ax = plt.subplots(figsize=(7, 6))
        imp.head(20)[::-1].plot.barh(ax=ax, color="#2a6f97")
        ax.set_title("Ən təsirli 20 xüsusiyyət (orta |SHAP|, median model)")
        fig.tight_layout()
        fig.savefig(out_dir / "shap_top20.png", dpi=110)
        plt.close(fig)
    except Exception as e:  # SHAP olmadan da hesabat yaradılır
        log.warning("SHAP alınmadı: %s", e)

    # Nümunə qrafik
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, (s, g) in zip(axes, bt[bt.fold == len(folds)].groupby("station")):
        g = g[g.h == 24].set_index("ts").sort_index().loc["2024-01-15":"2024-02-05"]
        ax.fill_between(g.index, g.q10, g.q90, alpha=.25, color="#2a6f97", label="10–90% interval")
        ax.plot(g.index, g.q50, color="#2a6f97", lw=1.2, label="proqnoz (24 saat əvvəl)")
        ax.plot(g.index, g.q_actual, color="#222", lw=.9, label="faktiki")
        ax.set_title(s)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_dir / "backtest_example.png", dpi=110)
    plt.close(fig)

    write_report(result, out_dir, ctx)
    return result


def write_report(r: dict, out_dir, ctx):
    def md(df):
        return df.round(3).to_markdown(index=False)
    lines = ["# Model qiymətləndirməsi (walk-forward)", "",
             f"Hesablandı: {pd.Timestamp.now():%Y-%m-%d %H:%M}. HDD bazası: {ctx.hdd_base}", "",
             "## Ümumi (bütün fold-lar, test havası = 1 gün əvvəlki proqnoz)", md(r["overall"]), "",
             "## Məntəqələr üzrə", md(r["by_station"]), "",
             "## Üfüq üzrə", md(r["by_horizon"]), "",
             "## Fold-lar üzrə", md(r["by_fold"]), "",
             "## Qeyri-müəyyənlik intervalı (10–90%, hədəf örtük 80%)", md(r["intervals"]), "",
             "## Növbəti 24 saatın cəmi", md(r["next24_total"]), "",
             "## Sızma testi (2024 fold-u: bütün hava dəyişənləri real proqnozdur)", md(r["leakage_2024"]), ""]
    if "ablation" in r:
        lines += ["## Ablation (xüsusiyyət qrupları ardıcıl əlavə edilir; median model)", md(r["ablation"]), ""]
    if "shap_groups" in r:
        lines += ["## SHAP: qruplar üzrə təsir payı", md(r["shap_groups"]), "",
                  "## SHAP: ən təsirli xüsusiyyətlər", md(r["shap_features"]), ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    js = {k: v.to_dict(orient="records") for k, v in r.items()}
    (out_dir / "metrics.json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    kv_set("model_metrics", json.loads(json.dumps(js, default=str)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(quick=a.quick)
    print((path("reports") / "model" / "report.md").read_text(encoding="utf-8"))
