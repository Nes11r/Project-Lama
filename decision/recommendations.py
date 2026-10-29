"""İzahlı tövsiyələr. Sistem YALNIZ TÖVSİYƏ rejimindədir — son qərar operatordadır."""
from __future__ import annotations

import pandas as pd

from decision.alerts import limits

DISCLAIMER = "Tövsiyələr məlumat xarakterlidir; heç bir əmr avtomatik icra edilmir. Son qərar operatordadır."


def recommendations(sid: str, fc: dict, alerts: list[dict]) -> dict:
    f = fc["forecast"].copy()
    lim = limits()[sid]
    f["hour"] = pd.to_datetime(f.ts).dt.hour
    top = f.nlargest(3, "q50").sort_values("ts")
    peak = f.loc[f.q50.idxmax()]
    util = 100 * peak.q50 / lim["peak_flow_warning"]
    recs = []

    peaks_txt = ", ".join(f"{pd.Timestamp(r.ts):%H}:00–{(pd.Timestamp(r.ts) + pd.Timedelta('1h')):%H}:00 "
                          f"({r.q50:.2f})" for r in top.itertuples())
    recs.append({"priority": "info", "title": "Gözlənilən pik saatlar",
                 "text": f"Ən yüksək sərf: {peaks_txt} min m³/saat.",
                 "basis": f"Pik xəbərdarlıq həddinin {util:.0f}%-i səviyyəsindədir."})

    if util >= 100:
        recs.append({"priority": "high", "title": "Pik tələbata hazırlıq",
                     "text": f"{pd.Timestamp(peak.ts):%H}:00-dan əvvəl giriş təzyiqi və tənzimləyici qurğuların iş "
                             "rejiminin yoxlanması, növbə heyətinin xəbərdar edilməsi tövsiyə olunur.",
                     "basis": f"Median proqnoz {peak.q50:.2f} ≥ xəbərdarlıq həddi {lim['peak_flow_warning']}."})
    elif util >= 85:
        recs.append({"priority": "medium", "title": "Pik saatlarda müşahidəni gücləndirin",
                     "text": "Pik saatlarda təzyiqin daha tez-tez izlənməsi tövsiyə olunur.",
                     "basis": f"Pik limitin {util:.0f}%-nə yaxınlaşır."})

    ex = fc.get("explanation") or {}
    if ex.get("temp_next24") is not None and ex["temp_next24"] < ex["temp_prev24"] - 3:
        recs.append({"priority": "medium", "title": "Kəskin soyuma",
                     "text": f"Orta temperaturun {ex['temp_prev24']:.1f}°C-dən {ex['temp_next24']:.1f}°C-yə enməsi "
                             "gözlənilir; istilik yükü artacaq.",
                     "basis": "Hava proqnozu (Open-Meteo) və modelin temperatur həssaslığı."})

    width = ((f.q90 - f.q10) / f.q50.clip(lower=1e-6)).median()
    if width > 0.5:
        recs.append({"priority": "info", "title": "Yüksək qeyri-müəyyənlik",
                     "text": "Proqnoz intervalı genişdir; qərarlarda yuxarı hədd (90%) ssenarisini də nəzərə alın.",
                     "basis": f"Orta nisbi interval eni {100 * width:.0f}%."})

    types = {a["type"] for a in alerts if not a["suppressed"]}
    if types & {"deviation_high", "night_excess"}:
        recs.append({"priority": "high", "title": "Sahə yoxlaması",
                     "text": "Faktiki sərf proqnozdan sistemli yüksəkdir: xəttin sızma baxımından yoxlanması və "
                             "sayğac göstəricilərinin müqayisəsi tövsiyə olunur.",
                     "basis": "Faktiki ↔ proqnoz sapması və/və ya gecə minimum sərfi artıb."})
    if types & {"deviation_low"}:
        recs.append({"priority": "medium", "title": "Təchizatın yoxlanması",
                     "text": "Faktiki sərf gözləniləndən aşağıdır: klapan vəziyyəti, təchizat kəsintisi və ya sensor "
                             "nasazlığı yoxlanmalıdır.", "basis": "Sistemli mənfi sapma."})
    if types & {"sensor_flatline", "data_stale"}:
        recs.append({"priority": "medium", "title": "Rabitə/sensor yoxlaması",
                     "text": "ROC rabitəsini və sərf sensorunu yoxlayın.",
                     "basis": "Data gəlmir və ya dəyişmir."})
    return {"items": recs, "disclaimer": DISCLAIMER}
