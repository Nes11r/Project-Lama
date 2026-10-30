"""Operator paneli (Streamlit prototipi). Bütün data FastAPI-dən alınır.

İşə salma:  streamlit run frontend/app.py
Mühit:      API_URL (default http://localhost:8000)
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

_url = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
API = (_url if _url.startswith("http") else f"http://{_url}") + "/api/v1"
HEADERS = {"X-API-Key": os.environ["API_TOKEN"]} if os.environ.get("API_TOKEN") else {}
SEV = {"critical": ("🔴", "Kritik"), "warning": ("🟠", "Xəbərdarlıq"), "info": ("🔵", "Məlumat")}
PRIO = {"high": "🔴", "medium": "🟠", "info": "🔵"}
C_FC, C_ACT, C_BAND = "#1f6feb", "#24292f", "rgba(31,111,235,0.18)"

st.set_page_config(page_title="Qaz sərfiyyatı — operator paneli", page_icon="🔥", layout="wide")

# Sadə giriş qoruması (PANEL_PASSWORD təyin olunubsa)
if os.environ.get("PANEL_PASSWORD") and not st.session_state.get("auth"):
    st.title("🔥 Qaz sərfiyyatı — operator paneli")
    pw = st.text_input("Parol", type="password")
    if st.button("Daxil ol"):
        if pw == os.environ["PANEL_PASSWORD"]:
            st.session_state["auth"] = True
            st.rerun()
        st.error("Parol yanlışdır")
    st.stop()


WAKE_STATUS = {502, 503, 504}  # Render free: API yuxudan oyananda proxy bu kodları qaytarır
WAKE_TIMEOUT_S = 120


def _request(method: str, url: str, **kw) -> requests.Response:
    """GET sorğusu API oyanana qədər təkrarlanır; yazma sorğuları təkrarlanmır."""
    deadline = time.monotonic() + WAKE_TIMEOUT_S
    note = None
    try:
        while True:
            try:
                r = requests.request(method, API + url, timeout=120, headers=HEADERS, **kw)
                if r.status_code not in WAKE_STATUS or method != "GET":
                    return r
            except requests.ConnectionError:
                if method != "GET":
                    raise
                r = None
            if time.monotonic() > deadline:
                if r is None:
                    raise requests.ConnectionError("API cavab vermir")
                return r
            if note is None:
                note = st.info("API oyanır (pulsuz serverdə ilk açılış 1 dəqiqəyə qədər çəkir)...")
            time.sleep(5)
    finally:
        if note is not None:
            note.empty()


def api(method: str, url: str, **kw):
    try:
        r = _request(method, url, **kw)
        try:
            body = r.json()
        except ValueError:
            st.error(f"API JSON qaytarmadı ({API + url}, HTTP {r.status_code}). "
                     "Səhifəni bir neçə saniyədən sonra yeniləyin.")
            return None
        if r.status_code >= 400:
            st.error(f"API xətası {r.status_code}: {body.get('detail', r.text) if isinstance(body, dict) else r.text}")
            return None
        return body
    except requests.RequestException as e:
        st.error(f"API ilə əlaqə yoxdur ({API}): {e}")
        return None


@st.cache_data(ttl=30)
def _stations_cached():
    res = api("GET", "/stations")
    if not res:
        raise RuntimeError("stations")  # uğursuz cavab keşə yazılmır
    return res


def get_stations():
    try:
        return _stations_cached()
    except RuntimeError:
        return []


def fmt(v, nd=2, unit=""):
    return "—" if v is None else f"{v:,.{nd}f}{unit}".replace(",", " ")


# ------------------------------------------------------------------ yan panel
stations = get_stations()
if not stations:
    st.stop()
names = {s["id"]: s["name"] for s in stations}
with st.sidebar:
    st.title("🔥 Qaz sərfiyyatı")
    page = st.radio("Bölmə", ["Əsas panel", "Tarixçə və dəqiqlik", "Xəbərdarlıqlar", "Tənzimləmələr"])
    sid = st.selectbox("Məntəqə", list(names), format_func=names.get)
    status = api("GET", "/status") or {}
    st.divider()
    if status.get("mode") == "replay":
        st.caption("DEMO rejimi: tarixi data real vaxt kimi oynadılır")
        now = pd.Timestamp(status["now"])
        st.markdown(f"**İndiki an:** {now:%d.%m.%Y %H:00}")
        c1, c2, c3 = st.columns(3)
        if c1.button("+1 s"):
            api("PUT", "/clock", json={"advance_hours": 1}); st.rerun()
        if c2.button("+6 s"):
            api("PUT", "/clock", json={"advance_hours": 6}); st.rerun()
        if c3.button("+24 s"):
            api("PUT", "/clock", json={"advance_hours": 24}); st.rerun()
        d = st.date_input("Tarixə keç", now.date(), min_value=datetime(2018, 2, 1), max_value=datetime(2024, 8, 20))
        hh = st.number_input("Saat", 0, 23, int(now.hour))
        if st.button("Təyin et"):
            api("PUT", "/clock", json={"now": f"{d} {hh:02d}:00"}); st.rerun()
    else:
        st.caption("CANLI rejim")
    st.divider()
    st.caption(f"Model: {status.get('model_version', '—')}")
    st.caption("🛡️ Avtomatik idarəetmə: **deaktiv** (yalnız tövsiyə)")
    bad = [k for k, v in (status.get("external_sources") or {}).items() if not v.get("ok", True)]
    if bad:
        st.warning("Əlçatmaz mənbə(lər): " + ", ".join(bad) + " — keş istifadə olunur")


# ------------------------------------------------------------------ əsas panel
def page_dashboard():
    d = api("GET", f"/dashboard/{sid}")
    if not d:
        return
    st.header(f"{d['station_name']} — cari vəziyyət və növbəti 24 saat")
    cur, fc, lim = d["current"], d["forecast"], d["limits"]
    f = pd.DataFrame(fc["forecast"])
    f["ts"] = pd.to_datetime(f["ts"])
    ex = fc.get("explanation") or {}

    k = st.columns(5)
    k[0].metric("Cari sərf", fmt(cur.get("flow"), 3, " min m³/s"))
    k[1].metric("Təzyiq", fmt(cur.get("pressure"), 0, " kPa"),
                help=f"Minimum hədd: {lim['min_pressure_kpa']} kPa (mütləq)")
    k[2].metric("Son 24 saat (faktiki)", fmt(cur.get("flow_24h_total"), 1, " min m³"))
    k[3].metric("Növbəti 24 saat (proqnoz)", fmt(ex.get("next24_total"), 1, " min m³"),
                None if ex.get("change_pct") is None else f"{ex['change_pct']:+.1f}%")
    peak = f.loc[f.q50.idxmax()]
    k[4].metric("Gözlənilən pik", f"{peak.q50:.2f}", f"{peak.ts:%H}:00", delta_color="off")

    # Qrafik: son 48 saat (faktiki + 1 gün əvvəlki proqnoz) + növbəti 24 saat
    r = pd.DataFrame(d.get("recent") or [])
    fig = go.Figure()
    if not r.empty:
        r["ts"] = pd.to_datetime(r["ts"])
        fig.add_trace(go.Scatter(x=r.ts, y=r.q90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=r.ts, y=r.q10, fill="tonexty", fillcolor="rgba(120,120,120,0.15)",
                                 line=dict(width=0), name="keçmiş proqnoz intervalı"))
        fig.add_trace(go.Scatter(x=r.ts, y=r.q50, line=dict(color="#8c959f", dash="dot"), name="keçmiş proqnoz"))
        fig.add_trace(go.Scatter(x=r.ts, y=r.actual, line=dict(color=C_ACT, width=2), name="faktiki"))
    fig.add_trace(go.Scatter(x=f.ts, y=f.q90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f.ts, y=f.q10, fill="tonexty", fillcolor=C_BAND, line=dict(width=0),
                             name="10–90% interval"))
    fig.add_trace(go.Scatter(x=f.ts, y=f.q50, line=dict(color=C_FC, width=2.5), name="proqnoz (median)"))
    fig.add_hline(y=lim["peak_flow_warning"], line=dict(color="#d4a72c", dash="dash"),
                  annotation_text="xəbərdarlıq həddi")
    fig.add_hline(y=lim["peak_flow_critical"], line=dict(color="#cf222e", dash="dash"),
                  annotation_text="kritik hədd")
    fig.add_vline(x=pd.Timestamp(d["now"]), line=dict(color="#57606a"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="min m³/saat",
                      legend=dict(orientation="h", y=-0.15), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Hava mənbəyi: {fc['weather_source']} · Model {fc['model_version']} · "
               f"Proqnoz başlanğıcı {pd.Timestamp(fc['origin']) + timedelta(hours=1):%d.%m %H:00}")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Xəbərdarlıqlar")
        al = [a for a in d["alerts"] if not a["suppressed"]]
        sup = [a for a in d["alerts"] if a["suppressed"]]
        if not al:
            st.success("Aktiv xəbərdarlıq yoxdur")
        for a in al:
            icon, lbl = SEV[a["severity"]]
            with st.expander(f"{icon} {a['title']}", expanded=a["severity"] != "info"):
                st.write(a["message"])
                st.caption(f"Əsas: {a['reason']}")
        if sup:
            st.caption(f"Qeydiyyatlı hadisəyə görə susdurulmuş: {len(sup)}")
    with c2:
        st.subheader("Proqnozun səbəbləri")
        if ex.get("factors"):
            st.caption(f"Son 24 saatla müqayisədə ({fmt(ex.get('prev24_actual_total'), 1)} → "
                       f"{fmt(ex.get('next24_total'), 1)} min m³)")
            for it in ex["factors"]:
                arrow = "🔺" if it["pct"] > 0 else "🔻"
                st.markdown(f"{arrow} {it['text']}")
        st.subheader("Tövsiyələr")
        for rec in d["recommendations"]["items"]:
            st.markdown(f"{PRIO.get(rec['priority'], '•')} **{rec['title']}** — {rec['text']}  \n"
                        f"<span style='color:#57606a;font-size:0.85em'>Əsaslandırma: {rec['basis']}</span>",
                        unsafe_allow_html=True)
        st.info(d["recommendations"]["disclaimer"])

    with st.expander("Saatlıq proqnoz cədvəli"):
        t = f[["ts", "q10", "q50", "q90", "temp"]].rename(columns={
            "ts": "Saat", "q10": "10%", "q50": "Median", "q90": "90%", "temp": "Hava °C"})
        st.dataframe(t.style.format({"10%": "{:.3f}", "Median": "{:.3f}", "90%": "{:.3f}", "Hava °C": "{:.1f}"}),
                     use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ tarixçə
def page_history():
    st.header(f"{names[sid]} — tarixçə və model dəqiqliyi")
    now = pd.Timestamp(status.get("now"))
    c1, c2 = st.columns(2)
    s = c1.date_input("Başlanğıc", (now - pd.Timedelta("7D")).date())
    e = c2.date_input("Son", now.date())
    h = api("GET", f"/history/{sid}", params={"start": str(s), "end": f"{e} 23:00"})
    if h:
        a = pd.DataFrame(h["actual"])
        fig = go.Figure()
        if not a.empty:
            a["ts"] = pd.to_datetime(a["ts"])
            fig.add_trace(go.Scatter(x=a.ts, y=a.q_raw, name="faktiki (xam)", line=dict(color=C_ACT)))
            rec = a[a.f_reconstructed == True]  # noqa: E712
            fig.add_trace(go.Scatter(x=rec.ts, y=rec.q_clean, mode="markers", name="'Cəmi' ilə bərpa edilmiş",
                                     marker=dict(color="#8250df", size=6)))
        da = pd.DataFrame(h.get("day_ahead") or [])
        if not da.empty:
            da["ts"] = pd.to_datetime(da["ts"])
            fig.add_trace(go.Scatter(x=da.ts, y=da.q50, name="1 gün əvvəlki proqnoz", line=dict(color=C_FC)))
            m = da.dropna(subset=["actual"])
            if len(m):
                mae = (m.q50 - m.actual).abs().mean()
                wape = 100 * (m.q50 - m.actual).abs().sum() / m.actual.abs().sum()
                st.caption(f"Seçilmiş dövr: MAE {mae:.3f} min m³/saat · WAPE {wape:.1f}%")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    acc = api("GET", f"/accuracy/{sid}")
    if acc:
        st.subheader("Walk-forward backtest (test havası = 1 gün əvvəlki proqnoz)")
        df = pd.DataFrame(acc["folds"])
        show = df[["period", "lgbm_MAE", "lgbm_WAPE%", "snaive_MAE", "snaive_WAPE%", "coverage%"]].rename(columns={
            "period": "Dövr", "lgbm_MAE": "Model MAE", "lgbm_WAPE%": "Model WAPE %",
            "snaive_MAE": "Seasonal naive MAE", "snaive_WAPE%": "Seasonal naive WAPE %",
            "coverage%": "10–90% örtük %"})
        st.dataframe(show.round(3), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        bh = pd.DataFrame(acc["by_horizon"])
        c1.plotly_chart(go.Figure(go.Scatter(x=bh.h, y=bh["WAPE%"], mode="lines+markers"))
                        .update_layout(title="Xəta üfüq üzrə (WAPE %)", xaxis_title="saat irəli", height=300),
                        use_container_width=True)
        bm = pd.DataFrame(acc["by_month"])
        c2.plotly_chart(go.Figure(go.Bar(x=bm.m, y=bm["WAPE%"]))
                        .update_layout(title="Xəta ay üzrə (WAPE %)", xaxis_title="ay", height=300),
                        use_container_width=True)


# ------------------------------------------------------------------ xəbərdarlıqlar
def page_alerts():
    st.header("Xəbərdarlıqlar")
    c1, c2 = st.columns(2)
    only_this = c1.checkbox("Yalnız seçilmiş məntəqə", True)
    hide_ack = c2.checkbox("Təsdiqlənmişləri gizlət", False)
    items = api("GET", "/alerts", params={"station": sid if only_this else None,
                                          "include_ack": not hide_ack}) or []
    if not items:
        st.success("Xəbərdarlıq yoxdur")
    for a in items:
        icon, lbl = SEV[a["severity"]]
        state = "✅ təsdiqlənib" if a.get("acknowledged") else ("🔕 susdurulub" if a["suppressed"] else "")
        with st.container(border=True):
            cc = st.columns([6, 1])
            cc[0].markdown(f"{icon} **{a['title']}** · {a['station_name']} · {a['ts'][:16].replace('T', ' ')} {state}")
            cc[0].write(a["message"])
            cc[0].caption(f"{lbl} · Əsas: {a['reason']}")
            if not a.get("acknowledged") and cc[1].button("Təsdiqlə", key=a["id"]):
                api("POST", f"/alerts/{a['id']}/ack", data={"operator": "operator"})
                st.rerun()


# ------------------------------------------------------------------ tənzimləmələr
def page_settings():
    st.header("Tənzimləmələr")
    lim = (api("GET", "/settings/limits") or {}).get(sid, {})
    st.subheader(f"Limitlər — {names[sid]}")
    with st.form("limits"):
        c = st.columns(3)
        w = c[0].number_input("Pik xəbərdarlıq (min m³/saat)", value=float(lim.get("peak_flow_warning", 0)), step=0.1)
        cr = c[1].number_input("Pik kritik (min m³/saat)", value=float(lim.get("peak_flow_critical", 0)), step=0.1)
        p = c[2].number_input("Minimum təzyiq (kPa, mütləq)", value=float(lim.get("min_pressure_kpa", 0)), step=5.0)
        if st.form_submit_button("Yadda saxla"):
            if api("PUT", f"/settings/limits/{sid}",
                   json={"peak_flow_warning": w, "peak_flow_critical": cr, "min_pressure_kpa": p}):
                st.success("Saxlanıldı")
                _stations_cached.clear()

    st.subheader("Yeni data yüklə (ROC CSV ixracı)")
    st.caption("Fayl `dataset/incoming/` qovluğuna yazılır, təmizləmə pipeline-ı avtomatik işə düşür. "
               "Qovluğa birbaşa atılan fayllar da hər dəqiqə yoxlanılır.")
    up = st.file_uploader("CSV", type=["csv"])
    if up and st.button("Yüklə və emal et"):
        with st.spinner("Emal olunur..."):
            res = api("POST", "/upload", data={"station": sid}, files={"file": (up.name, up.getvalue(), "text/csv")})
        if res:
            st.success(f"{res['rows_in_file']} sətir qəbul edildi → {res['saved_as']}")

    st.subheader("Model versiyaları")
    st.dataframe(pd.DataFrame(api("GET", "/models") or []), use_container_width=True, hide_index=True)
    st.caption("Yenidən öyrətmə: `python -m models.evaluate && python -m models.train` (README-yə bax)")

    st.subheader("Avtomatika interfeysi")
    spec = api("GET", "/control/spec") or {}
    st.error(f"Deaktiv. {spec.get('policy', '')}")
    with st.expander("Gələcək üçün nəzərdə tutulan interfeys"):
        st.json(spec)


{"Əsas panel": page_dashboard, "Tarixçə və dəqiqlik": page_history,
 "Xəbərdarlıqlar": page_alerts, "Tənzimləmələr": page_settings}[page]()
