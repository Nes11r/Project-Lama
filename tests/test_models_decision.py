import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from models.metrics import interval_metrics, metrics


def test_metrics_basic():
    m = metrics([1, 2, 3, 0], [1, 2, 4, 0])
    assert m["MAE"] == 0.25 and abs(m["WAPE%"] - 100 / 6) < 1e-9
    assert m["MAPE%"] > 0  # sıfır dəyər MAPE-dən çıxarılır
    im = interval_metrics([1, 2, 3], [0, 0, 0], [2, 2, 2])
    assert abs(im["coverage%"] - 200 / 3) < 1e-9


def test_quantile_forecaster_monotone():
    from models.lgbm import QuantileForecaster
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500), "station_code": rng.integers(0, 3, 500)})
    y = X.a * 2 + rng.normal(size=500)
    f = QuantileForecaster(["a", "station_code"], [0.1, 0.5, 0.9],
                           {"n_estimators": 30, "verbose": -1, "min_child_samples": 5}).fit(X, y)
    p = f.predict_log(X)
    assert (p.q10 <= p.q50).all() and (p.q50 <= p.q90).all()


def test_peak_alert(monkeypatch):
    from decision import alerts as al
    monkeypatch.setattr(al, "limits", lambda: {"turkan": {"peak_flow_warning": 5, "peak_flow_critical": 8,
                                                           "min_pressure_kpa": 300}})
    monkeypatch.setattr(al, "read_status", lambda: {})
    ts = pd.date_range("2024-01-10 01:00", periods=24, freq="h")
    fc = {"forecast": pd.DataFrame({"ts": ts, "q10": 4.0, "q50": np.r_[np.full(23, 4.0), 6.0], "q90": 7.0}),
          "events": []}
    hist = pd.DataFrame({"ts": pd.date_range("2024-01-09 01:00", periods=24, freq="h"), "q_raw": np.arange(24.0),
                         "p_kpa": 500.0, "f_dp_saturated": False})
    out = al.evaluate_alerts("turkan", ts[0], fc, pd.DataFrame(), hist)
    peak = [a for a in out if a["type"] == "peak_forecast"]
    assert peak and peak[0]["severity"] == "warning"


def test_control_endpoints_disabled():
    from api.main import app
    c = TestClient(app)  # lifespan işə düşmür → model yüklənmir
    assert c.post("/api/v1/control/setpoint").status_code == 403
    assert c.get("/api/v1/control/spec").json()["enabled"] is False
