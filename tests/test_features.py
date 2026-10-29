import numpy as np
import pandas as pd

from features.build import FeatureContext, make_rows

ORIGIN_COLS = ["lag_last", "mean_24", "mean_168", "trend_24", "min_24"]
TARGET_LAGS = ["q_lag24", "q_lag48", "q_lag168", "same_hour_mean7"]


def _ctx(h, w):
    return FeatureContext(hourly=h, weather={"turkan": w}, hdd_base={"turkan": 19.5})


def test_no_leakage_from_future_consumption(synthetic_hourly, synthetic_weather):
    origin = pd.Timestamp("2023-01-15 07:00")
    a = make_rows(_ctx(synthetic_hourly, synthetic_weather), "turkan", 24, origins=pd.DatetimeIndex([origin]))
    # Başlanğıcdan SONRAKI sərfi kəskin dəyişdir → xüsusiyyətlər dəyişməməlidir
    h2 = synthetic_hourly.copy()
    fut = h2.ts > origin
    h2.loc[fut, ["q_filled", "q_clean"]] *= 100
    b = make_rows(_ctx(h2, synthetic_weather), "turkan", 24, origins=pd.DatetimeIndex([origin]))
    cols = ORIGIN_COLS + TARGET_LAGS
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))
    assert len(a) == 24 and a.h.tolist() == list(range(1, 25))


def test_batch_rows_match_single_origin(synthetic_hourly, synthetic_weather):
    """Təlim (toplu) rejimində sətirlər tək-başlanğıclı canlı rejimlə eyni xüsusiyyətləri verməlidir."""
    ctx = _ctx(synthetic_hourly, synthetic_weather)
    origin = pd.Timestamp("2023-01-15 06:00")
    batch = make_rows(ctx, "turkan", 24, origin_step=3)
    batch = batch[batch.origin == origin].sort_values("h").reset_index(drop=True)
    single = make_rows(ctx, "turkan", 24, origins=pd.DatetimeIndex([origin])).sort_values("h").reset_index(drop=True)
    cols = ORIGIN_COLS + TARGET_LAGS + ["temp", "hdd", "is_nonworking"]
    np.testing.assert_allclose(batch[cols].to_numpy(float), single[cols].to_numpy(float), rtol=1e-6)


def test_lag_last_equals_origin_value(synthetic_hourly, synthetic_weather):
    origin = pd.Timestamp("2023-01-16 12:00")
    r = make_rows(_ctx(synthetic_hourly, synthetic_weather), "turkan", 24, origins=pd.DatetimeIndex([origin]))
    v = synthetic_hourly.set_index("ts").loc[origin, "q_filled"]
    assert np.allclose(r.lag_last, v)
    # t − 24 lag həmişə ≤ başlanğıc
    assert (r.ts - pd.Timedelta("24h") <= r.origin).all()
