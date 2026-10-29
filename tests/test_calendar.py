import pandas as pd

from external.calendar import build_calendar, hourly_calendar_features


def test_transferred_working_saturday():
    cal = build_calendar("2023-06-20", "2023-07-05")
    # 24.06.2023 (şənbə) iş günüdür, 27.06.2023 (çərşənbə axşamı) istirahət
    assert cal.loc["2023-06-24", "is_working_weekend"]
    assert not cal.loc["2023-06-24", "is_nonworking"]
    assert cal.loc["2023-06-27", "is_nonworking"]


def test_novruz_block_and_pre_holiday():
    cal = build_calendar("2024-03-10", "2024-03-31")
    assert cal.loc["2024-03-20", "holiday_kind"] == "novruz"
    assert cal.loc["2024-03-22", "block_len"] >= 5
    assert cal.loc["2024-03-19", "pre_holiday"]


def test_ramadan_flag():
    cal = build_calendar("2024-03-01", "2024-04-20")
    assert cal.loc["2024-03-25", "ramadan"]
    assert not cal.loc["2024-04-15", "ramadan"]


def test_hourly_sun_features():
    ts = pd.date_range("2024-01-15", periods=24, freq="h")
    f = hourly_calendar_features(ts, "sumqayit4")
    assert f.loc[ts[2], "is_dark"] == 1 and f.loc[ts[12], "is_dark"] == 0
