import numpy as np
import pandas as pd

from core.config import load_config
from data.cleaning import clean_station, reconstruct_from_totals
from data.loader import parse_roc_file

HEADER = 'TARİX,"XÜSUSİ ÇƏKİ\n(kq/m3)","TƏZYİQLƏR\nFƏRQİ (kPa)",TƏZYİQ (kPa),"TEMPERATUR\n(C)","SAATLIQ\nSƏRF(min m3)",SƏRF (min m3)\n'


def _write(tmp_path, lines):
    f = tmp_path / "Türkan test.csv"
    f.write_text(HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    return f


def test_parse_hour_ending_and_totals(tmp_path):
    lines = [f"01-01-2023 {h:02d}:00,0.74,5.0,500.0,10.0,1.000,24.000" for h in range(1, 24)]
    lines += ["02-01-2023 00:00,0.74,5.0,500.0,10.0,1.000,24.000", "Cəmi: 24.000,,,,,,",
              "02-01-2023 01:00,0.74,5.0,500.0,10.0,2.000,48.000", "02-01-2023 01:37,0.74,5.0,500.0,10.0,2.000,48.000"]
    rows, totals = parse_roc_file(_write(tmp_path, lines))
    # "01:00" damğası 00:00–01:00 intervalıdır
    assert rows.ts.min() == pd.Timestamp("2023-01-01 00:00")
    assert len(totals) == 1 and totals.gas_day.iloc[0] == pd.Timestamp("2023-01-01")
    assert totals.total.iloc[0] == 24.0
    assert (~rows.on_hour).sum() == 1  # 01:37 natamam saat


def test_reconstruct_single_missing_hour_exact():
    idx = pd.date_range("2023-01-01", periods=24, freq="h")
    true = pd.Series(np.linspace(1, 3, 24), index=idx)
    q = true.copy()
    q.iloc[10] = np.nan
    missing = q.isna()
    totals = pd.DataFrame({"gas_day": [pd.Timestamp("2023-01-01")], "total": [true.sum()]})
    out, flag = reconstruct_from_totals(q, missing, totals, 6, (0.2, 3.0))
    assert flag.sum() == 1
    assert abs(out.iloc[10] - true.iloc[10]) < 1e-9


def test_reconstruct_rejects_implausible_total():
    idx = pd.date_range("2023-01-01", periods=24, freq="h")
    q = pd.Series(2.0, index=idx)
    q.iloc[5] = np.nan
    totals = pd.DataFrame({"gas_day": [pd.Timestamp("2023-01-01")], "total": [46.0 + 50]})  # 50 ≫ 2
    out, flag = reconstruct_from_totals(q, q.isna(), totals, 6, (0.2, 3.0))
    assert flag.sum() == 0 and np.isnan(out.iloc[5])


def test_clean_flags_sensor_errors_and_events():
    ts = pd.date_range("2023-01-01", periods=24 * 10, freq="h")
    q = 2 + np.sin(2 * np.pi * ts.hour / 24)
    rows = pd.DataFrame({"ts": ts, "sg": 0.74, "dp_kpa": 5.0, "p_kpa": 500.0, "t_c": 10.0, "q_hour": q,
                         "on_hour": True, "file_order": 0})
    rows.loc[50, "t_c"] = 746.0           # sensor xətası
    rows.loc[60, "p_kpa"] = 100.0         # xətt boşaldılıb
    events = [{"station": "turkan", "start": "2023-01-08 10:00", "end": None, "exclude_from_training": True}]
    df, rep = clean_station("turkan", rows, pd.DataFrame(columns=["gas_day", "total"]),
                            load_config()["cleaning"], events)
    df = df.set_index("ts")
    assert np.isnan(df.loc[ts[50], "t_c"]) or df.loc[ts[50], "t_c"] < 60  # NaN → interpolasiya
    assert df.loc[ts[60], "f_depressurized"]
    assert not df.loc[ts[60], "train_ok"]
    assert df.loc["2023-01-08 10:00", "f_event"] and not df.loc["2023-01-08 09:00", "f_event"]
    assert not df.loc["2023-01-09 10:00", "train_ok"]
    assert rep["out_of_range"]["t_c"] == 1
