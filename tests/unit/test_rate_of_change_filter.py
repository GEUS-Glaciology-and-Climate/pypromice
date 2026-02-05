# test_rate_of_change_filter.py
import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.qc.rate_of_change_filter import flag_high_rate_of_change

TOL = 3.0
FACTOR = 2.2
CASE_A_FINAL_TIMES = ['2024-01-02T20:10:00']
CASE_B_FINAL_TIMES = ['2024-01-01T16:50:00']
CASE_C_FINAL_TIMES = ['2024-01-01T16:10:00', '2024-01-01T17:00:00']
CASE_D_FINAL_TIMES = []

def make_irregular_time(start="2024-01-01", n=200, base="10min", drop_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq=base)
    keep = rng.random(n) > drop_frac
    return t[keep].to_numpy()


def build_case_A():
    start, n, base = "2024-01-01", 400, "10min"
    t = pd.date_range(start, periods=n, freq=base)
    x = np.arange(len(t))

    slope = 1.0
    plateau_low, plateau_high = -30.0, -10.0
    x0 = 280
    plateau = np.clip(slope * (x - x0), plateau_low, plateau_high)
    base_sig = plateau + 2 * np.sin(2 * np.pi * x / 60)

    v = base_sig.copy()
    k = x0 - 15
    v[k] = v[k] + 10

    mask = np.ones_like(v, dtype=bool)
    mask[k-5:k-4] = False
    mask[k+1:k+5] = False

    return xr.Dataset({"t_u": (("time",), v[mask])}, coords={"time": t[mask]})


def build_case_B():
    t = pd.to_datetime(make_irregular_time(seed=2))
    x = np.arange(len(t))
    v = -25 + 0.015 * x + 0.15 * np.sin(2 * np.pi * x / 80)
    k = len(v) // 2
    v[k] = v[k - 1] + 8.0
    return xr.Dataset({"t_u": (("time",), v)}, coords={"time": t})


def build_case_C():
    t = pd.to_datetime(make_irregular_time(seed=3, drop_frac=0.05))
    x = np.arange(len(t))
    v = -15 + 0.25 * np.sin(2 * np.pi * x / 50)
    m0, m1 = 85, 130
    v[m0:m1] = np.nan
    v[m0+10] = -5.0
    v[m0+15] = -10.0
    return xr.Dataset({"t_u": (("time",), v)}, coords={"time": t})


def build_case_D():
    t = pd.to_datetime(make_irregular_time(seed=5))
    x = np.arange(len(t))
    v = -10 + 0.02 * x + 0.1 * np.sin(2 * np.pi * x / 70)
    k = len(v) // 2
    v[k - 1] = v[k - 1] + 6.0
    v[k] = v[k] + 6.0
    return xr.Dataset({"t_u": (("time",), v)}, coords={"time": t})

def _flag_times(ds):
    _, _, _, final = flag_high_rate_of_change(ds, "t_u", tol=TOL, factor=FACTOR)
    return pd.to_datetime(final.time.values[final.values]).strftime("%Y-%m-%dT%H:%M:%S").tolist()

def test_case_A_expected():
    assert _flag_times(build_case_A()) == CASE_A_FINAL_TIMES

def test_case_B_expected():
    assert _flag_times(build_case_B()) == CASE_B_FINAL_TIMES

def test_case_C_expected():
    assert _flag_times(build_case_C()) == CASE_C_FINAL_TIMES

def test_case_D_expected():
    assert _flag_times(build_case_D()) == CASE_D_FINAL_TIMES
