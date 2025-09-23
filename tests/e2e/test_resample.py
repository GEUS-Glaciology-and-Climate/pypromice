import numpy as np
import pandas as pd
import unittest

from pypromice.pipeline.resample import apply_completeness_filters


def _resample_and_filter(df_h, t, time_thresh=0.8, value_thresh=0.8):
    df_resampled = df_h.resample(t).mean()
    filtered = apply_completeness_filters(
        df_resampled,
        df_h,
        t,
        time_thresh=time_thresh,
        value_thresh=value_thresh,
    )
    return df_resampled, filtered


class TestCompletenessFilters(unittest.TestCase):

    def test_10min_to_hourly_with_nans(self):
        # 2 hours @10min; first hour complete, second hour missing 2 samples
        idx = pd.date_range("2025-01-01 00:00", periods=24, freq="10min")
        x = np.arange(24).astype(float)
        x[8:10] = np.nan  # two NaNs in second hour (4/6 present -> 0.666 < 0.8)
        df_h = pd.DataFrame({"x": x}, index=idx)

        df_res, filtered = _resample_and_filter(df_h, t="60min")

        self.assertIn(pd.infer_freq(df_res.index), {"h", "H"})  # resample result index
        # hour bins: 00:00 and 01:00
        self.assertFalse(pd.isna(filtered.loc["2025-01-01 00:00", "x"]))  # complete -> kept
        self.assertTrue(pd.isna(filtered.loc["2025-01-01 01:00", "x"]))   # incomplete -> masked

    def test_hourly_to_daily_with_nans(self):
        # 2 days @hourly; day1 has 20/24 good (pass), day2 has 15/24 good (fail)
        idx = pd.date_range("2025-02-01", periods=96, freq="60min")
        x = np.ones(96)
        day1_bad = [1, 5, 9, 13]             # 4 NaNs -> 20/24 present
        day2_bad = [24 + i for i in range(9)]  # 9 NaNs -> 15/24 present
        x[day1_bad + day2_bad] = np.nan
        df_h = pd.DataFrame({"x": x}, index=idx)

        df_res, filtered = _resample_and_filter(df_h, t="1D")

        self.assertFalse(pd.isna(filtered.loc["2025-02-01", "x"]))  # pass
        self.assertTrue(pd.isna(filtered.loc["2025-02-02", "x"]))   # fail

    def test_daily_to_monthly_ms_with_nans(self):
        # Two months daily; Jun has 28/30 present (pass), Jul has 20/31 present (fail)
        idx = pd.date_range("2025-06-01", periods=120, freq="1D")  # Jun(30) + Jul(31)
        x = np.arange(120, dtype=float)
        # Make 2 NaNs in June (28/30)
        x[[5, 17]] = np.nan
        # Make 11 NaNs in July (20/31 present -> below 0.8)
        july_start = 30
        x[[july_start + i for i in [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17]]] = np.nan
        df_h = pd.DataFrame({"x": x}, index=idx)

        df_res, filtered = _resample_and_filter(df_h, t="MS")

        # MS bins at month starts
        self.assertFalse(pd.isna(filtered.loc["2025-06-01", "x"]))  # June kept
        self.assertTrue(pd.isna(filtered.loc["2025-07-01", "x"]))   # July masked

    def test_mixed_10min_then_hourly_to_hourly(self):
        # First hour: 5 samples @10min (5/6=0.833 pass), second hour: 1 hourly sample (pass)
        idx_10 = pd.date_range("2025-03-01 00:00", periods=5, freq="10min")
        idx_60 = pd.date_range("2025-03-01 02:00", periods=3, freq="60min")
        idx = idx_10.union(idx_60)
        x = np.arange(len(idx), dtype=float)
        df_h = pd.DataFrame({"x": x}, index=idx)

        df_res, filtered = _resample_and_filter(df_h, t="60min")

        self.assertFalse(pd.isna(filtered.loc["2025-03-01 00:00", "x"]))  # 5/6 -> pass
        self.assertTrue(pd.isna(filtered.loc["2025-03-01 01:00", "x"]))   # 0 hourly -> failed
        self.assertFalse(pd.isna(filtered.loc["2025-03-01 02:00", "x"]))  # 1 hourly -> pass

    def test_mixed_hourly_then_daily_to_daily(self):
        # Day1: 20 hourly samples (20/24=0.833 pass)
        # Day2: a single daily sample (pass, counts as 1)
        idx_hourly = pd.date_range("2025-04-01 00:00", periods=30, freq="60min")
        idx_daily = pd.date_range("2025-04-03 00:00", periods=3, freq="D")
        idx = idx_hourly.union(idx_daily)
        x = np.arange(len(idx), dtype=float)
        df_h = pd.DataFrame({"x": x}, index=idx)

        df_res, filtered = _resample_and_filter(df_h, t="1D")

        self.assertFalse(pd.isna(filtered.loc["2025-04-01", "x"]))  # 20/24 -> pass
        self.assertTrue(pd.isna(filtered.loc["2025-04-02", "x"]))   # 6/24  -> failed
        self.assertFalse(pd.isna(filtered.loc["2025-04-03", "x"]))  # daily sample -> pass


if __name__ == "__main__":
    unittest.main()
