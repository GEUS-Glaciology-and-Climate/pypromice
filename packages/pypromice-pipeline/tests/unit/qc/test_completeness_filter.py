import numpy as np
import pandas as pd
import unittest

from pypromice.core.resampling import get_completeness_mask, DEFAULT_COMPLETENESS_THRESHOLDS


class TestCompletenessFilters(unittest.TestCase):

    def test_10min_to_hourly_with_nans(self):
        # 2 hours @10min; first hour complete, second hour missing 2 samples
        idx = pd.date_range("2025-01-01 00:00", periods=24, freq="10min")
        x = np.arange(24).astype(float)
        x[8:10] = np.nan  # two NaNs in second hour (4/6 present -> 0.666 < 0.8)
        df_h = pd.DataFrame({"x": x}, index=idx)

        # df_res, filtered = _resample_and_filter(df_h, t="60min")
        completeness_mask = get_completeness_mask(
            data_frame=df_h,
            completeness_thresholds=DEFAULT_COMPLETENESS_THRESHOLDS,
            resample_offset="60min",
        )

        # hour bins: 00:00 and 01:00
        self.assertTrue(
            completeness_mask.loc["2025-01-01 00:00", "x"]
        )  # complete -> kept
        self.assertFalse(
            completeness_mask.loc["2025-01-01 01:00", "x"]
        )  # incomplete -> masked

    def test_hourly_to_daily_with_nans(self):
        # 2 days @hourly; day1 has 20/24 good (pass), day2 has 15/24 good (fail)
        idx = pd.date_range("2025-02-01", periods=96, freq="60min")
        x = np.ones(96)
        day1_bad = [1, 5, 9, 13]  # 4 NaNs -> 20/24 present
        day2_bad = [24 + i for i in range(9)]  # 9 NaNs -> 15/24 present
        x[day1_bad + day2_bad] = np.nan
        df_h = pd.DataFrame({"x": x}, index=idx)

        # df_res, filtered = _resample_and_filter(df_h, t="1D")
        completeness_mask = get_completeness_mask(
            data_frame=df_h,
            completeness_thresholds=DEFAULT_COMPLETENESS_THRESHOLDS,
            resample_offset="1D",
        )

        self.assertTrue(completeness_mask.loc["2025-02-01", "x"])  # pass
        self.assertFalse(completeness_mask.loc["2025-02-02", "x"])  # fail

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

        # df_res, filtered = _resample_and_filter(df_h, t="MS")
        completeness_mask = get_completeness_mask(
            data_frame=df_h,
            completeness_thresholds=DEFAULT_COMPLETENESS_THRESHOLDS,
            resample_offset="MS",
        )

        # MS bins at month starts
        self.assertTrue(completeness_mask.loc["2025-06-01", "x"])  # June kept
        self.assertFalse(completeness_mask.loc["2025-07-01", "x"])  # July masked

    def test_mixed_10min_then_hourly_to_hourly(self):
        # First hour: 5 samples @10min (5/6=0.833 pass), second hour: 1 hourly sample (pass)
        idx_10 = pd.date_range("2025-03-01 00:00", periods=5, freq="10min")
        idx_60 = pd.date_range("2025-03-01 02:00", periods=3, freq="60min")
        idx = idx_10.union(idx_60)
        x = np.arange(len(idx), dtype=float)
        df_h = pd.DataFrame({"x": x}, index=idx)

        # df_res, filtered = _resample_and_filter(df_h, t="60min")
        completeness_mask = get_completeness_mask(
            data_frame=df_h,
            completeness_thresholds=DEFAULT_COMPLETENESS_THRESHOLDS,
            resample_offset="60min",
        )

        self.assertTrue(completeness_mask.loc["2025-03-01 00:00", "x"])  # 5/6 -> pass
        self.assertFalse(
            completeness_mask.loc["2025-03-01 01:00", "x"]
        )  # 0 hourly -> failed
        self.assertTrue(
            completeness_mask.loc["2025-03-01 02:00", "x"]
        )  # 1 hourly -> pass

    def test_mixed_hourly_then_daily_to_daily(self):
        # Day1: 20 hourly samples (20/24=0.833 pass)
        # Day2: a single daily sample (pass, counts as 1)
        idx_hourly = pd.date_range("2025-04-01 00:00", periods=30, freq="60min")
        idx_daily = pd.date_range("2025-04-03 00:00", periods=3, freq="D")
        idx = idx_hourly.union(idx_daily)
        x = np.arange(len(idx), dtype=float)
        df_h = pd.DataFrame({"x": x}, index=idx)

        # df_res, filtered = _resample_and_filter(df_h, t="1D")
        completeness_mask = get_completeness_mask(
            data_frame=df_h,
            completeness_thresholds=DEFAULT_COMPLETENESS_THRESHOLDS,
            resample_offset="1D",
        )

        self.assertTrue(completeness_mask.loc["2025-04-01", "x"])  # 20/24 -> pass
        self.assertFalse(completeness_mask.loc["2025-04-02", "x"])  # 6/24  -> failed
        self.assertTrue(
            completeness_mask.loc["2025-04-03", "x"]
        )  # daily sample -> pass

    def test_monthly_resampling(self):
        # Monthly resampling is a special case where period lengths are uneven.
        # 2023 is chosen a non-leap year to include February with 28 days.
        index = pd.date_range("2023-01-01", "2023-12-31", freq="10min")
        df = pd.DataFrame(
            {
                "x": np.random.random(len(index)),
                "y": np.random.random(len(index)),
            },
            index=index,
        )

        # df_res, filtered = _resample_and_filter(df, t="MS")
        completeness_mask = get_completeness_mask(
            data_frame=df,
            completeness_thresholds=DEFAULT_COMPLETENESS_THRESHOLDS,
            resample_offset="MS",
        )

        # The completeness mask should be True for all entries
        self.assertTrue(completeness_mask.values.ravel().all())


if __name__ == "__main__":
    unittest.main()
