import unittest

import numpy as np
import pandas as pd

from pypromice.core.variables.surface_height import (
    _first,
    _interpolate_within,
    _window,
    combine_surface_height,
    hampel,
)


def _hampel_per_sample(vals_orig, k=7*24, t0=15):
    """Hampel filter with the window defined in number of samples
    (implementation used for hourly data before the window was defined in time)."""
    vals = vals_orig.copy()
    L = 1.4826
    rolling_median = vals.rolling(k).median()
    difference = np.abs(rolling_median - vals)
    median_abs_deviation = difference.rolling(k).median()
    threshold = t0 * L * median_abs_deviation
    outlier_idx = difference > threshold
    outlier_idx[0:round(k/2)] = False
    vals.loc[outlier_idx] = np.nan
    return vals


class TestTimeBasedHelpers(unittest.TestCase):
    def test_interpolate_within_matches_row_limit_on_hourly_data(self):
        rng = np.random.default_rng(0)
        index = pd.date_range("2020-01-01", periods=2000, freq="h")
        values = np.cumsum(rng.normal(size=len(index)))
        values[rng.random(len(index)) < 0.3] = np.nan
        values[500:700] = np.nan          # gap longer than the limit
        values[:5] = np.nan               # leading gap
        values[-5:] = np.nan              # trailing gap
        series = pd.Series(values, index=index)

        pd.testing.assert_series_equal(
            _interpolate_within(series, "3D"),
            series.interpolate(limit=72),
        )

    def test_interpolate_within_depends_on_time_not_on_rows(self):
        # 10 min data, 2 days gap: only the first 12 h after the last valid
        # value are filled with a limit of 12 h
        index = pd.date_range("2020-01-01", periods=6 * 24 * 5, freq="10min")
        series = pd.Series(1.0, index=index)
        series.iloc[6 * 24:6 * 24 * 3] = np.nan
        series.iloc[-1] = 3.0

        filled = _interpolate_within(series, "12h")
        gap = filled.iloc[6 * 24:6 * 24 * 3]
        self.assertEqual(int(gap.notnull().sum()), 6 * 12)
        self.assertTrue(gap.iloc[:6 * 12].notnull().all())

    def test_interpolate_within_irregular_time_steps(self):
        index = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 01:00",
                                  "2020-01-01 01:10", "2020-01-02 00:00"])
        series = pd.Series([0.0, np.nan, 1.0, 2.0], index=index)
        filled = _interpolate_within(series, "3D")
        # interpolated in time (60 min out of 70 min), not by row position
        self.assertAlmostEqual(filled.iloc[1], 60 / 70)

    def test_window_and_first(self):
        index = pd.date_range("2020-01-01", periods=10 * 24, freq="h")
        series = pd.Series(np.arange(len(index), dtype=float), index=index)
        window = _window(series, index[48], "1D", "2D")
        self.assertEqual(window.index[0], index[24])      # included
        self.assertEqual(window.index[-1], index[48 + 47])  # t + 2D excluded
        self.assertEqual(len(_first(series, "7D")), 7 * 24)
        self.assertEqual(len(_first(series.iloc[:0], "7D")), 0)


class TestHampel(unittest.TestCase):
    def test_same_as_per_sample_filter_on_hourly_data(self):
        rng = np.random.default_rng(1)
        index = pd.date_range("2020-01-01", periods=1500, freq="h")
        values = np.cumsum(rng.normal(0, 0.05, len(index)))
        values[rng.random(len(index)) < 0.02] += rng.normal(0, 3, 1)[0]
        values[rng.random(len(index)) < 0.05] = np.nan
        series = pd.Series(values, index=index)

        for k, t0 in [(7 * 24, 15), (24, 5)]:
            pd.testing.assert_series_equal(
                hampel(series, k=k, t0=t0),
                _hampel_per_sample(series, k=k, t0=t0),
            )

    def test_window_does_not_depend_on_time_step(self):
        # one isolated outlier in a smooth 10-min series is removed
        index = pd.date_range("2020-01-01", periods=6 * 24 * 20, freq="10min")
        series = pd.Series(np.sin(np.arange(len(index)) / 500), index=index)
        series.iloc[6 * 24 * 10] += 5
        cleaned = hampel(series, k=24, t0=5)
        self.assertTrue(np.isnan(cleaned.iloc[6 * 24 * 10]))
        self.assertEqual(int(cleaned.isnull().sum()), 1)


def _ablation_site(freq):
    """Synthetic ablation site: 3 years of PT, upper SR and stake records.

    The ice surface falls by 2.5 m every summer, snow accumulates in winter,
    the surface height records are offset when the sensors are reset in
    September and the stake record is missing for 3 weeks around that time
    (so that it only comes back well after the end of the ablation season).
    """
    index = pd.date_range("2018-01-01", "2020-12-31 23:50", freq="10min")
    t = pd.Series(index, index=index)
    doy = index.dayofyear.values + index.hour.values / 24

    ice = np.zeros(len(index))
    snow = np.zeros(len(index))
    for year in (2018, 2019, 2020):
        start = pd.Timestamp(f"{year}-06-15")
        end = pd.Timestamp(f"{year}-09-05")
        frac = np.clip((index - start) / (end - start), 0, 1)
        ice -= 2.5 * np.where(index < end, frac, 1) * (index.year == year)
        ice -= 2.5 * (index.year > year)
    snow = np.where(doy < 135, 1.0 + 0.5 * doy / 135, 0)
    snow = np.where(doy > 248, 1.5 * (doy - 248) / (365 - 248), snow)
    snow = np.where((doy >= 135) & (doy <= 166), 1.5 * (166 - doy) / 31, snow)
    snow = np.where((doy > 166) & (doy <= 248), 0, snow)

    surface = ice + snow
    rng = np.random.default_rng(0)
    offset1 = 0.8 * (index.year - 2018) + 0.8 * (doy > 253)
    offset2 = -0.5 * (index.year - 2018) - 0.5 * (doy > 253)
    df = pd.DataFrame({
        "z_surf_1": surface + offset1 + rng.normal(0, 0.01, len(index)),
        "z_surf_2": surface + offset2 + rng.normal(0, 0.01, len(index)),
        "z_ice_surf": ice + 0.3 + rng.normal(0, 0.005, len(index)),
    }, index=index)
    stake_gap = (doy > 240) & (doy < 262)
    df.loc[stake_gap, "z_surf_2"] = np.nan
    if freq == "10min":
        return df
    return df.resample(freq).mean()


class TestCombineSurfaceHeightResolution(unittest.TestCase):
    def test_result_does_not_depend_on_the_time_step(self):
        df_10min = _ablation_site("10min")
        df_hourly = _ablation_site("h")

        combined_10min, _, _, _ = combine_surface_height(df_10min.copy(), "ablation")
        combined_hourly, _, _, _ = combine_surface_height(df_hourly.copy(), "ablation")

        diff = (combined_10min.resample("h").mean() - combined_hourly).abs()
        self.assertEqual(
            (combined_10min.resample("h").mean().isnull()
             != combined_hourly.isnull()).sum(), 0)
        self.assertLess(diff.mean(), 0.02)
        # the only larger differences are the single hours at which the
        # surface height switches at the end of each ablation season
        self.assertLess(diff.quantile(0.999), 0.1)
        self.assertLessEqual(int((diff > 0.2).sum()), 3)

    def test_irregular_time_steps_run(self):
        df = _ablation_site("10min")
        rng = np.random.default_rng(3)
        irregular = df.iloc[np.sort(rng.choice(len(df), len(df) // 4, replace=False))]
        combined, _, _, _ = combine_surface_height(irregular.copy(), "ablation")
        self.assertEqual(len(combined), len(irregular))
        self.assertTrue(combined.notnull().any())


if __name__ == "__main__":
    unittest.main()
