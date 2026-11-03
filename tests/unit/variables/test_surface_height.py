import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.qc.github_data_issues import (
    adjustData,
    get_dynamic_adjust_after,
)
from pypromice.core.variables.surface_height import (
    ADJUSTMENT_COLUMNS,
    ShiftRecorder,
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


class TestShiftRecorder(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2020-01-01", periods=10 * 24, freq="h")
        self.series = pd.Series(1.0, index=index)

    def test_shift_applies_from_start_onward_and_is_recorded(self):
        recorder = ShiftRecorder()
        start = self.series.index[48]
        self.assertTrue(recorder.shift(self.series, "z_surf_1", start, 0.5, "test"))
        self.assertTrue((self.series.iloc[:48] == 1.0).all())
        self.assertTrue((self.series.iloc[48:] == 1.5).all())
        self.assertEqual(len(recorder.records), 1)

    def test_shifts_before_dynamic_adjust_after_are_skipped(self):
        recorder = ShiftRecorder(self.series.index[100])
        self.assertFalse(recorder.shift(self.series, "z_surf_1", self.series.index[99], 0.5, "early"))
        self.assertTrue(recorder.shift(self.series, "z_surf_1", self.series.index[100], 0.5, "late"))
        self.assertTrue((self.series.iloc[:100] == 1.0).all())
        self.assertTrue((self.series.iloc[100:] == 1.5).all())
        self.assertEqual([r["reason"] for r in recorder.records], ["late"])

    def test_shift_without_start_does_nothing(self):
        recorder = ShiftRecorder()
        self.assertFalse(recorder.shift(self.series, "z_surf_1", None, 0.5, "test"))
        self.assertTrue((self.series == 1.0).all())

    def test_non_finite_shift_is_not_recorded(self):
        recorder = ShiftRecorder()
        recorder.shift(self.series, "z_surf_1", self.series.index[0], np.nan, "test")
        self.assertEqual(len(recorder.records), 0)

    def test_frame_has_adjustment_file_format(self):
        recorder = ShiftRecorder()
        t0 = self.series.index[10]
        recorder.shift(self.series, "z_surf_2", t0, 0.12346, "first, with a comma")
        recorder.shift(self.series, "z_surf_2", t0, 0.1, "second # one")
        recorder.shift(self.series, "z_ice_surf", t0, 1e-6, "negligible")
        df = recorder.to_frame()

        self.assertEqual(list(df.columns), ADJUSTMENT_COLUMNS)
        # the two shifts at the same time are summed, the negligible one dropped
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["variable"], "z_surf_2")
        self.assertEqual(row["adjust_function"], "add")
        self.assertAlmostEqual(row["adjust_value"], 0.2235)
        self.assertEqual(row["t0"], "2020-01-01T10:00:00+00:00")
        # no character that would break the csv when edited by hand
        self.assertNotIn(",", row["comment"])
        self.assertNotIn("#", row["comment"])

    def test_empty_frame_has_header(self):
        self.assertEqual(list(ShiftRecorder().to_frame().columns), ADJUSTMENT_COLUMNS)


def _write_adjustments(directory, station_id, rows, marker=None):
    """Write an adjustment file with the given rows and, optionally, the
    dynamic_adjust_after row at the time ``marker``."""
    if marker is not None:
        marker_row = pd.DataFrame(
            [[pd.Timestamp(marker, tz="UTC").isoformat(), "", "z_surf_combined",
              "dynamic_adjust_after", "", "reviewed", ""]],
            columns=ADJUSTMENT_COLUMNS)
        rows = pd.concat([rows.astype(object), marker_row], ignore_index=True)
    rows.to_csv(Path(directory) / f"{station_id}.csv", index=False)


def _adjust(df, directory, station_id="TST"):
    """Apply the adjustment file to a surface height dataframe, as in the pipeline."""
    ds = df.rename_axis("time").to_xarray()
    ds.attrs["station_id"] = station_id
    ds = adjustData(ds, directory, var_list=["z_surf_1", "z_surf_2", "z_ice_surf"])
    return ds.to_dataframe()[list(df.columns)]


class TestDynamicAdjustAfter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _rows(self, *specs):
        return pd.DataFrame(
            [[t0, "", var, func, val, "", ""] for t0, var, func, val in specs],
            columns=ADJUSTMENT_COLUMNS)

    def test_reads_time_of_marker(self):
        _write_adjustments(self.dir, "TST", self._rows(
            ("2019-05-01T00:00:00+00:00", "z_surf_1", "add", 0.1)),
            marker="2021-01-01T12:00:00")
        self.assertEqual(get_dynamic_adjust_after(self.dir, "TST"),
                         pd.Timestamp("2021-01-01 12:00"))

    def test_none_without_marker_or_file(self):
        _write_adjustments(self.dir, "TST", self._rows(
            ("2019-05-01T00:00:00+00:00", "z_surf_1", "add", 0.1)))
        self.assertIsNone(get_dynamic_adjust_after(self.dir, "TST"))
        self.assertIsNone(get_dynamic_adjust_after(self.dir, "OTHER"))

    def test_latest_marker_is_used(self):
        rows = self._rows(
            ("2019-05-01T00:00:00+00:00", "z_surf_combined", "dynamic_adjust_after", ""),
            ("2021-05-01T00:00:00+00:00", "z_surf_combined", "dynamic_adjust_after", ""))
        _write_adjustments(self.dir, "TST", rows)
        self.assertEqual(get_dynamic_adjust_after(self.dir, "TST"),
                         pd.Timestamp("2021-05-01"))

    def test_adjust_data_ignores_marker(self):
        index = pd.date_range("2020-01-01", periods=48, freq="h")
        df = pd.DataFrame({"z_surf_1": 1.0, "z_surf_2": 1.0, "z_ice_surf": 1.0}, index=index)
        # a marker on a variable that exists must not be applied either
        rows = self._rows(
            ("2020-01-01T10:00:00+00:00", "z_surf_1", "add", 0.5),
            ("2020-01-01T20:00:00+00:00", "z_surf_1", "dynamic_adjust_after", ""))
        _write_adjustments(self.dir, "TST", rows)
        adjusted = _adjust(df, self.dir)
        self.assertTrue((adjusted["z_surf_1"].iloc[:10] == 1.0).all())
        self.assertTrue((adjusted["z_surf_1"].iloc[10:] == 1.5).all())


def _site_variants():
    base = _ablation_site("h")
    yield "ablation", base, "ablation"

    no_pt = base.copy()
    no_pt.loc["2019", "z_ice_surf"] = np.nan
    yield "ablation without PT in 2019", no_pt, "ablation"

    no_hs2 = base.copy()
    no_hs2.loc["2019-09-01":"2020-03-01", "z_surf_2"] = np.nan
    yield "ablation without hs2 in winter", no_hs2, "ablation"

    accumulation = base[["z_surf_1", "z_surf_2"]].copy()
    yield "accumulation", accumulation, "accumulation"


class TestAutomaticAdjustmentsOffline(unittest.TestCase):
    """The shifts made automatically, once in the adjustment file, replace the
    automatic procedure."""
    # offsets are written with 0.1 mm precision
    TOLERANCE = 2e-3

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    # In the adjustment file the shifts apply to the raw series, before the
    # outlier filter and the interpolation of gaps: around a shift the step is
    # smoothed into a short ramp. Further than this from a shift, the
    # two are the same. The same goes for the values interpolated in the gap
    # of a series, when a shift falls in the gap.
    NEAR_SHIFT = pd.Timedelta("4D")

    def _assert_same(self, expected, actual, index, shift_times, measured=None):
        near = np.zeros(len(index), dtype=bool)
        for t in shift_times:
            near |= np.abs(index - t) < self.NEAR_SHIFT
        if measured is not None:
            near |= ~measured
        expected = np.asarray(expected, float)[~near]
        actual = np.asarray(actual, float)[~near]
        self.assertEqual(int((np.isnan(expected) != np.isnan(actual)).sum()), 0)
        # (z_ice_surf is all missing at accumulation sites)
        both = ~np.isnan(expected) & ~np.isnan(actual)
        if both.any():
            self.assertLess(np.abs(expected[both] - actual[both]).max(), self.TOLERANCE)

    def _assert_all_same(self, expected, actual, df, recorder):
        """Compare the outputs of combine_surface_height: z_surf_combined,
        z_ice_surf, z_surf_1_adj and z_surf_2_adj"""
        shift_times = [r["t0"] for r in recorder.records]
        measured = [None, None, None, df["z_surf_2"].notnull().values]
        for e, a, m in zip(expected, actual, measured):
            self._assert_same(e, a, df.index, shift_times, m)

    def test_adjustment_file_reproduces_automatic_result(self):
        for name, df, site_type in _site_variants():
            with self.subTest(name):
                recorder = ShiftRecorder()
                expected = combine_surface_height(df.copy(), site_type, recorder=recorder)
                self.assertGreater(len(recorder.records), 0)

                # the adjustments written offline are in the file, up to the
                # end of the record
                end = df.index[-1] + pd.Timedelta("1h")
                _write_adjustments(self.dir, "TST", recorder.to_frame(), marker=end)
                adjusted = _adjust(df, self.dir)
                T = get_dynamic_adjust_after(self.dir, "TST")
                self.assertEqual(T, end)

                recorder_static = ShiftRecorder(T)
                actual = combine_surface_height(
                    adjusted, site_type, recorder=recorder_static)

                # nothing left to adjust automatically
                self.assertEqual(len(recorder_static.records), 0)
                self._assert_all_same(expected, actual, df, recorder)

    def test_automatic_adjustments_only_after_marker(self):
        _, df, site_type = next(iter(_site_variants()))
        recorder = ShiftRecorder()
        expected = combine_surface_height(df.copy(), site_type, recorder=recorder)
        rows = recorder.to_frame()

        # the adjustments before T are in the file, the others are left to
        # the automatic procedure
        T = pd.Timestamp("2019-09-30")
        before = rows.loc[pd.to_datetime(rows["t0"], utc=True).dt.tz_localize(None) < T]
        after = rows.drop(before.index)
        self.assertGreater(len(before), 0)
        self.assertGreater(len(after), 0)

        _write_adjustments(self.dir, "TST", before, marker=T)
        adjusted = _adjust(df, self.dir)
        recorder_partial = ShiftRecorder(get_dynamic_adjust_after(self.dir, "TST"))
        actual = combine_surface_height(adjusted, site_type, recorder=recorder_partial)

        # same result, and the shifts found after T are those of the full run
        self._assert_all_same(expected, actual, df, recorder)
        found = recorder_partial.to_frame()
        self.assertEqual(list(found["variable"]), list(after["variable"]))
        self.assertEqual(list(found["t0"]), list(after["t0"]))
        np.testing.assert_allclose(found["adjust_value"], after["adjust_value"],
                                   atol=self.TOLERANCE)

    def test_written_file_can_be_read_back_by_adjust_data(self):
        _, df, site_type = next(iter(_site_variants()))
        recorder = ShiftRecorder()
        combine_surface_height(df.copy(), site_type, recorder=recorder)
        path = self.dir / "TST.csv"
        recorder.write(path)

        written = pd.read_csv(path, comment="#", skipinitialspace=True)
        self.assertEqual(list(written.columns), ADJUSTMENT_COLUMNS)
        self.assertEqual(len(written), len(recorder.to_frame()))
        # and it is a valid adjustment file
        _adjust(df, self.dir)


if __name__ == "__main__":
    unittest.main()
