import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.qc import persistence
from pypromice.core.qc.persistence import find_persistent_regions, persistence_qc


def _qc_has_flag(qc_da: xr.DataArray, flag_value: int) -> np.ndarray:
    return (qc_da.values.astype("int64") & int(flag_value)) != 0


def _make_ds(input_series: pd.Series) -> xr.Dataset:
    n = len(input_series)
    return xr.Dataset(
        data_vars=dict(
            t_u=(("time",), input_series.values.astype("float64")),
            t_u_qc=(("time",), np.full(n, "OK", dtype=object)),
        ),
        coords=dict(time=input_series.index),
    )


class PersistenceQATestCase(unittest.TestCase):
    def test_1_hour_persistent(self):
        self._test_1_hour_repeat(10)

    def test_1_hour_second_index(self):
        self._test_1_hour_repeat(0)

    def test_1_hour_last_index(self):
        self._test_1_hour_repeat(-2)

    def _test_1_hour_repeat(self, index: int):
        self.assertTrue(index < -1 or index >= 0)
        time_range = pd.date_range(
            start="2023-01-26", end="2023-01-27", freq="h", inclusive="left"
        )
        input_series = pd.Series(index=time_range, data=np.arange(0, len(time_range)))
        input_series.iloc[index + 1] = input_series.iloc[index]
        min_repeats = 1

        ds = _make_ds(input_series)
        ds_out = persistence_qc(ds, variable_thresholds={"t_u": {"max_diff": 0.001, "period": min_repeats}})

        expected_output = input_series.map(lambda _: False)
        expected_output.iloc[index + 1] = True
        got = (ds_out["t_u_qc"].values == 'PERSISTENCE')

        np.testing.assert_array_equal(expected_output.values.astype(bool), got)

    def test_no_persistent_period(self):
        time_range = pd.date_range(
            start="2023-01-26", end="2023-01-27", freq="h", inclusive="left"
        )
        input_series = pd.Series(index=time_range, data=np.arange(0, len(time_range)))
        min_repeats = 1

        ds = _make_ds(input_series)
        ds_out = persistence_qc(ds, variable_thresholds={"t_u": {"max_diff": 0.001, "period": min_repeats}})

        expected_output = input_series.map(lambda _: False)
        got = (ds_out["t_u_qc"].values == 'PERSISTENCE')

        np.testing.assert_array_equal(expected_output.values.astype(bool), got)

    def test_persistent_period_longer_than_period_threshold(self):
        time_range = pd.date_range(
            start="2023-01-26", end="2023-01-28", freq="h", inclusive="left"
        )
        index_start = 23
        index_end = 33
        min_repeats = 4
        expected_filter_start = 24
        expected_filter_end = 33
        input_series = pd.Series(index=time_range, data=np.arange(0, len(time_range)))
        input_series.iloc[index_start:index_end] = input_series.iloc[index_start]

        ds = _make_ds(input_series)
        ds_out = persistence_qc(ds, variable_thresholds={"t_u": {"max_diff": 0.001, "period": min_repeats}})

        expected_output = input_series.map(lambda _: False)
        expected_output.iloc[expected_filter_start:expected_filter_end] = True
        got = (ds_out["t_u_qc"].values == 'PERSISTENCE')

        np.testing.assert_array_equal(expected_output.values.astype(bool), got)

    def test_period_threshold_longer_than_persistent_period(self):
        time_range = pd.date_range(
            start="2023-01-26", end="2023-01-28", freq="h", inclusive="left"
        )
        index_start = 23
        index_end = 27
        min_repeats = 10
        input_series = pd.Series(index=time_range, data=np.arange(0, len(time_range)))
        input_series.iloc[index_start:index_end] = input_series.iloc[index_start]

        ds = _make_ds(input_series)
        ds_out = persistence_qc(ds, variable_thresholds={"t_u": {"max_diff": 0.001, "period": min_repeats}})

        expected_output = input_series.map(lambda _: False)
        got = (ds_out["t_u_qc"].values == 'PERSISTENCE')

        np.testing.assert_array_equal(expected_output.values.astype(bool), got)

    def test_persistent_period_at_the_end(self):
        time_range = pd.date_range(
            start="2023-01-26", end="2023-01-28", freq="h", inclusive="left"
        )
        index_start = 23
        min_repeats = 4
        expected_filter_start = 24
        input_series = pd.Series(index=time_range, data=np.arange(0, len(time_range)))
        input_series.iloc[index_start:] = input_series.iloc[index_start]

        ds = _make_ds(input_series)
        ds_out = persistence_qc(ds, variable_thresholds={"t_u": {"max_diff": 0.001, "period": min_repeats}})

        expected_output = input_series.map(lambda _: False)
        expected_output[expected_filter_start:] = True
        got = (ds_out["t_u_qc"].values == 'PERSISTENCE')

        np.testing.assert_array_equal(expected_output.values.astype(bool), got)

    def test_dont_filter_nan_values(self):
        time_range = pd.date_range(
            start="2023-01-26", end="2023-01-27", freq="h", inclusive="left"
        )
        input_series = pd.Series(
            index=time_range, data=np.zeros_like(time_range, dtype="float")
        )
        min_repeats = 4
        input_series.iloc[:] = np.nan
        input_series.iloc[9] = -11
        input_series.iloc[10:12] = -10
        input_series.iloc[15] = -9

        ds = _make_ds(input_series)
        ds_out = persistence_qc(ds, variable_thresholds={"t_u": {"max_diff": 0.001, "period": min_repeats}})

        expected_output = input_series.map(lambda _: False)
        got = (ds_out["t_u_qc"].values == 'PERSISTENCE')

        np.testing.assert_array_equal(expected_output.values.astype(bool), got)

    def test_series_with_nan_values_between_persistent_values(self):
        time_range = pd.date_range(
            start="2023-01-26", end="2023-01-27", freq="h", inclusive="left"
        )
        values = np.zeros_like(time_range, dtype="float")
        values[:] = np.nan
        values[9] = -11
        values[16] = -11
        values[17] = -9
        series = pd.Series(index=time_range, data=values)
        expected_mask = np.zeros_like(values, dtype="bool")
        period = 4
        expected_mask[16] = True

        ds = _make_ds(series)
        ds_out = persistence_qc(ds, variable_thresholds={"t_u": {"max_diff": 0.01, "period": period}})

        got = (ds_out["t_u_qc"].values == 'PERSISTENCE')
        np.testing.assert_array_equal(expected_mask, got)

    def test_get_duration_consecutive_true(self):
        delta_time_hours = np.random.random(24) * 2
        time_range = pd.to_datetime("2023-01-25") + pd.to_timedelta(
            delta_time_hours.cumsum(), unit="h"
        )
        values = time_range == False
        values[0:2] = True
        values[6] = True
        values[10:14] = True
        values[-3:] = True
        series = pd.Series(index=time_range, data=values)

        duration_consecutive_true = persistence.get_duration_consecutive_true(series)

        self.assertTrue(
            np.isnan(duration_consecutive_true[0]), "The first index should be ignored"
        )
        np.testing.assert_almost_equal(
            duration_consecutive_true.iloc[1],
            delta_time_hours[1],
        )
        np.testing.assert_almost_equal(
            duration_consecutive_true.iloc[6],
            delta_time_hours[6],
        )
        np.testing.assert_almost_equal(
            duration_consecutive_true.iloc[10:14],
            delta_time_hours[10:14].cumsum(),
        )
        np.testing.assert_almost_equal(
            duration_consecutive_true.iloc[-3:],
            delta_time_hours[-3:].cumsum(),
        )


if __name__ == "__main__":
    unittest.main()
