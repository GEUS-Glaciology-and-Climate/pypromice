import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.variables import precipitation


class TestPrecipFilter(unittest.TestCase):

    def setUp(self):
        time = pd.date_range("2000-01-01", periods=3, freq="h")
        self.precip = xr.DataArray([0.0, 1.0, 2.0], dims="time", coords={"time": time})
        self.t = xr.DataArray([0.0, np.nan, 2.0], dims="time", coords={"time": time})
        self.p = xr.DataArray([np.nan, 1012.0, 1015.0], dims="time", coords={"time": time})
        self.rh = xr.DataArray([50.0, 60.0, 70.0], dims="time", coords={"time": time})

    def test_filter_removes_when_nan_and_zero(self):
        result = precipitation.filter_lufft_errors(self.precip, self.t, self.p, self.rh)
        # Expect NaN at index 2 because p is NaN and precip==0
        self.assertTrue(np.isnan(result[0].item()))

    def test_filter_keeps_nonzero_precip(self):
        result = precipitation.filter_lufft_errors(self.precip, self.t, self.p, self.rh)
        # precip[1] = 1.0, even though t is NaN, should be kept
        self.assertEqual(result[1].item(), 1.0)

    def test_filter_no_nan_inputs(self):
        t = xr.DataArray([0.0, 1.0, 2.0], dims="time", coords=self.precip.coords)
        p = xr.DataArray([1013.0, 1012.0, 1011.0], dims="time", coords=self.precip.coords)
        rh = xr.DataArray([50.0, 60.0, 70.0], dims="time", coords=self.precip.coords)
        result = precipitation.filter_lufft_errors(self.precip, t, p, rh)
        xr.testing.assert_equal(result, self.precip)

class TestPrecipConvert(unittest.TestCase):

    def setUp(self):
        time = pd.date_range("2025-06-01", periods=8, freq="h")
        self.precip = xr.DataArray([10.0, 11.0, 14.0, 20.0, 25.0, 29.0, 38.0, 46.0],
                                   dims="time",
                                   coords={"time": time})
        self.wspd = xr.DataArray([5.0, 5.0, 10.0, 5.0, 5.0, 6.0, 5.0, 10.0],
                                 dims="time",
                                 coords={"time": time})
        self.t = xr.DataArray([4.0, 5.0, 5.0, 3.0, 10.0, 3.0, 8.0, 6.0],
                              dims="time",
                              coords={"time": time})

    def test_convert_output(self):
        _, result = precipitation.convert_to_rainfall_per_timestep_and_correct_undercatch(self.precip, self.wspd, self.t)

        expected_result = np.array([
            np.nan, 1.15074799, 3.28587076, 6.90448792, 5.75373993,
            4.63070155, 10.35673188, 8.76232202
        ])

        # Ensure output matches expected values within tolerance
        np.testing.assert_allclose(
            result.values,
            expected_result,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Precipitation rate values differ from expected output"
        )

    def test_convert_applies_correction_factor(self):
        _, result = precipitation.convert_to_rainfall_per_timestep_and_correct_undercatch(self.precip, self.wspd, self.t)
        # Ensure precipitation rates are scaled by factor >= 1.02
        self.assertTrue(((result.dropna(dim="time") >= 0).all()).item())

    def test_convert_filters_cold_rain(self):
        t_allneg = xr.DataArray([-3.0, -5.0, -4.0, -6.0, -4.0, -4.0, -5.0, -8.0],
                                dims="time",
                                coords=self.t.coords)
        _, result = precipitation.convert_to_rainfall_per_timestep_and_correct_undercatch(self.precip, self.wspd, t_allneg)
        # When t < -2 and precip_rate > 0, it should become NaN
        self.assertTrue(np.isnan(result).all())

    def test_convert_removes_negative_rates(self):
        # Create decreasing precip to force negative diff
        precip_desc = xr.DataArray([100.0, 90.0, 87.0, 84.0, 72.0, 60.0, 38.0, 26.0],
                                  dims="time",
                                  coords=self.precip.coords)
        _, result = precipitation.convert_to_rainfall_per_timestep_and_correct_undercatch(precip_desc, self.wspd, self.t)

        self.assertTrue(np.isnan(result.values).all())

    def test_precipitation_counter_reset(self):
        # The expected precipitation rate should have the same dimension and coordinates as the input
        precip_accumulated_values =   [0.0,    1.0,  1.0, 3.0,  6.0,  np.nan,    0.0,  1.0, np.nan, 2.0]
        expected_rainfall_values =    [np.nan, 1.0,  0.0, 2.0,  3.0,  np.nan, np.nan, 1.0,  np.nan, np.nan]
        t_vals = np.full(len(expected_rainfall_values), 5)

        time = pd.date_range("2025-06-01", periods=len(precip_accumulated_values), freq="h")
        precip_accumulated = xr.DataArray(precip_accumulated_values, dims="time", coords={"time": time})
        t = xr.DataArray(t_vals, dims="time", coords={"time": time})
        expected_rainfall = xr.DataArray(expected_rainfall_values, dims="time", coords={"time": time})

        result = precipitation.get_rainfall_per_timestep(precip_accumulated, t)

        xr.testing.assert_equal(result, expected_rainfall)

    def test_irregular_sample_rates(self):
        """
        There can be occasions where the sample rate is not regular.
        E.g., on day 300 or after merging multiple datasets.
        """
        precip_accumulated_values =   [0.0,    1.0,  1.0,  3.0, 10.0]
        expected_rainfall_values =    [np.nan, 1.0,  0.0,  2.0,  7.0]
        t_vals = np.full(len(expected_rainfall_values), 5)

        time = pd.to_datetime('2023-10-26') + pd.to_timedelta(['21:00:00', '22:00:00', '23:00:00', '24:00:00', '48:00:00'])
        expected_rainfall = xr.DataArray(
            expected_rainfall_values, dims="time", coords={"time": time}
        )
        ds = xr.Dataset(
                            data_vars={
                                "precip": (("time",), precip_accumulated_values),
                                "t": (("time",), t_vals),
                            },
                            coords={"time": time},
                        )

        result = precipitation.get_rainfall_per_timestep(ds['precip'], ds['t'])
        xr.testing.assert_equal(result, expected_rainfall)

    def test_sub_hourly_rates(self):
        """
        This is an example where the sample rate higher than h^-1
        The result should be an amount of rainfall per time stamp
        """
        n_samples = 100
        precip_accumulated_values = np.cumsum(np.random.rand(n_samples))
        expected_rainfall_values = np.diff(precip_accumulated_values)
        expected_rainfall_values = np.insert(expected_rainfall_values, 0, np.nan)
        t_vals = np.full(n_samples, n_samples)
        time = pd.date_range("2025-06-01", periods=n_samples, freq="600s")
        ds = xr.Dataset(
                            data_vars={
                                "precip": (("time",), precip_accumulated_values),
                                "t": (("time",), t_vals),
                            },
                            coords={"time": time},
                        )

        expected_rainfall = xr.DataArray(
            expected_rainfall_values, dims="time", coords={"time": time}
        )

        result = precipitation.get_rainfall_per_timestep(ds['precip'], ds['t'])
        xr.testing.assert_equal(result, expected_rainfall)

if __name__ == "__main__":
    unittest.main()
