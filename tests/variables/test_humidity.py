import unittest
import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.variables.humidity import correct, calculate_specific_humidity, convert


class TestHumidityProcessing(unittest.TestCase):
    def setUp(self):
        self.time = pd.date_range("2025-08-01", periods=12, freq="H")
        # Temperatures: mix of freezing and above freezing
        t_values = [-10, -5, -2, 0, 1, 3, 5, 7, 10, 12, 15, 18]
        # Introduce some NaNs
        t_values[1] = np.nan
        t_values[8] = np.nan
        self.t = xr.DataArray(t_values, coords=[("time", self.time)])

        # Relative humidity: arbitrary values 40-95%
        rh_values = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        self.rh = xr.DataArray(rh_values, coords=[("time", self.time)])

        # Pressure: slightly varying around 1000 hPa
        p_values = [1015, 1013, 1012, 1010, 1008, 1005, 1003, 1000, 998, 995, 992, 990]
        # Introduce NaNs
        p_values[3] = np.nan
        p_values[10] = np.nan
        self.p = xr.DataArray(p_values, coords=[("time", self.time)])

    def test_correct(self):
        result = correct(self.rh, self.t)

        # Freezing (<0) should adjust relative humidity (ignoring NaNs)
        self.assertNotEqual(result.values[0], self.rh.values[0])
#        self.assertTrue(np.isnan(result.values[1]))  # t[1] is NaN

        # Non-freezing (>=0) should stay the same
        np.testing.assert_allclose(result.values[4:7], self.rh.values[4:7])

        # Coordinates preserved
        np.testing.assert_array_equal(result.time.values, self.time.values)

    def test_calculate_specific_humidity(self):
        rh_corr = correct(self.rh, self.t)
        qh = calculate_specific_humidity(self.t, self.p, rh_corr)

        # Values should be positive and reasonably small (~kg/kg)
        valid_mask = ~np.isnan(qh.values)
        self.assertTrue(np.all(qh.values[valid_mask] >= 0))
        self.assertTrue(np.all(qh.values[valid_mask] <= 0.02))  # rough upper bound

        # NaN masking works
        self.assertTrue(np.isnan(qh.values[1]))  # t[1] was NaN
        self.assertTrue(np.isnan(qh.values[3]))  # p[3] was NaN
        self.assertTrue(np.isnan(qh.values[8]))  # t[8] was NaN
        self.assertTrue(np.isnan(qh.values[10])) # p[10] was NaN

        # Coordinates preserved
        np.testing.assert_array_equal(qh.time.values, self.time.values)

    def test_convert(self):
        rh_corr = correct(self.rh, self.t)
        qh = calculate_specific_humidity(self.t, self.p, rh_corr)
        qh_g = convert(qh)

        # Values scaled by 1000 (ignoring NaNs)
        valid_mask = ~np.isnan(qh.values)
        np.testing.assert_allclose(qh_g.values[valid_mask], qh.values[valid_mask] * 1000)

        # Type preserved
        self.assertIsInstance(qh_g, xr.DataArray)

        # Coordinates preserved
        np.testing.assert_array_equal(qh_g.time.values, self.time.values)


if __name__ == "__main__":
    unittest.main()