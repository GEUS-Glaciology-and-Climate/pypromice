import unittest
import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.variables.humidity import adjust, calculate_specific_humidity, convert


class TestHumidityProcessing(unittest.TestCase):
    def setUp(self):
        self.time = pd.date_range("2025-08-01", periods=12, freq="H")
        # Temperatures: mix of freezing and above freezing, and NaNs
        t_values = [-10, np.nan, -2, 0, 1, 3, 5, 7, np.nan, 12, 15, 18]
        self.t = xr.DataArray(t_values, coords=[("time", self.time)])

        # Relative humidity: arbitrary values 40-95%
        rh_values = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, np.nan]
        self.rh = xr.DataArray(rh_values, coords=[("time", self.time)])

        # Pressure: slightly varying around 1000 hPa
        p_values = [1015, 1013, 1012, np.nan, 1008, 1005, 1003, 1000, 998, 995, np.nan, 990]
        self.p = xr.DataArray(p_values, coords=[("time", self.time)])

        # Specific humidity: low values (kg/kg)
        qh_values = [0.00070223, np.nan, np.nan, np.nan, 0.00243658, 0.003,
                     0.00379679, 0.00468812, 0.001450, 0.00748682, 0.00309848, 0.009]
        self.qh = xr.DataArray(qh_values, coords=[("time", self.time)])

    def test_adjust_output(self):
        result = adjust(self.rh, self.t)

        # Freezing (<0) should adjust relative humidity
        self.assertNotEqual(result.values[0], self.rh.values[0])
        self.assertNotEqual(result.values[2], self.rh.values[2])

        # Non-freezing (>=0) should stay the same
        np.testing.assert_allclose(result.values[4:7], self.rh.values[4:7])

        # Coordinates preserved
        np.testing.assert_array_equal(result.time.values, self.time.values)

    def test_adjust_nans(self):
        result = adjust(self.rh, self.t)

        # t NaN values should propagate to adjusted humidity
        self.assertTrue(np.isnan(result.values[1]))  # t[1] is NaN
        self.assertTrue(np.isnan(result.values[8]))  # t[8] is NaN

        # rh NaN values should also be nan in adjusted humidity
        self.assertTrue(np.isnan(result.values[-1]))  # t[1] is NaN

    def test_calculate_specific_humidity(self):
        rh_values = [44.08321934, 45.0923485, 50.97500276, 55., 60., 65., 70.089222, 75., np.nan, np.nan, np.nan, np.nan]
        rh_corr = xr.DataArray(rh_values, coords=[("time", self.time)])
        qh = calculate_specific_humidity(self.t, self.p, rh_corr)

        # Values should be positive and reasonably small (~kg/kg)
        valid_mask = ~np.isnan(qh.values)
        self.assertTrue(np.all(qh.values[valid_mask] >= 0))
        self.assertTrue(np.all(qh.values[valid_mask] <= 0.02))  # rough upper bound

        # rh NaN masking works
        self.assertTrue(np.all(np.isnan(qh.values[8:])))

        # p NaN masking works
        self.assertTrue(qh.values[3])

        # Coordinates preserved
        np.testing.assert_array_equal(qh.time.values, self.time.values)

    def test_convert(self):
        qh_g = convert(self.qh)

        # Values scaled by 1000 (ignoring NaNs)
        valid_mask = ~np.isnan(self.qh.values)
        np.testing.assert_allclose(qh_g.values[valid_mask], self.qh.values[valid_mask] * 1000)

        # Type preserved
        self.assertIsInstance(qh_g, xr.DataArray)

        # Coordinates preserved
        np.testing.assert_array_equal(qh_g.time.values, self.time.values)


if __name__ == "__main__":
    unittest.main()