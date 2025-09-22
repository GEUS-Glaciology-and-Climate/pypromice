import unittest
import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.variables.station_boom_height import (
    adjust,
    adjust_and_include_uncorrected_values
)

class TestStationBoomHeight(unittest.TestCase):

    def setUp(self):
        # Create a small test dataset
        self.time = pd.date_range("2025-08-01", periods=5, freq="1H")
        self.z_boom = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0],
                                   coords=[("time", self.time)])
        self.air_temp = xr.DataArray([0.0, 10.0, -5.0, np.nan, 20.0],
                                     coords=[("time", self.time)])

    def test_adjust_basic(self):
        result = adjust(self.z_boom, self.air_temp)
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.shape, self.z_boom.shape)

        # Check formula manually for first element (air temp = 0°C)
        expected = 1.0 * ((0.0 + 273.15)/273.15)**0.5
        self.assertAlmostEqual(result[0].item(), expected)

        # Check positive, negative, and large temperatures
        expected_second = 2.0 * ((10.0 + 273.15)/273.15)**0.5
        self.assertAlmostEqual(result[1].item(), expected_second)

        expected_third = 3.0 * ((-5.0 + 273.15)/273.15)**0.5
        self.assertAlmostEqual(result[2].item(), expected_third)

    def test_negative_air_temperature(self):
        air_temp_neg = xr.DataArray([-20.0, -10.0, -5.0, -1.0, -300.0], coords=[("time", self.time)])
        result = adjust(self.z_boom, air_temp_neg)

        # Last value would be negative inside sqrt == NaN
        self.assertTrue(np.all(np.isfinite(result.values[:-1])))

        # Last element is effectively sqrt(-0.01), so should be NaN
        self.assertTrue(np.isnan(result[-1].item()))

    def test_adjust_and_include_uncorrected_values_basic(self):
        result = adjust_and_include_uncorrected_values(self.z_boom, self.air_temp)
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.shape, self.z_boom.shape)

        # For NaN air temp (index 3), uncorrected value should be returned
        self.assertAlmostEqual(result[3].item(), self.z_boom[3].item())

        # Other indices should be adjusted
        for i in [0, 1, 2, 4]:
            expected = self.z_boom[i] * ((self.air_temp[i] + 273.15)/273.15)**0.5
            self.assertAlmostEqual(result[i].item(), expected)

    def test_all_nan_air_temperature(self):
        air_temp_nan = xr.DataArray([np.nan]*5, coords=[("time", self.time)])
        result = adjust_and_include_uncorrected_values(self.z_boom, air_temp_nan)

        # Should return the original z_boom values
        np.testing.assert_array_equal(result.values, self.z_boom.values)
        
    def test_shape_and_time_coords_preserved(self):
        result = adjust(self.z_boom, self.air_temp)
        np.testing.assert_array_equal(result.time.values, self.time.values)

        result2 = adjust_and_include_uncorrected_values(self.z_boom, self.air_temp)
        np.testing.assert_array_equal(result2.time.values, self.time.values)

if __name__ == "__main__":
    unittest.main()
