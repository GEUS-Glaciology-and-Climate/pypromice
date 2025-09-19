import unittest
import pandas as pd
import numpy as np
import xarray as xr

from pypromice.core.variables.air_temperature import (
    clip_and_interpolate,
    get_cloud_coefficients
)


class TestClipAndInterpolate(unittest.TestCase):
    def setUp(self):
        self.time = pd.date_range("2025-08-01", periods=6, freq="1H")
        values = [1.0, 10.0, np.nan, np.nan, 7.0, 15.0]
        self.temp = xr.DataArray(values, coords=[("time", self.time)])

    def test_clip_values(self):
        result = clip_and_interpolate(self.temp, lo=2.0, hi=14.0)
        # Values outside [2, 14] should be NaN
        self.assertTrue(np.isnan(result.values[0]))   # 1.0 clipped
        self.assertTrue(np.isnan(result.values[-1]))  # 15.0 clipped
        self.assertFalse(np.isnan(result.values[-2])) # 7.0 not clipped

    def test_interpolation_within_gap(self):
        expected_output = [np.nan, 10.0, 9.0, 8.0, 7.0, np.nan]
        result = clip_and_interpolate(self.temp, lo=2.0, hi=14.0)

        # Check interpolated output
        np.testing.assert_allclose(
            result.values,
            expected_output,
            equal_nan=True)

        # Check time coordinate is preserved
        np.testing.assert_array_equal(result.time.values, self.time.values)

    def test_interpolation_exceeds_max_gap(self):
        # Use larger freq so interpolation exceeds max_interp
        time = pd.date_range("2023-01-01", periods=24, freq="1H")
        values = [1, 1, np.nan, 3, 6, 8, 9, 10,
                  50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                  1, 1, 5]
        temp = xr.DataArray(values, coords=[("time", time)])
        result = clip_and_interpolate(temp, lo=0, hi=40, max_interp=pd.Timedelta("12H"))

        # Entire range [8:20] should be NaN (clipped out, gap > 12h)
        self.assertTrue(np.all(np.isnan(result.values[8:20])))

        # Smaller gap (index 2) should be interpolated
        self.assertFalse(np.isnan(result.values[2]))

class TestCloudCoefficients(unittest.TestCase):

    def setUp(self):
        self.time = pd.date_range("2025-08-01", periods=5, freq="1H")
        self.temp = xr.DataArray([0.0, 10.0, 20.0, -5.0, np.nan],
                                 coords=[("time", self.time)])

    def test_output_types_and_shape(self):
        LR_overcast, LR_clear = get_cloud_coefficients(self.temp)
        self.assertIsInstance(LR_overcast, xr.DataArray)
        self.assertIsInstance(LR_clear, xr.DataArray)
        # Output should have same shape as input
        self.assertEqual(LR_overcast.shape, self.temp.shape)
        self.assertEqual(LR_clear.shape, self.temp.shape)

    def test_values_positive_or_nan(self):
        LR_overcast, LR_clear = get_cloud_coefficients(self.temp)
        # Values must be positive or NaN if input was NaN
        for val, t in zip(LR_overcast.values, self.temp.values):
            if np.isnan(t):
                self.assertTrue(np.isnan(val))
            else:
                self.assertGreater(val, 0)
        for val, t in zip(LR_clear.values, self.temp.values):
            if np.isnan(t):
                self.assertTrue(np.isnan(val))
            else:
                self.assertGreater(val, 0)

    def test_monotonic_increase_with_temp(self):
        # Overcast and clear radiation should increase with temperature
        temps = xr.DataArray([0, 5, 10, 15, 20])
        LR_overcast, LR_clear = get_cloud_coefficients(temps)
        self.assertTrue(np.all(np.diff(LR_overcast.values) > 0))
        self.assertTrue(np.all(np.diff(LR_clear.values) > 0))

    def test_extreme_cold(self):
        # Extremely low temperature (-50°C)
        temp = xr.DataArray([-50.0])
        LR_overcast, LR_clear = get_cloud_coefficients(temp)
        self.assertGreater(LR_overcast.item(), 0)
        self.assertGreater(LR_clear.item(), 0)

    def test_zero_temperature(self):
        # Check zero degrees Celsius
        temp = xr.DataArray([0.0])
        LR_overcast, LR_clear = get_cloud_coefficients(temp)
        self.assertGreater(LR_overcast.item(), 0)
        self.assertGreater(LR_clear.item(), 0)

if __name__ == "__main__":
    unittest.main()
