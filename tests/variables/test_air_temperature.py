import unittest
import pandas as pd
import numpy as np
import xarray as xr

from pypromice.core.variables.air_temperature import clip_and_interpolate


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


if __name__ == "__main__":
    unittest.main()
