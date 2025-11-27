import unittest
import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.variables.wind import (
    correct_wind_speed,
    filter_wind_direction,
    calculate_directional_wind_speed,
)

class TestWindProcessing(unittest.TestCase):
    def setUp(self):
        self.time = pd.date_range("2025-08-01", periods=4, freq="H")
        self.wspd = xr.DataArray([0.0, 2.0, 4.0, 6.0], coords=[("time", self.time)])
        self.wdir = xr.DataArray([0.0, 90.0, 180.0, 270.0], coords=[("time", self.time)])

    def test_correct_wind_speed(self):
        coefficient = 1.7
        result = correct_wind_speed(self.wspd, coefficient)
        expected = self.wspd.values * coefficient

        np.testing.assert_allclose(result.values, expected)
        np.testing.assert_array_equal(result.time.values, self.time.values)

    def test_filter_wind_direction(self):
        result = filter_wind_direction(self.wdir, self.wspd)

        # Wind direction where wspd=0 should be NaN
        self.assertTrue(np.isnan(result.values[0]))

        # Non-zero wspd should preserve wdir
        np.testing.assert_allclose(result.values[1:], [90.0, 180.0, 270.0])

        # Time coordinate preserved
        np.testing.assert_array_equal(result.time.values, self.time.values)

    def test_filter_all_zero_wind_speeds(self):
        wspd = xr.DataArray([0.0, 0.0, 0.0, 0.0], coords=[("time", self.time)])
        wdir = xr.DataArray([10.0, 20.0, 30.0, 40.0], coords=[("time", self.time)])
        result = filter_wind_direction(wdir, wspd)

        # All values should be NaN
        self.assertTrue(np.all(np.isnan(result.values)))

    def test_calculate_directional_wind_speed(self):
        wspd_x, wspd_y = calculate_directional_wind_speed(self.wspd, self.wdir)

        # Expected values
        expected_x = self.wspd.values * np.sin(np.deg2rad(self.wdir.values))
        expected_y = self.wspd.values * np.cos(np.deg2rad(self.wdir.values))

        np.testing.assert_allclose(wspd_x.values, expected_x)
        np.testing.assert_allclose(wspd_y.values, expected_y)

        # Coordinates preserved
        np.testing.assert_array_equal(wspd_x.time.values, self.time.values)
        np.testing.assert_array_equal(wspd_y.time.values, self.time.values)

    def test_calculate_with_nan_wind_direction(self):
        wspd = xr.DataArray([1.0, 2.0, 3.0, 4.0], coords=[("time", self.time)])
        wdir = xr.DataArray([np.nan, np.nan, np.nan, np.nan], coords=[("time", self.time)])
        wspd_x, wspd_y = calculate_directional_wind_speed(wspd, wdir)

        # Both components should be all NaN
        self.assertTrue(np.all(np.isnan(wspd_x.values)))
        self.assertTrue(np.all(np.isnan(wspd_y.values)))


if __name__ == "__main__":
    unittest.main()
