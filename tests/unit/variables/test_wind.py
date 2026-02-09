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
        self.time = pd.date_range("2025-08-01", periods=4, freq="h")
        self.wspd = xr.DataArray([0.0, 2.0, 4.0, 6.0], coords=[("time", self.time)])
        self.wdir = xr.DataArray([0.0, 90.0, 180.0, 270.0], coords=[("time", self.time)])
        self.ds = xr.Dataset(
                        data_vars={
                            "wspd": ("time", [0.0, 2.0, 4.0, 6.0]),
                            "wdir": ("time", [0.0, 90.0, 180.0, 270.0]),
                        },
                        coords={"time": pd.date_range("2025-08-01", periods=4, freq="h")},
                    )

    def test_correct_wind_speed(self):
        coefficient = 1.7
        result = correct_wind_speed(self.wspd, coefficient)
        expected = self.wspd.values * coefficient

        np.testing.assert_allclose(result.values, expected)
        np.testing.assert_array_equal(result.time.values, self.time.values)

    def test_filter_wind_direction(self):
        result = filter_wind_direction(self.ds, tag="")

        # Wind direction where wspd=0 should be flagged
        self.assertTrue(result.wdir_qc.isel(time=0).item() == 'ZERO_WSPD')

        # Non-zero wspd should preserve wdir
        np.testing.assert_allclose(result.wdir.values[1:], [90.0, 180.0, 270.0])

        # Time coordinate preserved
        np.testing.assert_array_equal(result.time.values, self.time.values)

    def test_filter_all_zero_wind_speeds(self):
        wspd = xr.DataArray([0.0, 0.0, 0.0, 0.0], coords=[("time", self.time)])
        wdir = xr.DataArray([10.0, 20.0, 30.0, 40.0], coords=[("time", self.time)])

        ds = xr.Dataset(
            data_vars={
                "wspd": wspd,
                "wdir": wdir,
            }
        )
        result = filter_wind_direction(ds, tag="")

        # All values should be NaN
        self.assertTrue((result.wdir_qc == "ZERO_WSPD").all())

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
