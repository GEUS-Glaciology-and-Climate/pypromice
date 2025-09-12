import unittest
import numpy as np
import xarray as xr
from unittest.mock import patch

from pypromice.core.variables.radiation import (
    convert_sr,
    convert_lr,
    filter_lr,
    T_0,
    filter_sr)

class TestRadiationConversions(unittest.TestCase):
    def setUp(self):
        # Create simple xr.DataArray objects for testing
        self.sr = xr.DataArray([100.0, 200.0, 300.0], dims="time")
        self.lr = xr.DataArray([50.0, 100.0, 150.0], dims="time")
        self.t_rad = xr.DataArray([20.0, 25.0, 30.0], dims="time")

    def test_convert_sr_scaling(self):
        coef = 2.0
        result = convert_sr(self.sr, coef)
        expected = (self.sr * 10) / coef
        xr.testing.assert_allclose(result, expected)

    def test_convert_lr_calculation(self):
        coef = 5.0
        result = convert_lr(self.lr, self.t_rad, coef)
        expected = (self.lr * 10) / coef + 5.67e-8 * (self.t_rad + T_0) ** 4
        xr.testing.assert_allclose(result, expected)

    def test_filter_lr_with_missing_t_rad(self):
        t_rad_with_nan = xr.DataArray([20.0, np.nan, 30.0], dims="time")
        result = filter_lr(self.lr, t_rad_with_nan)
        expected = self.lr.where(t_rad_with_nan.notnull())
        xr.testing.assert_equal(result, expected)

    def test_filter_lr_all_valid(self):
        result = filter_lr(self.lr, self.t_rad)
        # No NaNs in t_rad, should be identical to lr
        xr.testing.assert_equal(result, self.lr)

    def test_filter_lr_all_missing(self):
        t_rad_all_nan = xr.DataArray([np.nan, np.nan, np.nan], dims="time")
        result = filter_lr(self.lr, t_rad_all_nan)
        self.assertTrue(result.isnull().all())

if __name__ == "__main__":
    unittest.main()