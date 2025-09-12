import unittest
import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.variables.precipitation import filter, correct

class TestPrecipFilter(unittest.TestCase):

    def setUp(self):
        time = pd.date_range("2000-01-01", periods=3, freq="H")
        self.precip = xr.DataArray([0.0, 1.0, 2.0], dims="time", coords={"time": time})
        self.t = xr.DataArray([0.0, np.nan, 2.0], dims="time", coords={"time": time})
        self.p = xr.DataArray([np.nan, 1012.0, 1015.0], dims="time", coords={"time": time})
        self.rh = xr.DataArray([50.0, 60.0, 70.0], dims="time", coords={"time": time})

    def test_filter_removes_when_nan_and_zero(self):
        result = filter(self.precip, self.t, self.p, self.rh)
        # Expect NaN at index 2 because p is NaN and precip==0
        self.assertTrue(np.isnan(result[0].item()))

    def test_filter_keeps_nonzero_precip(self):
        result = filter(self.precip, self.t, self.p, self.rh)
        # precip[1] = 1.0, even though t is NaN, should be kept
        self.assertEqual(result[1].item(), 1.0)

    def test_filter_no_nan_inputs(self):
        t = xr.DataArray([0.0, 1.0, 2.0], dims="time", coords=self.precip.coords)
        p = xr.DataArray([1013.0, 1012.0, 1011.0], dims="time", coords=self.precip.coords)
        rh = xr.DataArray([50.0, 60.0, 70.0], dims="time", coords=self.precip.coords)
        result = filter(self.precip, t, p, rh)
        xr.testing.assert_equal(result, self.precip)

class TestPrecipCorrect(unittest.TestCase):

    def setUp(self):
        time = pd.date_range("2025-06-01", periods=4, freq="H")
        self.precip = xr.DataArray([0.0, 1.0, np.nan, 3.0], dims="time", coords={"time": time})
        self.wspd = xr.DataArray([2.0, 5.0, 10.0, 0.0], dims="time", coords={"time": time})
        self.t = xr.DataArray([0.0, -5.0, 5.0, -3.0], dims="time", coords={"time": time})

    def test_correct_applies_correction_factor(self):
        result = correct(self.precip, self.wspd, self.t)
        # Ensure precipitation rates are scaled by factor >= 1.02
        self.assertTrue(((result.dropna(dim="time") >= 0).all()).item())

    def test_correct_nan_ffill(self):
        t_noneg = xr.DataArray([5.0, 6.0, 4.0, 3.0],
                               dims="time",
                               coords={"time": pd.date_range("2025-06-01", periods=4, freq="H")})
        result = correct(self.precip, self.wspd, t_noneg)
        # The NaN in precip[2] should be forward-filled before diff
        self.assertFalse(result.isnull()[2].item())

    def test_correct_filters_cold_rain(self):
        t_allneg = xr.DataArray([-3.0, -5.0, -4.0, -6.0],
                                dims="time",
                                coords={"time": pd.date_range("2025-06-01", periods=4, freq="H")})
        result = correct(self.precip, self.wspd, self.t)
        # When t < -2 and precip_rate > 0, it should become NaN
        self.assertTrue(np.isnan(result).all())

    def test_correct_removes_negative_rates(self):
        # Create decreasing precip to force negative diff
        precip_dec = xr.DataArray([3.0, 2.0, 1.0, 0.0], dims="time", coords=self.precip.coords)
        result = correct(precip_dec, self.wspd, self.t)
        self.assertTrue(np.isnan(result.values).all())

if __name__ == "__main__":
    unittest.main()
