import unittest
import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.variables.gps import (
    decode_and_convert,
    filter,
    piecewise_smoothing_and_interpolation,
)

class TestFilter(unittest.TestCase):

    def test_outside_threshold(self):
        """Test that values far from the baseline are all masked."""
        time = pd.date_range("2025-01-01", periods=3, freq="D")
        gps_lat = xr.DataArray([1, 2, 3], dims=["time"], coords={"time": time})
        gps_lon = xr.DataArray([4, 5, 6], dims=["time"], coords={"time": time})
        # Third value differ from baseline by >100
        gps_alt = xr.DataArray([1, 1, 10], dims=["time"], coords={"time": time}) * 200  # baseline = 200 → 3 is outside ±100
        lat_f, lon_f, alt_f = filter(gps_lat, gps_lon, gps_alt)
        self.assertTrue(np.isnan(alt_f.values[-1]))
        self.assertTrue(np.isnan(lat_f.values[-1]))
        self.assertTrue(np.isnan(lon_f.values[-1]))

    def test_all_within_threshold(self):
        """Test that values close to baseline are preserved."""
        time = pd.date_range("2025-01-01", periods=3, freq="D")
        gps_lat = xr.DataArray([1, 2, 3], dims=["time"], coords={"time": time})
        gps_lon = xr.DataArray([4, 5, 6], dims=["time"], coords={"time": time})
        gps_alt = xr.DataArray([1000, 1005, 995], dims=["time"], coords={"time": time})

        lat_f, lon_f, alt_f = filter(gps_lat, gps_lon, gps_alt)
        self.assertFalse(np.any(np.isnan(alt_f)))
        self.assertFalse(np.any(np.isnan(lat_f)))
        self.assertFalse(np.any(np.isnan(lon_f)))

    def test_single_value(self):
        """Test that a single value array is handled correctly."""
        time = pd.date_range("2025-01-01", periods=1)
        gps_lat = xr.DataArray([10], dims=["time"], coords={"time": time})
        gps_lon = xr.DataArray([20], dims=["time"], coords={"time": time})
        gps_alt = xr.DataArray([1000], dims=["time"], coords={"time": time})

        lat_f, lon_f, alt_f = filter(gps_lat, gps_lon, gps_alt)
        self.assertFalse(np.isnan(alt_f.sel(time=time[0])))
        self.assertFalse(np.isnan(lat_f.sel(time=time[0])))
        self.assertFalse(np.isnan(lon_f.sel(time=time[0])))

    def test_multiple_months_resample(self):
        """Test that monthly median resampling works correctly across multiple months."""
        time = pd.date_range("2025-01-01", periods=60, freq="D")  # two months
        gps_lat = xr.DataArray(np.arange(60), dims=["time"], coords={"time": time})
        gps_lon = xr.DataArray(np.arange(60, 120), dims=["time"], coords={"time": time})
        gps_alt = xr.DataArray([1000]*30 + [1200]*30, dims=["time"], coords={"time": time})

        lat_f, lon_f, alt_f = filter(gps_lat, gps_lon, gps_alt)
        
        # First month should be kept (1000 ±100), second month values outside threshold (1200 vs median 1000) → masked
        first_month_mask = np.isnan(alt_f.sel(time=slice("2025-01-20","2025-01-30")))
        second_month_mask = np.isnan(alt_f.sel(time=slice("2025-02-01","2025-02-10")))

        self.assertTrue(np.all(first_month_mask))
        self.assertFalse(np.all(second_month_mask))

if __name__ == "__main__":
    unittest.main()

