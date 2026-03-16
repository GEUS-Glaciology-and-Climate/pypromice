import unittest
import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.variables.gps import (
    decode_and_convert,
    decode,
    filter
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

class TestDecode(unittest.TestCase):

    def setUp(self):
        self.time = pd.date_range("2025-01-01", periods=3, freq="D")

    def test_object_decoder_branch(self):
        """If sample contains 'NH', gps_object_decoder is used for all arrays."""
        gps_lat = xr.DataArray(["NH7950.08638", "NH7951.08638"], dims=["time"], coords={"time": self.time[:2]})
        gps_lon = xr.DataArray(["WH02509.85174", "WH02505.85174"], dims=["time"], coords={"time": self.time[:2]})
        gps_time = xr.DataArray(["GT121322.00", "GT121321.00"], dims=["time"], coords={"time": self.time[:2]})

        lat, lon, time = decode(gps_lat, gps_lon, gps_time)

        np.testing.assert_array_equal(lat.values, [7950.08638, 7951.08638])
        np.testing.assert_array_equal(lon.values, [2509.85174, 2505.85174])
        np.testing.assert_array_equal(time.values, [121322.00, 121321.00])

    def test_l_string_decoder_branch(self):
        """If sample contains 'L', gps_l_string_decoder is used for lat/lon and gps_object_decoder for time."""
        gps_lat = xr.DataArray(["L6500000", "L6600000"], dims=["time"], coords={"time": self.time[:2]})
        gps_lon = xr.DataArray(["L04000000", "L04100000"], dims=["time"], coords={"time": self.time[:2]})
        gps_time = xr.DataArray(["L100", "L200"], dims=["time"], coords={"time": self.time[:2]})

        lat, lon, time = decode(gps_lat, gps_lon, gps_time)

        np.testing.assert_array_equal(lat.values, [65., 66.])
        np.testing.assert_array_equal(lon.values, [40., 41.])
        np.testing.assert_array_equal(time.values, [100, 200])

    def test_fallback_branch(self):
        """If no 'NH' or 'L' in sample, fallback to gps_object_decoder for all."""
        gps_lat = xr.DataArray(["6628.93936", "6628.93940"], dims=["time"], coords={"time": self.time[:2]})
        gps_lon = xr.DataArray(["04617.59187", "04617.59190"], dims=["time"], coords={"time": self.time[:2]})
        gps_time = xr.DataArray(["100", "200"], dims=["time"], coords={"time": self.time[:2]})

        lat, lon, time = decode(gps_lat, gps_lon, gps_time)

        np.testing.assert_array_equal(lat.values, [6628.93936, 6628.93940])
        np.testing.assert_array_equal(lon.values, [04617.59187, 04617.59190])
        np.testing.assert_array_equal(time.values, [100, 200])

    def test_error_handling_returns_none(self):
        """If decoder raises an error, decode should return (None, None, None)."""
        gps_lat = xr.DataArray(["bad"], dims=["time"], coords={"time": [self.time[0]]})
        gps_lon = xr.DataArray(["bad"], dims=["time"], coords={"time": [self.time[0]]})
        gps_time = xr.DataArray(["bad"], dims=["time"], coords={"time": [self.time[0]]})

        lat, lon, time = decode(gps_lat, gps_lon, gps_time)

        self.assertIsNone(lat)
        self.assertIsNone(lon)
        self.assertIsNone(time)

class TestDecodeAndConvert(unittest.TestCase):

    def setUp(self):
        self.time = pd.date_range("2025-01-01", periods=3, freq="D")

    def test_decimal_minutes_conversion(self):
        """Values that look like decimal minutes should be converted correctly."""
        gps_lat = xr.DataArray([65.1234, 65.5678, 65.9999], dims=["time"], coords={"time": self.time})
        gps_lon = xr.DataArray([-40.1234, -40.5678, -40.9999], dims=["time"], coords={"time": self.time})
        gps_time = xr.DataArray([1, 2, 3], dims=["time"], coords={"time": self.time})

        lat, lon, time = decode_and_convert(gps_lat, gps_lon, gps_time, latitude=65.0, longitude=-40.0)

        self.assertEqual(lat.dims, ("time",))
        self.assertEqual(lon.dims, ("time",))
        self.assertEqual(time.dims, ("time",))
        self.assertEqual(len(lat), 3)

    def test_ddmm_conversion(self):
        """Values that look like degrees+decimal minutes should be converted correctly."""
        gps_lat = xr.DataArray([6012.345, 6024.567, 6036.789], dims=["time"], coords={"time": self.time})
        gps_lon = xr.DataArray([12024.567, 12036.789, 12048.901], dims=["time"], coords={"time": self.time})
        gps_time = xr.DataArray([100, 200, 300], dims=["time"], coords={"time": self.time})

        lat, lon, time = decode_and_convert(gps_lat, gps_lon, gps_time, latitude=65.0, longitude=-40.0)

        self.assertEqual(lat.dims, ("time",))
        self.assertEqual(lon.dims, ("time",))
        self.assertEqual(time.dims, ("time",))
        self.assertTrue(np.all(np.isfinite(lat)))
        self.assertTrue(np.all(np.isfinite(lon)))

    def test_string_inputs_are_coerced(self):
        """String inputs should be coerced to numeric, invalid strings → NaN."""
        gps_lat = xr.DataArray(["65.1234", "bad", "66.0"], dims=["time"], coords={"time": self.time})
        gps_lon = xr.DataArray(["-40.1234", "-41.0", "oops"], dims=["time"], coords={"time": self.time})
        gps_time = xr.DataArray(["1", "2", "3"], dims=["time"], coords={"time": self.time})

        lat, lon, time = decode_and_convert(gps_lat, gps_lon, gps_time, latitude=65.0, longitude=-40.0)

        self.assertTrue(np.isnan(lat.values[1]))
        self.assertTrue(np.isnan(lon.values[2]))
        self.assertEqual(len(lat), 3)
        self.assertEqual(len(lon), 3)
        self.assertEqual(len(time), 3)

    def test_attributes_are_preserved(self):
        """Array attributes should be preserved after conversion."""
        gps_lat = xr.DataArray([65.0], dims=["time"], coords={"time": [self.time[0]]}, attrs={"units": "deg"})
        gps_lon = xr.DataArray([-40.0], dims=["time"], coords={"time": [self.time[0]]}, attrs={"units": "deg"})
        gps_time = xr.DataArray([123], dims=["time"], coords={"time": [self.time[0]]}, attrs={"description": "gps time"})

        lat, lon, time = decode_and_convert(gps_lat, gps_lon, gps_time, latitude=65.0, longitude=-40.0)

        self.assertEqual(lat.attrs["units"], "deg")
        self.assertEqual(lon.attrs["units"], "deg")
        self.assertEqual(time.attrs["description"], "gps time")


if __name__ == "__main__":
    unittest.main()

