import unittest
import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.variables.gps import (
    decode_and_convert,
    decode,
    filter
)


def _make_ds(time, gps_lat, gps_lon, gps_alt):
    ds = xr.Dataset(
        data_vars={
            "gps_lat": ("time", gps_lat),
            "gps_lon": ("time", gps_lon),
            "gps_alt": ("time", gps_alt),
        },
        coords={"time": time},
    )
    n = ds.sizes["time"]
    for v in ["gps_lat", "gps_lon", "gps_alt"]:
        ds[f"{v}_qc"] = xr.DataArray(np.full(n, "OK", dtype=object), dims=("time",), coords={"time": ds["time"]})
    return ds


class TestFilter(unittest.TestCase):
    def test_outside_threshold(self):
        """Test that values far from the baseline are all masked."""
        time = pd.date_range("2025-01-01", periods=3, freq="D")
        ds = _make_ds(
            time,
            gps_lat=[1, 2, 3],
            gps_lon=[4, 5, 6],
            gps_alt=(xr.DataArray([1, 1, 10], coords=[("time", time)]) * 200).values,
        )

        ds_out = filter(ds)

        assert ds_out["gps_alt_qc"].to_series().tolist() == ["OK", "OK", "GPS_FILTER"]
        assert ds_out["gps_lat_qc"].to_series().tolist() == ["OK", "OK", "GPS_FILTER"]
        assert ds_out["gps_lon_qc"].to_series().tolist() == ["OK", "OK", "GPS_FILTER"]

    def test_all_within_threshold(self):
        """Test that values close to baseline are preserved."""
        time = pd.date_range("2025-01-01", periods=3, freq="D")
        ds = _make_ds(
            time,
            gps_lat=[1, 2, 3],
            gps_lon=[4, 5, 6],
            gps_alt=[1000, 1005, 995],
        )

        ds_out = filter(ds)

        assert (ds_out["gps_alt_qc"].to_series() == "OK").all()
        assert (ds_out["gps_lat_qc"].to_series() == "OK").all()
        assert (ds_out["gps_lon_qc"].to_series() == "OK").all()

    def test_single_value(self):
        """Test that a single value array is handled correctly."""
        time = pd.date_range("2025-01-01", periods=1, freq="D")
        ds = _make_ds(
            time,
            gps_lat=[10],
            gps_lon=[20],
            gps_alt=[1000],
        )

        ds_out = filter(ds)

        assert ds_out["gps_alt_qc"].item() == "OK"
        assert ds_out["gps_lat_qc"].item() == "OK"
        assert ds_out["gps_lon_qc"].item() == "OK"

    def test_multiple_months_resample(self):
        time = pd.date_range("2025-01-01", periods=60, freq="D")
        gps_alt = np.array([1000] * 30 + [1200] * 30, dtype=float)

        ds = _make_ds(
            time,
            gps_lat=np.arange(60, dtype=float),
            gps_lon=np.arange(60, 120, dtype=float),
            gps_alt=gps_alt,
        )

        ds_out = filter(ds)

        jan = ds_out.sel(time=slice("2025-01-01", "2025-01-30"))
        feb = ds_out.sel(time=slice("2025-02-01", "2025-03-01"))

        assert (jan["gps_alt_qc"].to_series() == "OK").all()
        assert (jan["gps_lat_qc"].to_series() == "OK").all()
        assert (jan["gps_lon_qc"].to_series() == "OK").all()
        # in this case both month should be kept because they are both stable
        assert (feb["gps_alt_qc"].to_series() == "OK").all()
        assert (feb["gps_lat_qc"].to_series() == "OK").all()
        assert (feb["gps_lon_qc"].to_series() == "OK").all()


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
