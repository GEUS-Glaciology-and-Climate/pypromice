# updates needed:
# 1) remove unused imports: patch, calculate_directional_wind_speed, filter_sr (old sig), calculate_correction_factor (unless you still test it)
# 2) update filter_sr usage: it now takes (ds, ZenithAngle_rad, ZenithAngle_deg, AngleDif_deg) and returns (ds_out, flags)
# 3) ds must contain variables named "dsr", "usr", "cc" (not dsr/usr arrays passed separately)
# 4) update correct_sr usage: it now takes DataArrays (dsr_filtered, usr_filtered, cc, phi_sensor_rad, theta_sensor_rad, lat, Declination_rad, HourAngle_rad, ZenithAngle_rad, ZenithAngle_deg, AngleDif_deg)
# 5) remove calculate_TOA import from your old list if duplicated; keep calculate_TOA from radiation module
# 6) fix the bug in your current test_calculate_albedo_basic: (albedo[~mask].isnull()).all() not (~mask)

import unittest
import numpy as np
import xarray as xr

from pypromice.core.variables.radiation import (
    convert_sr,
    convert_lr,
    T_0,
    filter_sr,
    correct_sr,
    calculate_albedo,
    calculate_TOA,
    calculate_correction_factor
)

class TestRadiationConversions(unittest.TestCase):
    def setUp(self):
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


class TestShortwaveRadiation(unittest.TestCase):
    def setUp(self):
        # Representative test data for high Arctic latitude
        self.lat = 75.0  # Greenland/Arctic latitude
        self.dsr = xr.DataArray([500.0, -10.0, 2000.0, 400.0, 600.0], dims="time")
        self.usr = xr.DataArray([50.0, -5.0, 1000.0, 100.0, 500.0], dims="time")
        self.cc = xr.DataArray([0.0, 0.5, 0.5, 0.8, 0.3], dims="time")
        self.ds = xr.Dataset({"dsr": self.dsr, "usr": self.usr, "cc": self.cc})
        self.ZenithAngle_deg = xr.DataArray([15.0, 110.0, 45.0, 60.0, 50.0], dims="time")
        self.ZenithAngle_rad = np.deg2rad(self.ZenithAngle_deg)
        self.AngleDif_deg = xr.DataArray([10.0, 95.0, 100.0, 45.0, np.nan], dims="time")
        self.phi_sensor_rad = xr.DataArray([0.0, 0.1, 0.2, 0.3, 0.4], dims="time")
        self.theta_sensor_rad = xr.DataArray([0.0, 0.1, 0.2, 0.3, 0.0], dims="time")
        self.Declination_rad = xr.DataArray([0.0, 0.1, -0.1, 0.0, 0.05], dims="time")
        self.HourAngle_rad = xr.DataArray([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 0.0], dims="time")

    def test_filter_sr_basic(self):
        ds_f, flags = filter_sr(
            self.ds.copy(),
            self.ZenithAngle_rad,
            self.ZenithAngle_deg,
            self.AngleDif_deg,
        )

        self.assertIsInstance(ds_f, xr.Dataset)
        self.assertIsInstance(flags, tuple)
        self.assertEqual(len(flags), 4)

        bad_flag = flags[0]
        self.assertTrue((ds_f["dsr"].where(bad_flag).dropna("time") == 0).all())
        self.assertTrue((ds_f["usr"].where(bad_flag).dropna("time") == 0).all())

        self.assertGreaterEqual(np.nanmin(ds_f["dsr"].values), 0)
        self.assertGreaterEqual(np.nanmin(ds_f["usr"].values), 0)

    def test_TOA_limit_filtering(self):
        ds_f, _ = filter_sr(
            self.ds.copy(),
            self.ZenithAngle_rad,
            self.ZenithAngle_deg,
            self.AngleDif_deg,
        )

        dsr_c, usr_c, _ = correct_sr(
            ds_f["dsr"],
            ds_f["usr"],
            ds_f["cc"],
            self.phi_sensor_rad,
            self.theta_sensor_rad,
            self.lat,
            self.Declination_rad,
            self.HourAngle_rad,
            self.ZenithAngle_rad,
            self.ZenithAngle_deg,
            self.AngleDif_deg,
        )

        isr_toa = calculate_TOA(self.ZenithAngle_deg, self.ZenithAngle_rad)
        TOA_limit_dsr = 1.2 * isr_toa + 150
        TOA_limit_usr = 0.8 * (1.2 * isr_toa + 150)

        for i in range(self.dsr.size):
            if dsr_c[i].notnull() and (dsr_c[i].item() > TOA_limit_dsr[i].item()):
                self.assertTrue(np.isnan(dsr_c[i].item()))
            if usr_c[i].notnull() and (usr_c[i].item() > TOA_limit_usr[i].item()):
                self.assertTrue(np.isnan(usr_c[i].item()))

    def test_sun_on_lower_dome_filter(self):
        ds_f, _ = filter_sr(
            self.ds.copy(),
            self.ZenithAngle_rad,
            self.ZenithAngle_deg,
            self.AngleDif_deg,
        )

        assert ds_f["usr_qc"].to_series().tolist() == ["OK", "OK", "SUN_ON_LOWER_DOME", "OK", "OK"]
        assert ds_f["dsr_qc"].to_series().tolist() == ["OK", "OK", "SUN_ON_LOWER_DOME", "OK", "OK"]

    def test_correct_sr_basic(self):
        ds_f, _ = filter_sr(
            self.ds.copy(),
            self.ZenithAngle_rad,
            self.ZenithAngle_deg,
            self.AngleDif_deg,
        )

        dsr_c, usr_c, toa_flags = correct_sr(
            ds_f["dsr"],
            ds_f["usr"],
            ds_f["cc"],
            self.phi_sensor_rad,
            self.theta_sensor_rad,
            self.lat,
            self.Declination_rad,
            self.HourAngle_rad,
            self.ZenithAngle_rad,
            self.ZenithAngle_deg,
            self.AngleDif_deg,
        )

        self.assertIsInstance(dsr_c, xr.DataArray)
        self.assertIsInstance(usr_c, xr.DataArray)
        self.assertIsInstance(toa_flags, xr.DataArray)
        self.assertTrue((dsr_c.sel(time=dsr_c.notnull()) >= 0).all())

    def test_calculate_albedo_basic(self):
        ds_f, _ = filter_sr(
            self.ds.copy(),
            self.ZenithAngle_rad,
            self.ZenithAngle_deg,
            self.AngleDif_deg,
        )

        dsr_c, usr_c, _ = correct_sr(
            ds_f["dsr"],
            ds_f["usr"],
            ds_f["cc"],
            self.phi_sensor_rad,
            self.theta_sensor_rad,
            self.lat,
            self.Declination_rad,
            self.HourAngle_rad,
            self.ZenithAngle_rad,
            self.ZenithAngle_deg,
            self.AngleDif_deg,
        )

        albedo, mask = calculate_albedo(
            ds_f["dsr"],
            ds_f["usr"],
            dsr_c,
            ds_f["cc"],
            self.ZenithAngle_deg,
            self.AngleDif_deg,
        )

        self.assertIsInstance(albedo, xr.DataArray)
        self.assertIsInstance(mask, xr.DataArray)

        self.assertTrue(albedo.where(~mask).isnull().all())
        self.assertTrue(((albedo.sel(time=mask) > 0) & (albedo.sel(time=mask)< 1)).all())

    def test_correction_factor_extremes(self):
        # Zenith = 0° (overhead sun)
        zen_deg_overhead = xr.DataArray([0.0], dims="time")
        zen_rad_overhead = np.deg2rad(zen_deg_overhead)

        theta_sensor = xr.DataArray([0.0], dims="time")
        phi_sensor = xr.DataArray([0.0], dims="time")
        decl = xr.DataArray([0.0], dims="time")
        ha = xr.DataArray([0.0], dims="time")
        lat = self.lat
        DifFrac = xr.DataArray([0.5], dims="time")

        CorFac = calculate_correction_factor(
            phi_sensor, theta_sensor, decl, ha,
            zen_rad_overhead, zen_deg_overhead, lat, DifFrac
        )
        assert np.all(np.isfinite(CorFac.values))
        assert (CorFac.values > 0).all()

        # Zenith > 90° (sun below horizon)
        zen_deg_below = xr.DataArray([120.0], dims="time")
        zen_rad_below = np.deg2rad(zen_deg_below)

        CorFac2 = calculate_correction_factor(
            phi_sensor, theta_sensor, decl, ha,
            zen_rad_below, zen_deg_below, lat, DifFrac
        )
        assert CorFac2.item() == 1.0

        # Missing tilt info
        theta_sensor_nan = xr.DataArray([np.nan], dims="time")

        CorFac3 = calculate_correction_factor(
            phi_sensor, theta_sensor_nan, decl, ha,
            zen_rad_overhead, zen_deg_overhead, lat, DifFrac
        )
        assert np.isnan(CorFac3.item())

        # Zenith = 90° (sun at horizon)
        zen_deg_horizon = xr.DataArray([90.0], dims="time")
        zen_rad_horizon = np.deg2rad(zen_deg_horizon)

        CorFac4 = calculate_correction_factor(
            phi_sensor, theta_sensor, decl, ha,
            zen_rad_horizon, zen_deg_horizon, lat, DifFrac
        )
        assert np.all(np.isfinite(CorFac4.values))
        assert (CorFac4.values > 0).all()


if __name__ == "__main__":
    unittest.main()
