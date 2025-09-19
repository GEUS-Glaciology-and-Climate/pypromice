import unittest
import numpy as np
import xarray as xr
from unittest.mock import patch

from pypromice.core.variables.radiation import (
    convert_sr,
    convert_lr,
    filter_lr,
    T_0,
    filter_sr,
    correct_sr,
    calculate_albedo,
    calculate_correction_factor,
    calculate_TOA
)

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


class TestShortwaveRadiation(unittest.TestCase):

    def setUp(self):
        # Representative test data for high Arctic latitude
        self.lat = 75.0  # Greenland/Arctic latitude
        self.dsr = xr.DataArray([500.0, -10.0, 2000.0, 400.0, 600.0])
        self.usr = xr.DataArray([50.0, -5.0, 1000.0, 100.0, 500.0])
        self.cc = xr.DataArray([0.0, 0.5, 0.5, 0.8, 0.3])
        # Zenith angles: Sun above horizon, below horizon, mid-range
        self.ZenithAngle_deg = xr.DataArray([15.0, 110.0, 45.0, 60.0, 50.0])
        self.ZenithAngle_rad = np.deg2rad(self.ZenithAngle_deg)
        # Angle differences include lower-dome scenario and NaN
        self.AngleDif_deg = xr.DataArray([10.0, 95.0, 100.0, 45.0, np.nan])
        self.phi_sensor_rad = xr.DataArray([0.0, 0.1, 0.2, 0.3, 0.4])
        self.theta_sensor_rad = xr.DataArray([0.0, 0.1, 0.2, 0.3, 0.0])
        self.Declination_rad = xr.DataArray([0.0, 0.1, -0.1, 0.0, 0.05])
        self.HourAngle_rad = xr.DataArray([0.0, np.pi/2, np.pi, 3*np.pi/2, 0.0])

    def test_filter_sr_basic(self):
        dsr_f, usr_f, flags = filter_sr(
            self.dsr, self.usr, self.cc,
            self.ZenithAngle_rad, self.ZenithAngle_deg,
            self.AngleDif_deg
        )

        self.assertIsInstance(dsr_f, xr.DataArray)
        self.assertIsInstance(usr_f, xr.DataArray)
        self.assertIsInstance(flags, tuple)
        self.assertEqual(len(flags), 4)

        # Sun below horizon (>95 deg) → zero
        bad_flag = flags[0]
        self.assertTrue((dsr_f[bad_flag] == 0).all())
        self.assertTrue((usr_f[bad_flag] == 0).all())

        # Negative values clipped to zero
        self.assertGreaterEqual(np.nanmin(dsr_f.values), 0)
        self.assertGreaterEqual(np.nanmin(usr_f.values), 0)

    def test_TOA_limit_filtering(self):
        dsr_f, usr_f, _ = filter_sr(
            self.dsr, self.usr, self.cc,
            self.ZenithAngle_rad, self.ZenithAngle_deg,
            self.AngleDif_deg
        )
        dsr_c, usr_c, toa_flags = correct_sr(
            dsr_f, usr_f, self.cc,
            self.phi_sensor_rad, self.theta_sensor_rad,
            self.lat, self.Declination_rad, self.HourAngle_rad,
            self.ZenithAngle_rad, self.ZenithAngle_deg,
            self.AngleDif_deg
        )

        # Calculate TOA
        isr_toa = calculate_TOA(self.ZenithAngle_deg, self.ZenithAngle_rad)
        TOA_limit_dsr = 1.2 * isr_toa + 150
        TOA_limit_usr = 0.8 * (1.2 * isr_toa + 150)

        # Downwelling above TOA == NaN
        for i in range(len(self.dsr)):
            if dsr_c[i].notnull() and dsr_c[i].item() > TOA_limit_dsr[i]:
                self.assertTrue(np.isnan(dsr_c[i].item()))
            if usr_c[i].notnull() and usr_c[i].item() > TOA_limit_usr[i]:
                self.assertTrue(np.isnan(usr_c[i].item()))

    def test_sun_on_lower_dome_filter(self):
        dsr_f, usr_f, flags = filter_sr(
            self.dsr, self.usr, self.cc,
            self.ZenithAngle_rad, self.ZenithAngle_deg,
            self.AngleDif_deg
        )
        sunonlowerdome_flag = flags[1]

        # Measurements flagged as sun on lower dome == NaN
        self.assertTrue((dsr_f[sunonlowerdome_flag].isnull()).all())
        self.assertTrue((usr_f[sunonlowerdome_flag].isnull()).all())

    def test_correct_sr_basic(self):
        dsr_f, usr_f, _ = filter_sr(
            self.dsr, self.usr, self.cc,
            self.ZenithAngle_rad, self.ZenithAngle_deg,
            self.AngleDif_deg
        )
        dsr_c, usr_c, toa_flags = correct_sr(
            dsr_f, usr_f, self.cc,
            self.phi_sensor_rad, self.theta_sensor_rad,
            self.lat, self.Declination_rad, self.HourAngle_rad,
            self.ZenithAngle_rad, self.ZenithAngle_deg,
            self.AngleDif_deg
        )
        self.assertIsInstance(dsr_c, xr.DataArray)
        self.assertIsInstance(usr_c, xr.DataArray)
        self.assertIsInstance(toa_flags, xr.DataArray)
        self.assertTrue((dsr_c[~dsr_c.isnull()] >= 0).all())

    def test_calculate_albedo_basic(self):
        dsr_f, usr_f, _ = filter_sr(
            self.dsr, self.usr, self.cc,
            self.ZenithAngle_rad, self.ZenithAngle_deg,
            self.AngleDif_deg
        )
        dsr_c, usr_c, _ = correct_sr(
            dsr_f, usr_f, self.cc,
            self.phi_sensor_rad, self.theta_sensor_rad,
            self.lat, self.Declination_rad, self.HourAngle_rad,
            self.ZenithAngle_rad, self.ZenithAngle_deg,
            self.AngleDif_deg
        )
        albedo, mask = calculate_albedo(
            dsr_f, usr_f, dsr_c, self.cc,
            self.ZenithAngle_deg, self.AngleDif_deg
        )
        self.assertIsInstance(albedo, xr.DataArray)
        self.assertIsInstance(mask, xr.DataArray)
        # Albedo NaN where mask is False
        self.assertTrue((albedo[~mask].isnull()).all())
        # Albedo 0–1 where mask is True
        self.assertTrue(((albedo[mask] > 0) & (albedo[mask] < 1)).all())

    def test_correction_factor_extremes(self):
        # Zenith = 0° (overhead, polar day scenario)
        zen_deg_overhead = xr.DataArray([0.0])
        zen_rad_overhead = np.deg2rad(zen_deg_overhead)
        theta_sensor = xr.DataArray([0.0])
        phi_sensor = xr.DataArray([0.0])
        decl = xr.DataArray([0.0])
        ha = xr.DataArray([0.0])
        lat = self.lat
        DifFrac = xr.DataArray([0.5])

        CorFac = calculate_correction_factor(
            phi_sensor, theta_sensor, decl, ha,
            zen_rad_overhead, zen_deg_overhead, lat, DifFrac
        )
        self.assertTrue(np.all(np.isfinite(CorFac.values)))
        self.assertTrue((CorFac.values > 0).all())

        # Zenith > 90° (Sun below horizon, polar night scenario)
        zen_deg_below = xr.DataArray([120.0])
        zen_rad_below = np.deg2rad(zen_deg_below)
        CorFac2 = calculate_correction_factor(
            phi_sensor, theta_sensor, decl, ha,
            zen_rad_below, zen_deg_below, lat, DifFrac
        )
        self.assertEqual(CorFac2.item(), 1.0)

        # NaN inputs (missing sensor tilt)
        theta_sensor_nan = xr.DataArray([np.nan])
        CorFac3 = calculate_correction_factor(
            phi_sensor, theta_sensor_nan, decl, ha,
            zen_rad_overhead, zen_deg_overhead, lat, DifFrac
        )
        self.assertTrue(np.isnan(CorFac3.item()))

        # Zenith = 90° (sun at horizon)
        zen_deg_horizon = xr.DataArray([90.0])
        zen_rad_horizon = np.deg2rad(zen_deg_horizon)
        CorFac4 = calculate_correction_factor(
            phi_sensor, theta_sensor, decl, ha,
            zen_rad_horizon, zen_deg_horizon, lat, DifFrac
        )
        self.assertTrue(np.all(np.isfinite(CorFac4.values)))
        self.assertTrue((CorFac4.values > 0).all())


if __name__ == "__main__":
    unittest.main()