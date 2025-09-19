import unittest
import numpy as np
import xarray as xr

from pypromice.core.variables.station_pose import (
    calculate_spherical_tilt,
    calculate_declination,
    calculate_hour_angle,
    calculate_sun_direction_degrees,
    calculate_zenith,
    calculate_angle_difference,
)


class TestSolarCalculations(unittest.TestCase):

    def test_calculate_spherical_tilt_basic(self):
        tilt_x = xr.DataArray([0, 10, -10])
        tilt_y = xr.DataArray([0, 5, -5])
        phi, theta = calculate_spherical_tilt(tilt_x, tilt_y)

        self.assertIsInstance(phi, xr.DataArray)
        self.assertIsInstance(theta, xr.DataArray)
        self.assertEqual(phi.shape, tilt_x.shape)
        self.assertEqual(theta.shape, tilt_y.shape)
        # horizontal case: tilt_x=0, tilt_y=0 should yield theta=0
        self.assertAlmostEqual(theta[0].item(), 0.0, places=8)

    def test_invalid_inputs_shape_mismatch(self):
        tilt_x = xr.DataArray([0, 1])
        tilt_y = xr.DataArray([0, 1, 2])  # mismatch
        with self.assertRaises(ValueError):
            calculate_spherical_tilt(tilt_x, tilt_y)

    def test_calculate_declination_range(self):
        doy = xr.DataArray([1, 80, 172, 355])
        hour = xr.DataArray([12, 0, 6, 18])
        minute = xr.DataArray([0, 30, 45, 15])
        decl = calculate_declination(doy, hour, minute)

        self.assertIsInstance(decl, xr.DataArray)

        # Physically, declination is within ~±23.44°, but the approximation may overshoot.
        # allow ±25° to be robust to the empirical formula.
        lower_bound = np.deg2rad(-25)
        upper_bound = np.deg2rad(25)

        # All outputs are within the relaxed bounds
        self.assertGreaterEqual(decl.min().item(), lower_bound)
        self.assertLessEqual(decl.max().item(), upper_bound)

        # Extra sanity: no NaNs and all finite
        self.assertTrue(np.all(np.isfinite(decl.values)))

    def test_declination_regression_known_dates(self):
        # DOYs chosen near equinoxes/solstices (approximate)
        doy = xr.DataArray([80, 172, 265, 355])  # Mar equinox, Jun solstice, Sep equinox, Dec solstice
        hour = xr.DataArray([12, 12, 12, 12])  # Noon
        minute = xr.DataArray([0, 0, 0, 0])

        decl = calculate_declination(doy, hour, minute)
        self.assertIsInstance(decl, xr.DataArray)
        self.assertTrue(np.all(np.isfinite(decl.values)))  # Sanity check

        # Expected declination values in degrees (approximate physical values)
        # Mar/Sept equinox ~ 0°, Jun solstice ~ +23.44°, Dec solstice ~ -23.44°
        expected_deg = np.array([0.0, 23.44, 0.0, -23.44])
        expected_rad = np.deg2rad(expected_deg)

        # Tolerance: 1 degree (in radians). Empirical formula can overshoot slightly.
        tol_rad = np.deg2rad(1.0)

        # Determine differences between calculated and expected declination values
        decl_vals = decl.values
        diffs = np.abs(decl_vals - expected_rad)

        # Assert all absolute differences are within tolerance
        self.assertTrue(np.all(diffs <= tol_rad),
                        msg=f"Declination diffs (rad): {diffs}, tol (rad): {tol_rad}")

    def test_calculate_hour_angle_midday(self):
        hour = xr.DataArray([12])
        minute = xr.DataArray([0])
        lon = 0.0
        ha = calculate_hour_angle(hour, minute, lon)
        self.assertIsInstance(ha, xr.DataArray)

        # At noon and lon=0, hour angle should be ~0
        self.assertAlmostEqual(ha.item(), 0.0, places=8)

    def test_calculate_sun_direction_degrees(self):
        ha_rad = xr.DataArray([0, np.pi/2, np.pi, 3*np.pi/2])
        direction = calculate_sun_direction_degrees(ha_rad)
        self.assertIsInstance(direction, xr.DataArray)
        self.assertTrue(all((0 <= d <= 360) for d in direction))

        # At hour angle 0 -> direction should be 180
        self.assertAlmostEqual(direction[0].item(), 180.0, places=8)

    def test_calculate_zenith_extremes(self):
        lat = 0.0  # equator
        decl = xr.DataArray([0.0])
        ha = xr.DataArray([0.0])
        zen_rad, zen_deg = calculate_zenith(lat, decl, ha)

        self.assertIsInstance(zen_rad, xr.DataArray)
        self.assertIsInstance(zen_deg, xr.DataArray)

        # At equator, decl=0, noon -> zenith=0 rad overhead
        self.assertAlmostEqual(zen_rad.item(), 0.0, places=8)
        self.assertAlmostEqual(zen_deg.item(), 0.0, places=8)

    def test_zenith_regression_known_cases(self):
        # Case 1: Equator, equinox, noon -> Sun directly overhead
        lat = 0.0
        decl = xr.DataArray([0.0])  # declination = 0 rad
        ha = xr.DataArray([0.0])  # hour angle = 0 rad
        zen_rad, zen_deg = calculate_zenith(lat, decl, ha)

        self.assertAlmostEqual(zen_rad.item(), 0.0, places=6)
        self.assertAlmostEqual(zen_deg.item(), 0.0, places=6)

        # Case 2: 45°N latitude, equinox, noon -> zenith = 45°
        lat = 45.0
        decl = xr.DataArray([0.0])
        ha = xr.DataArray([0.0])
        zen_rad, zen_deg = calculate_zenith(lat, decl, ha)

        expected_rad = np.deg2rad(45.0)
        expected_deg = 45.0
        self.assertAlmostEqual(zen_rad.item(), expected_rad, places=6)
        self.assertAlmostEqual(zen_deg.item(), expected_deg, places=6)

        # Case 3: North Pole, equinox, noon -> Sun at horizon -> zenith = 90°
        lat = 90.0
        decl = xr.DataArray([0.0])
        ha = xr.DataArray([0.0])
        zen_rad, zen_deg = calculate_zenith(lat, decl, ha)

        expected_rad = np.deg2rad(90.0)
        expected_deg = 90.0
        self.assertAlmostEqual(zen_rad.item(), expected_rad, places=6)
        self.assertAlmostEqual(zen_deg.item(), expected_deg, places=6)

        # 45°N, equinox, midnight (hour angle = 180°) -> Sun below horizon
        lat = 45.0
        decl = xr.DataArray([0.0])
        ha = xr.DataArray([np.pi])  # hour angle = 180° = midnight

        zen_rad, zen_deg = calculate_zenith(lat, decl, ha)

        # Zenith should be 180° - 45° = 135° from overhead
        expected_rad = np.deg2rad(135.0)
        expected_deg = 135.0

        self.assertAlmostEqual(zen_rad.item(), expected_rad, places=6)
        self.assertAlmostEqual(zen_deg.item(), expected_deg, places=6)

    def test_calculate_angle_difference_alignment(self):
        # Setup: sun directly overhead, sensor horizontal -> difference ~0
        zenith = xr.DataArray([0.0])
        hour_angle = xr.DataArray([0.0])
        phi_sensor = xr.DataArray([0.0])
        theta_sensor = xr.DataArray([0.0])

        angle = calculate_angle_difference(zenith, hour_angle, phi_sensor, theta_sensor)
        self.assertIsInstance(angle, xr.DataArray)
        self.assertAlmostEqual(angle.item(), 0.0, places=8)

if __name__ == "__main__":
    unittest.main()
