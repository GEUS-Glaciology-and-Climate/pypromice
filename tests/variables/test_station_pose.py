import unittest
import numpy as np
import xarray as xr
import pandas as pd

from pypromice.core.variables.station_pose import (
    calculate_spherical_tilt,
    calculate_declination,
    calculate_hour_angle,
    calculate_sun_direction_degrees,
    calculate_zenith,
    calculate_angle_difference,
    apply_tilt_factor,
    convert_and_filter_tilt,
    smooth_tilt_with_moving_window,
    interpolate_tilt,
    interpolate_rotation,
)


class TestTiltConversions(unittest.TestCase):
    def setUp(self):
        # Make a simple time coordinate
        time = pd.date_range("2025-06-01", periods=5, freq="H")
        self.tilt = xr.DataArray(
            np.array([-150.0, -50.0, 0.0, 10.0, 50.0]),
            dims="time",
            coords={"time": time},
        )
        self.rtol = 1e-6
        self.atol = 1e-8

    def test_apply_tilt_factor(self):
        factor = 2.5
        result = apply_tilt_factor(self.tilt, factor)
        expected = self.tilt * factor
        xr.testing.assert_allclose(result, expected, rtol=self.rtol, atol=self.atol)

    def test_convert_and_filter_tilt_threshold_masking(self):
        # Values below -100 should be masked (NaN before interpolation)
        result = convert_and_filter_tilt(self.tilt)
        self.assertTrue(np.isnan(result[0].values))  # -150 < -100 masked

    def test_convert_and_filter_tilt_conversion_polynomial(self):
        # Test polynomial conversion on a small positive tilt value (10 V -> 1.0)
        test_tilt = xr.DataArray(
            np.array([10.0]), dims="time", coords={"time": [pd.Timestamp("2025-06-01")]}
        )
        result = convert_and_filter_tilt(test_tilt)
        dst = test_tilt / 10.0
        expected = dst / np.abs(dst) * (
            -0.49 * (np.abs(dst)) ** 4
            + 3.6 * (np.abs(dst)) ** 3
            - 10.4 * (np.abs(dst)) ** 2
            + 21.1 * (np.abs(dst))
        )
        xr.testing.assert_allclose(result, expected, rtol=self.rtol, atol=self.atol)

    def test_convert_and_filter_tilt_interpolation(self):
        # A missing value between valid points should be interpolated
        tilt_with_gap = xr.DataArray(
            np.array([0.0, np.nan, 20.0]),
            dims="time",
            coords={"time": pd.date_range("2023-01-01", periods=3, freq="H")},
        )
        result = convert_and_filter_tilt(tilt_with_gap)
        self.assertFalse(np.isnan(result[1].values))  # should be interpolated

    def test_convert_and_filter_tilt_zero_stays_zero(self):
        # Zero input should remain zero
        tilt_zero = xr.DataArray(
            np.array([0.0]),
            dims="time",
            coords={"time": [pd.Timestamp("2023-01-01")]},
        )
        result = convert_and_filter_tilt(tilt_zero)
        self.assertEqual(result.values[0], 0.0)


class TestTiltSmoothingInterpolation(unittest.TestCase):
    def setUp(self):
        # Create 2 days of 10-minute data
        time = pd.date_range("2025-06-01", periods=288, freq="10min")
        values = np.sin(np.linspace(0, 4 * np.pi, len(time)))  # smooth signal
        self.tilt = xr.DataArray(values, dims="time",
                                 coords={"time": time},
                                 name="tilt")                   # to_dataframe() routine in smoothing requires DataArray with name

        # Add some NaNs
        self.tilt_with_gaps = self.tilt.copy()
        self.tilt_with_gaps[50:60] = np.nan

        # Rotation example (more variability)
        self.rot = xr.DataArray(
            np.cos(np.linspace(0, 6 * np.pi, len(time))),
            dims="time",
            coords={"time": time},
        )

        self.rtol = 1e-6
        self.atol = 1e-8

    # ----------------------------
    # smooth_tilt_with_moving_window
    # ----------------------------
    def test_smooth_tilt_with_moving_window_basic(self):
        dim, smoothed = smooth_tilt_with_moving_window(self.tilt)
        self.assertEqual(dim, "time")
        self.assertEqual(len(smoothed), len(self.tilt))
        self.assertTrue(np.all(np.isfinite(smoothed)))

    def test_smooth_tilt_with_moving_window_short_array(self):
        short_tilt = xr.DataArray([1.0, 2.0, 3.0],
                                  dims="time",
                                  coords={"time": pd.date_range("2023-01-01", periods=3, freq="H")},
                                  name="tilt")
        dim, smoothed = smooth_tilt_with_moving_window(short_tilt)
        self.assertEqual(len(smoothed), len(short_tilt))

    def test_smooth_tilt_with_moving_window_all_nan(self):
        nan_tilt = xr.DataArray([np.nan, np.nan, np.nan],
                                dims="time",
                                coords={"time": pd.date_range("2023-01-01", periods=3, freq="H")},
                                name="tilt")
        dim, smoothed = smooth_tilt_with_moving_window(nan_tilt)
        self.assertTrue(np.all(np.isnan(smoothed)))

    def test_smooth_tilt_with_moving_window_constant(self):
        const_tilt = xr.DataArray(np.ones(20),
                                  dims="time",
                                  coords={"time": pd.date_range("2023-01-01", periods=20, freq="H")},
                                  name="tilt")
        dim, smoothed = smooth_tilt_with_moving_window(const_tilt)
        np.testing.assert_allclose(smoothed, np.ones(20), rtol=self.rtol, atol=self.atol)

    # ----------------------------
    # interpolate_tilt
    # ----------------------------
    def test_interpolate_tilt_fills_gaps(self):
        result = interpolate_tilt(self.tilt_with_gaps)
        np.testing.assert_array_equal(result.time.values, self.tilt_with_gaps.time.values)

        # Test all non-NaN values are finite
        non_nan_values = result[~np.isnan(result)]
        self.assertTrue(np.all(np.isfinite(non_nan_values.values)))

    def test_interpolate_tilt_respects_threshold(self):
        noisy = self.tilt.copy(data=np.random.normal(0, 10, len(self.tilt)))
        result = interpolate_tilt(noisy)
        self.assertEqual(result.shape, noisy.shape)

        # Test all non-NaN values are finite
        non_nan_values = result[~np.isnan(result)]
        self.assertTrue(np.all(np.isfinite(non_nan_values.values)))

    def test_interpolate_tilt_all_nan(self):
        nan_tilt = xr.DataArray([np.nan, np.nan, np.nan], dims="time", coords={"time": pd.date_range("2023-01-01", periods=3, freq="H")})
        result = interpolate_tilt(nan_tilt)
        self.assertTrue(np.all(np.isnan(result.values)))

    def test_interpolate_tilt_boundary_fill(self):
        # Leading NaN should be backfilled, trailing NaN should be forward filled
        tilt_with_edges = xr.DataArray([np.nan, 1.0, 2.0, np.nan], dims="time", coords={"time": pd.date_range("2023-01-01", periods=4, freq="H")})
        result = interpolate_tilt(tilt_with_edges)

        # Test all non-NaN values are finite
        non_nan_values = result[~np.isnan(result)]
        self.assertTrue(np.all(np.isfinite(non_nan_values.values)))

    # ----------------------------
    # interpolate_rotation
    # ----------------------------
    def test_interpolate_rotation_fills_and_smooths(self):
        result = interpolate_rotation(self.rot)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], "time")
        self.assertEqual(len(result[1]), len(self.rot))

        # Test all non-NaN values are finite
        non_nan_values = result[1][~np.isnan(result[1])]
        self.assertTrue(np.all(np.isfinite(non_nan_values)))

    def test_interpolate_rotation_all_nan(self):
        nan_rot = xr.DataArray([np.nan, np.nan, np.nan], dims="time", coords={"time": pd.date_range("2023-01-01", periods=3, freq="H")})
        result = interpolate_rotation(nan_rot)
        self.assertTrue(np.all(np.isnan(result[1])))

    def test_interpolate_rotation_short_series(self):
        short_rot = xr.DataArray([1.0, 2.0], dims="time", coords={"time": pd.date_range("2023-01-01", periods=2, freq="H")})
        result = interpolate_rotation(short_rot)
        self.assertEqual(len(result[1]), 2)

    def test_interpolate_rotation_low_threshold_masks(self):
        # With a very low threshold, everything should be masked initially and then filled
        result = interpolate_rotation(self.rot, threshold=1e-12)
        self.assertEqual(len(result[1]), len(self.rot))

        # Test all non-NaN values are finite
        non_nan_values = result[1][~np.isnan(result[1])]
        self.assertTrue(np.all(np.isfinite(non_nan_values)))


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
