import unittest
import xarray as xr
import numpy as np

from pypromice.core.variables.pressure_transducer_depth import (
    correct_and_calculate_depth,
    apply_offset)


class TestPressureTransducerFunctions(unittest.TestCase):
    def setUp(self):
        # Common test data
        self.z_pt = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims="time")
        self.air_pressure = xr.DataArray(np.array([1013.0, 1015.0, 1010.0]), dims="time")
        self.pt_z_factor = 1.0
        self.pt_z_coef = 1.0
        self.pt_z_p_coef = 1013.0
        self.rtol = 1e-6
        self.atol = 1e-8

    def test_correct_and_calculate_depth_supported_antifreeze(self):
        for antifreeze, rho in [(50, 1092), (100, 1145)]:
            with self.subTest(antifreeze=antifreeze):
                z_pt_cor, z_pt = correct_and_calculate_depth(
                    self.z_pt,
                    self.air_pressure,
                    pt_antifreeze=antifreeze,
                    pt_z_factor=self.pt_z_factor,
                    pt_z_coef=self.pt_z_coef,
                    pt_z_p_coef=self.pt_z_p_coef,
                )

                # Validate outputs
                self.assertIsInstance(z_pt_cor, xr.DataArray)
                self.assertIsInstance(z_pt, xr.DataArray)
                self.assertTrue(np.all(np.isfinite(z_pt_cor)))
                self.assertTrue(np.all(np.isfinite(z_pt)))

                # Expected z_pt
                expected_z_pt = self.z_pt * self.pt_z_coef * self.pt_z_factor * 998.0 / rho

                # Expected z_pt_cor (includes air pressure term)
                expected_z_pt_cor = (
                    expected_z_pt
                    + 100 * (self.pt_z_p_coef - self.air_pressure) / (rho * 9.81)
                )

                # Compare with tolerance
                xr.testing.assert_allclose(z_pt, expected_z_pt, rtol=self.rtol, atol=self.atol)
                xr.testing.assert_allclose(z_pt_cor, expected_z_pt_cor, rtol=self.rtol, atol=self.atol)

    def test_correct_and_calculate_depth_invalid_antifreeze_logs(self):
        with self.assertLogs("pypromice.core.variables.pressure_transducer_depth", level="INFO") as cm:
            z_pt_cor, z_pt = correct_and_calculate_depth(
                self.z_pt,
                self.air_pressure,
                pt_antifreeze=75,  # unsupported
                pt_z_factor=self.pt_z_factor,
                pt_z_coef=self.pt_z_coef,
                pt_z_p_coef=self.pt_z_p_coef,
            )

        # Ensure logging
        self.assertTrue(any("Incorrect metadata" in message for message in cm.output))

        # Outputs should be NaN
        self.assertTrue(np.all(np.isnan(z_pt_cor)))
        self.assertTrue(np.all(np.isnan(z_pt)))

    def test_apply_offset_various(self):
        for offset in [2, -1, 0]:
            with self.subTest(offset=offset):
                result = apply_offset(self.z_pt, offset)
                expected = self.z_pt + offset
                xr.testing.assert_allclose(result, expected, rtol=self.rtol, atol=self.atol)

    def test_nan_propagation(self):
        # z_pt contains NaN
        z_pt_nan = xr.DataArray(np.array([1.0, np.nan, 3.0]), dims="time")
        z_pt_cor, z_pt = correct_and_calculate_depth(
            z_pt_nan,
            self.air_pressure,
            pt_antifreeze=50,
            pt_z_factor=self.pt_z_factor,
            pt_z_coef=self.pt_z_coef,
            pt_z_p_coef=self.pt_z_p_coef,
        )
        self.assertTrue(np.isnan(z_pt.values[1]))
        self.assertTrue(np.isnan(z_pt_cor.values[1]))

        # air_pressure contains NaN
        air_pressure_nan = xr.DataArray(np.array([1013.0, np.nan, 1010.0]), dims="time")
        z_pt_cor, z_pt = correct_and_calculate_depth(
            self.z_pt,
            air_pressure_nan,
            pt_antifreeze=50,
            pt_z_factor=self.pt_z_factor,
            pt_z_coef=self.pt_z_coef,
            pt_z_p_coef=self.pt_z_p_coef,
        )
        self.assertTrue(np.isnan(z_pt_cor.values[1]))

        # z_pt itself should still be finite (air pressure not in formula)
        self.assertTrue(np.all(np.isfinite(z_pt.values)))


if __name__ == "__main__":
    unittest.main()
