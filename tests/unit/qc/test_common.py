import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.qc.common import (clean_view, finalize_qc, flag_qc,
                                      has_qc_flags, is_ok)


def _flagged_ds():
    time = pd.date_range("2021-01-01", periods=6, freq="h")
    ds = xr.Dataset({"t_u": ("time", np.arange(6.0))}, coords={"time": time})
    ds = flag_qc(ds, "t_u", "PERSISTENCE", mask=True, index_slice={"time": time[:2]})
    # first flag wins: the later, broader flag must not overwrite it
    ds = flag_qc(ds, "t_u", "OUT_OF_LIMITS", mask=True, index_slice={"time": time[:4]})
    return ds


class FlagQcTestCase(unittest.TestCase):
    def test_flag_never_touches_data_and_first_flag_wins(self):
        ds = _flagged_ds()
        np.testing.assert_array_equal(ds["t_u"].values, np.arange(6.0))
        self.assertEqual(ds["t_u_qc"].dtype, np.int8)
        # codes: 1 PERSISTENCE (first two), 3 OUT_OF_LIMITS (next two), 0 OK
        np.testing.assert_array_equal(ds["t_u_qc"].values, [1, 1, 3, 3, 0, 0])
        self.assertTrue(has_qc_flags(ds))
        self.assertEqual(int(is_ok(ds, "t_u").sum()), 2)

    def test_clean_view_does_not_modify_the_dataset(self):
        ds = _flagged_ds()
        clean = clean_view(ds)
        self.assertEqual(int(clean["t_u"].isnull().sum()), 4)
        self.assertEqual(int(ds["t_u"].isnull().sum()), 0)


class FinalizeQcTestCase(unittest.TestCase):
    def test_default_removes_flagged_data_and_flag_variables(self):
        out = finalize_qc(_flagged_ds())
        self.assertFalse(has_qc_flags(out))
        self.assertEqual(int(out["t_u"].isnull().sum()), 4)

    def test_keep_qc_flags_removes_data_but_keeps_flags(self):
        out = finalize_qc(_flagged_ds(), keep_qc_flags=True)
        self.assertTrue(has_qc_flags(out))
        self.assertEqual(int(out["t_u"].isnull().sum()), 4)
        np.testing.assert_array_equal(out["t_u_qc"].values, [1, 1, 3, 3, 0, 0])

    def test_keep_flagged_data_keeps_everything(self):
        out = finalize_qc(_flagged_ds(), keep_flagged_data=True)
        self.assertTrue(has_qc_flags(out))
        np.testing.assert_array_equal(out["t_u"].values, np.arange(6.0))

    def test_keep_flagged_data_wins_over_keep_qc_flags(self):
        out = finalize_qc(_flagged_ds(), keep_flagged_data=True, keep_qc_flags=False)
        self.assertTrue(has_qc_flags(out))
        self.assertEqual(int(out["t_u"].notnull().sum()), 6)


if __name__ == "__main__":
    unittest.main()
