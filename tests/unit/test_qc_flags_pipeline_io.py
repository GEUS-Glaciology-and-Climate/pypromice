"""QC flag variables through the file-based pipeline steps (join_l2, L3, write)."""
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.qc.common import flag_qc, has_qc_flags
from pypromice.io.write import prepare_and_write
from pypromice.pipeline.join_l2 import _combine_qc_flags
from pypromice.pipeline.L2toL3 import _restore_qc_flags
import pypromice.resources


def _ds(time, values, flag_first=None):
    ds = xr.Dataset({"t_u": ("time", np.asarray(values, float))}, coords={"time": time})
    if flag_first is not None:
        ds = flag_qc(ds, "t_u", "PERSISTENCE", mask=True, index_slice={"time": time[:flag_first]})
    return ds


class CombineQcFlagsTestCase(unittest.TestCase):
    def test_flags_follow_the_values_chosen_by_combine_first(self):
        t1 = pd.date_range("2021-01-01", periods=4, freq="h")
        t2 = pd.date_range("2021-01-01 02:00", periods=4, freq="h")
        # file1 (preferred): flags on its first two samples (raw values kept)
        ds1 = _ds(t1, [1, 2, 3, 4], flag_first=2)
        # file2 (fills gaps): no flag variable at all (== never flagged)
        ds2 = _ds(t2, [30, 40, 50, 60])
        all_ds = ds1.combine_first(ds2)
        out = _combine_qc_flags(ds1, ds2, all_ds)
        self.assertEqual(out["t_u_qc"].dtype, np.int8)
        # times: 00h 01h (file1 flagged), 02h 03h (file1 ok), 04h 05h (file2 only)
        np.testing.assert_array_equal(out["t_u_qc"].values, [1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(out["t_u"].values, [1, 2, 3, 4, 50, 60])

    def test_flag_kept_when_the_flagged_value_was_already_removed(self):
        t = pd.date_range("2021-01-01", periods=3, freq="h")
        ds1 = _ds(t, [np.nan, 2, 3])
        ds1 = flag_qc(ds1, "t_u", "MANUAL", mask=True)  # nothing left to flag (NaN)
        ds1["t_u_qc"].values[0] = 5  # flag of an already-removed sample
        ds2 = _ds(t, [np.nan, np.nan, np.nan])
        out = _combine_qc_flags(ds1, ds2, ds1.combine_first(ds2))
        self.assertEqual(int(out["t_u_qc"].values[0]), 5)


class RestoreQcFlagsTestCase(unittest.TestCase):
    def test_only_unchanged_variables_get_their_flags_back(self):
        t = pd.date_range("2021-01-01", periods=4, freq="h")
        l2 = xr.Dataset({"t_u": ("time", [1.0, 2, 3, 4]), "p_u": ("time", [9.0, 9, 9, 9])},
                        coords={"time": t})
        l2 = flag_qc(l2, "t_u", "PERSISTENCE", mask=True, index_slice={"time": t[:1]})
        l2 = flag_qc(l2, "p_u", "PERSISTENCE", mask=True, index_slice={"time": t[:1]})
        l3 = l2.drop_vars(["t_u_qc", "p_u_qc"]).copy(deep=True)
        l3["t_u"] = l3["t_u"].where(~np.isin(np.arange(4), [0]))      # cleaned, unchanged by L3
        passthrough = {"t_u": l3["t_u"].values.copy(), "p_u": l3["p_u"].values.copy()}
        l3["p_u"] = l3["p_u"] + 1                                       # modified by L3

        out = _restore_qc_flags(l3.copy(deep=True), l2, passthrough, keep_flagged_data=True)
        self.assertIn("t_u_qc", out)
        self.assertNotIn("p_u_qc", out)
        self.assertEqual(out["t_u"].values[0], 1.0)  # raw value of the flagged sample restored

        out = _restore_qc_flags(l3.copy(deep=True), l2, passthrough, keep_flagged_data=False)
        self.assertIn("t_u_qc", out)
        self.assertTrue(np.isnan(out["t_u"].values[0]))  # flags only, data stays removed


class WriteNeverLeaksFlaggedDataTestCase(unittest.TestCase):
    def test_flagged_values_are_removed_unless_flags_are_written(self):
        import tempfile
        t = pd.date_range("2021-01-01", periods=6, freq="h")
        ds = _ds(t, np.arange(6.0), flag_first=2)
        ds.attrs.update(level="L2", number_of_booms=1, station_id="TEST",
                        latitude=70.0, longitude=-50.0, altitude=1000.0)
        vars_df = pypromice.resources.load_variables()
        with tempfile.TemporaryDirectory() as tmp:
            prepare_and_write(ds.copy(deep=True), tmp, vars_df, {}, time="mixed", resample=False)
            with xr.open_dataset(f"{tmp}/TEST/TEST_mixed.nc") as out:
                out.load()
            self.assertFalse(has_qc_flags(out))
            self.assertEqual(int(out["t_u"].isnull().sum()), 2)

            prepare_and_write(ds.copy(deep=True), tmp, vars_df, {}, time="mixed",
                              resample=False, include_qc_flags=True)
            with xr.open_dataset(f"{tmp}/TEST/TEST_mixed.nc") as out:
                out.load()
            self.assertTrue(has_qc_flags(out))
            self.assertEqual(int(out["t_u"].isnull().sum()), 0)  # raw kept, labeled by t_u_qc


if __name__ == "__main__":
    unittest.main()
