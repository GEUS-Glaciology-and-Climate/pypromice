import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.qc.common import flag_qc
from pypromice.io.write import addVars, getColNames, prepare_and_write
import pypromice.resources


def _make_ds():
    time = pd.date_range("2021-01-01", periods=6, freq="h")
    ds = xr.Dataset(
        {"t_u": ("time", np.arange(6.0)), "not_in_csv": ("time", np.arange(6.0))},
        coords={"time": time},
    )
    ds.attrs.update(level="L2", number_of_booms=1, station_id="TEST")
    ds = flag_qc(ds, "t_u", "PERSISTENCE", mask=True, index_slice={"time": time[:2]})
    ds = flag_qc(ds, "not_in_csv", "PERSISTENCE", mask=True, index_slice={"time": time[:2]})
    return ds


class QcFlagVariablesTestCase(unittest.TestCase):
    def setUp(self):
        self.vars_df = pypromice.resources.load_variables()

    def test_getColNames_only_adds_qc_of_listed_variables(self):
        ds = _make_ds()
        without = getColNames(self.vars_df, ds)
        with_qc = getColNames(self.vars_df, ds, include_qc_flags=True)
        self.assertNotIn("t_u_qc", without)
        self.assertIn("t_u_qc", with_qc)
        self.assertEqual(with_qc.index("t_u_qc"), with_qc.index("t_u") + 1)
        self.assertNotIn("not_in_csv_qc", with_qc)

    def test_addVars_builds_qc_attributes_idempotently(self):
        ds = _make_ds()
        for _ in range(2):
            ds = addVars(ds, self.vars_df)
        attrs = ds["t_u_qc"].attrs
        parent_long_name = self.vars_df.loc["t_u", "long_name"]
        self.assertEqual(attrs["long_name"], f"QC flag associated with {parent_long_name}")
        self.assertEqual(attrs["standard_name"], "status_flag")
        self.assertIn("flag_meanings", attrs)  # kept from the flag variable itself
        self.assertEqual(ds["t_u"].attrs["ancillary_variables"], "t_u_qc")
        # a flag whose parent is not in variables.csv is left untouched
        self.assertNotIn("QC flag associated with", ds["not_in_csv_qc"].attrs["long_name"])

    def test_resampled_output_with_qc_flags_is_refused(self):
        with self.assertRaises(ValueError):
            prepare_and_write(_make_ds(), ".", self.vars_df, {}, time="60min",
                              resample=True, include_qc_flags=True)


if __name__ == "__main__":
    unittest.main()
