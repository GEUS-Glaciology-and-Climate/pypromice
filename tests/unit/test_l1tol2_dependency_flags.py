import unittest
from pathlib import Path

import numpy as np

from pypromice.core.qc.common import flag_qc
from pypromice.pipeline.aws import AWS
from pypromice.pipeline.L1toL2 import toL2

TEST_DATA_ROOT_PATH = Path(__file__).parent.parent / "data"


class FlaggedParentBeforeDerivationTestCase(unittest.TestCase):
    """A t_rad flag set during L2 (e.g. by persistence) must reach dlr and
    ulr before t_surf and cc are computed from them."""

    def test_flagged_t_rad_leaves_t_surf_and_cc_empty(self):
        aws = AWS(
            (TEST_DATA_ROOT_PATH / "test_config1_raw.toml").as_posix(),
            TEST_DATA_ROOT_PATH.as_posix(),
            data_issues_repository=TEST_DATA_ROOT_PATH / "data_issues",
            magnetic_declination_file=(
                TEST_DATA_ROOT_PATH
                / "magnetic_declination_configurations/test_magdec_config1.toml"
            ).as_posix(),
            var_file=None,
            meta_file=None,
        )
        aws.getL1()
        L1 = aws.L1A

        reference = toL2(L1.copy(), aws.vars, aws.data_issues_repository / "flags",
                         aws.data_issues_repository / "adjustments")
        ok = reference["t_surf"].notnull().values
        self.assertGreater(ok.sum(), 10, "test data needs valid t_surf values")

        # flag t_rad on half of the samples that give a valid t_surf
        mask = np.zeros(L1.sizes["time"], dtype=bool)
        mask[np.flatnonzero(ok)[::2]] = True
        flagged = flag_qc(L1.copy(), "t_rad", "PERSISTENCE",
                          mask=L1["t_rad"].copy(data=mask))
        L2 = toL2(flagged, aws.vars, aws.data_issues_repository / "flags",
                  aws.data_issues_repository / "adjustments")

        for var in ["dlr", "ulr", "t_surf", "cc"]:
            if var in L2:
                self.assertTrue(L2[var].values[mask].size > 0)
                self.assertTrue(np.isnan(L2[var].values[mask]).all(), var)
        # samples whose t_rad is still accepted are untouched (the in-pipeline
        # persistence check may reject a few isolated values left by the mask)
        kept = ~mask & L2["t_rad"].notnull().values
        self.assertGreater(kept.sum(), 0.9 * (~mask & ok).sum())
        np.testing.assert_array_equal(L2["t_surf"].values[kept],
                                      reference["t_surf"].values[kept])


if __name__ == "__main__":
    unittest.main()
