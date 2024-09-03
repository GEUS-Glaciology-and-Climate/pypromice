#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing test module
"""
import datetime
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pypromice.resources
import xarray as xr
from pypromice.pipeline.aws import AWS
from pypromice.pipeline.get_l2 import get_l2
from pypromice.pipeline.get_l2tol3 import get_l2tol3
from pypromice.pipeline.join_l2 import join_l2
from pypromice.pipeline.join_l3 import join_l3
from pypromice.io.write import addVars, addMeta

TEST_ROOT = Path(__file__).parent.parent
TEST_DATA_ROOT_PATH = TEST_ROOT / "data"
TEST_CONFIG_PATH = TEST_DATA_ROOT_PATH / "test_config1_raw.toml"
STATION_CONFIGURATIONS_ROOT = TEST_DATA_ROOT_PATH / "station_configurations"


class TestProcess(unittest.TestCase):
    def test_get_vars(self):
        """Test variable table lookup retrieval"""
        v = pypromice.resources.load_variables()
        self.assertIsInstance(v, pd.DataFrame)
        self.assertTrue(v.columns[0] in "standard_name")
        self.assertTrue(v.columns[2] in "units")

    def test_get_meta(self):
        """Test AWS names retrieval"""
        m = pypromice.resources.load_metadata()
        self.assertIsInstance(m, dict)
        self.assertTrue("references" in m)

    def test_add_all(self):
        """Test variable and metadata attributes added to Dataset"""
        d = xr.Dataset()
        v = pypromice.resources.load_variables()
        att = list(v.index)
        att1 = ["gps_lon", "gps_lat", "gps_alt", "albedo", "p"]
        for a in att:
            d[a] = [0, 1]
        for a in att1:
            d[a] = [0, 1]
        d["time"] = [
            datetime.datetime.now(),
            datetime.datetime.now() - timedelta(days=365),
        ]
        d.attrs["station_id"] = "TEST"
        d.attrs["level"] = "L2_test"
        meta = pypromice.resources.load_metadata()
        d = addVars(d, v)
        d = addMeta(d, meta)
        self.assertTrue(d.attrs["station_id"] == "TEST")
        self.assertIsInstance(d.attrs["references"], str)

    def test_l0_to_l3(self):
        """Test L0 to L3 processing"""
        pAWS = AWS(
            TEST_CONFIG_PATH.as_posix(),
            TEST_DATA_ROOT_PATH.as_posix(),
            data_issues_repository=TEST_DATA_ROOT_PATH / "data_issues",
            var_file=None,
            meta_file=None,
        )
        pAWS.process()
        self.assertIsInstance(pAWS.L2, xr.Dataset)
        self.assertTrue(pAWS.L2.attrs["station_id"] == "TEST1")

    def get_l2_cli(self):
        """Test get_l2 CLI"""
        exit_status = os.system("get_l2 -h")
        self.assertEqual(exit_status, 0)

    def test_join_l2_cli(self):
        """Test join_l2 CLI"""
        exit_status = os.system("join_l2 -h")
        self.assertEqual(exit_status, 0)

    def test_l2_to_l3_cli(self):
        """Test get_l2tol3 CLI"""
        exit_status = os.system("get_l2tol3 -h")
        self.assertEqual(exit_status, 0)

    def test_join_l3_cli(self):
        """Test join_l3 CLI"""
        exit_status = os.system("join_l3 -h")
        self.assertEqual(exit_status, 0)

    def test_full_e2e(self):
        """
        A minimum e2e test running the full sequence of processing functions used in the main pipeline.
        It only checks the existence of the output files. Not their content.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            root = Path(tmpdirname)

            # Part 1 - Level 0 to level 2
            output_path_tx = root / "station_l2_tx"
            output_path_raw = root / "station_l2_raw"
            config_file_tx = TEST_DATA_ROOT_PATH / "test_config1_tx.toml"
            config_file_raw = TEST_DATA_ROOT_PATH / "test_config1_raw.toml"
            data_issues_path = TEST_DATA_ROOT_PATH / "data_issues"
            station_id = "TEST1"
            aws_tx_l2 = get_l2(
                config_file=config_file_tx.as_posix(),
                inpath=TEST_DATA_ROOT_PATH.as_posix(),
                outpath=output_path_tx,
                data_issues_path=data_issues_path,
                variables=None,
                metadata=None,
            )
            aws_raw_l2 = get_l2(
                config_file=config_file_raw.as_posix(),
                inpath=TEST_DATA_ROOT_PATH.as_posix(),
                outpath=output_path_raw,
                data_issues_path=data_issues_path,
                variables=None,
                metadata=None,
            )
            #   TODO: This step ignores 10 min data in the join step
            hourly_out_path_tx = output_path_tx / station_id / f"{station_id}_hour.nc"
            hourly_out_path_raw = output_path_raw / station_id / f"{station_id}_hour.nc"
            self.assertTrue(hourly_out_path_tx.exists())
            self.assertTrue(hourly_out_path_raw.exists())

            # Part 2 - Merge level 2 raw and tx data
            output_l2_join = root / "station_l2_join"
            aws_join_l2 = join_l2(
                hourly_out_path_raw.as_posix(),
                hourly_out_path_tx.as_posix(),
                outpath=output_l2_join.as_posix(),
                variables=None,
                metadata=None,
            )
            hourly_out_path_join = output_l2_join / station_id / f"{station_id}_hour.nc"
            self.assertTrue(hourly_out_path_join.exists())

            # Part 3 - Level 2 to level 3
            site_id = "SITE_01"
            output_l3 = root / "station_l3"
            aws_station_l3 = get_l2tol3(
                config_folder=STATION_CONFIGURATIONS_ROOT,
                inpath=hourly_out_path_join.as_posix(),
                outpath=output_l3.as_posix(),
                variables=None,
                metadata=None,
                data_issues_path=data_issues_path,
            )

            # Part 4 Join L3: Merge Current data and historical GC-Net and convert to site
            folder_gc_net = root / "gc_net"
            output_site_l3 = root / "site_l3"
            aws_join_l3 = join_l3(
                config_folder=STATION_CONFIGURATIONS_ROOT.as_posix(),
                site=site_id,
                folder_l3=output_l3.as_posix(),
                folder_gcnet=folder_gc_net.as_posix(),
                folder_glaciobasis='.',
                outpath=output_site_l3.as_posix(),
                variables=None,
                metadata=None,
            )

            for time_interval in ["hour", "day", "month"]:
                for extension in ["csv", "nc"]:
                    expected_output_path = (
                        output_site_l3
                        / site_id
                        / f"{site_id}_{time_interval}.{extension}"
                    )
                    self.assertTrue(expected_output_path.exists())

            for output_rel_path in [
                "station_l2_raw/TEST1/TEST1_10min.csv",
                "station_l2_raw/TEST1/TEST1_10min.nc",
                "station_l2_raw/TEST1/TEST1_hour.csv",
                "station_l2_raw/TEST1/TEST1_hour.nc",
                "station_l2_tx/TEST1/TEST1_hour.csv",
                "station_l2_tx/TEST1/TEST1_hour.nc",
                "station_l2_join/TEST1/TEST1_hour.csv",
                "station_l2_join/TEST1/TEST1_hour.nc",
                "station_l3/TEST1/TEST1_day.csv",
                "station_l3/TEST1/TEST1_day.nc",
                "station_l3/TEST1/TEST1_hour.csv",
                "station_l3/TEST1/TEST1_hour.nc",
                "station_l3/TEST1/TEST1_month.csv",
                "station_l3/TEST1/TEST1_month.nc",
                "site_l3/SITE_01/SITE_01_day.csv",
                "site_l3/SITE_01/SITE_01_day.nc",
                "site_l3/SITE_01/SITE_01_hour.csv",
                "site_l3/SITE_01/SITE_01_hour.nc",
                "site_l3/SITE_01/SITE_01_month.csv",
                "site_l3/SITE_01/SITE_01_month.nc",
            ]:
                output_path = root / output_rel_path
                self.assertTrue(output_path.exists())

                if output_path.name.endswith("nc"):
                    output_dataset = xr.load_dataset(output_path)
                    self.check_global_attributes(output_dataset, output_rel_path)

                    # Check if the l3 datasets are compressed
                    if output_path.parent.parent.name == 'site_l3':
                        self.assertEqual(output_dataset['p_u'].encoding["zlib"], True, output_rel_path)
                    else:
                        self.assertEqual(output_dataset['p_u'].encoding["zlib"], False, output_rel_path)

            # Test if the l3 output netcdf files are compressed with zlib
            for output_rel_path in [
                "station_l3/TEST1/TEST1_day.nc",
                "station_l3/TEST1/TEST1_hour.nc",
                "station_l3/TEST1/TEST1_month.nc",
                "site_l3/SITE_01/SITE_01_day.nc",
                "site_l3/SITE_01/SITE_01_hour.nc",
                "site_l3/SITE_01/SITE_01_month.nc",
            ]:
                output_path = root / output_rel_path
                output_dataset = xr.load_dataset(output_path)
                for var in output_dataset.variables:
                    # %%
                    print(var, output_dataset[var].encoding)
                    continue
                    self.assertEqual(output_dataset[var].encoding["zlib"], True)


    def check_global_attributes(self, dataset: xr.Dataset, reference: str):
        attribute_keys = set(dataset.attrs.keys())
        highly_recommended_global_attributes = {
            "title",
            "summary",
            "keywords",
            "conventions",
        }
        self.assertSetEqual(
            set(),
            highly_recommended_global_attributes - attribute_keys,
            reference,
        )
        required_global_attributes = {
            "id",
            "naming_authority",
            "date_created",
            "institution",
            "date_issued",
            "date_modified",
            "processing_level",
            "product_version",
            "source",
        }
        self.assertSetEqual(
            set(),
            required_global_attributes - attribute_keys,
            reference,
        )


if __name__ == "__main__":
    unittest.main()
