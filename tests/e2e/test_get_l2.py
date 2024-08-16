import datetime
import json
import tempfile
import unittest
from importlib import metadata
from pathlib import Path

import pandas as pd
import xarray as xr

from pypromice.process.get_l2 import get_l2

TEST_ROOT = Path(__file__).parent.parent
TEST_DATA_ROOT_PATH = TEST_ROOT / "data"


class GetL2TestCase(unittest.TestCase):
    def test_get_l2_tx(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            output_path = Path(tmpdirname) / "output"
            config_file = TEST_DATA_ROOT_PATH / "test_config1_tx.toml"

            aws = get_l2(
                config_file=config_file.as_posix(),
                inpath=TEST_DATA_ROOT_PATH.as_posix(),
                outpath=output_path,
                variables=None,
                metadata=None,
            )

            station_id = "TEST1"
            expected_dir = output_path / station_id
            expected_dataset_paths = {
                "nc_hour": expected_dir / f"{station_id}_hour.nc",
                "csv_hour": expected_dir / f"{station_id}_hour.csv",
            }
            self.assertSetEqual({expected_dir}, set(output_path.iterdir()))
            self.assertSetEqual(
                set(expected_dataset_paths.values()), set(expected_dir.iterdir())
            )

    def test_get_l2_raw(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            output_path = Path(tmpdirname) / "output"
            config_file = TEST_DATA_ROOT_PATH / "test_config1_raw.toml"

            aws = get_l2(
                config_file=config_file.as_posix(),
                inpath=TEST_DATA_ROOT_PATH.as_posix(),
                outpath=output_path,
                variables=None,
                metadata=None,
            )

            station_id = "TEST1"
            expected_dir = output_path / station_id
            expected_dataset_paths = {
                "nc_hour": expected_dir / f"{station_id}_hour.nc",
                "csv_hour": expected_dir / f"{station_id}_hour.csv",
                "nc_10min": expected_dir / f"{station_id}_10min.nc",
                "csv_10min": expected_dir / f"{station_id}_10min.csv",
            }
            self.assertSetEqual({expected_dir}, set(output_path.iterdir()))
            self.assertSetEqual(
                set(expected_dataset_paths.values()), set(expected_dir.iterdir())
            )
            # Test output file format
            dataset_hour = xr.open_dataset(expected_dataset_paths["nc_hour"])
            dataset_10min = xr.open_dataset(expected_dataset_paths["nc_10min"])

            self.assertEqual(
                dataset_10min.attrs["id"],
                f"dk.geus.promice.station.{station_id}.L2.10min",
            )
            self.assertEqual(
                dataset_hour.attrs["id"],
                f"dk.geus.promice.station.{station_id}.L2.hourly",
            )
            self.assertEqual(
                dataset_10min.attrs["title"],
                f"AWS measurements from {station_id} processed to level 2. 10min average.",
            )
            self.assertEqual(
                dataset_hour.attrs["title"],
                f"AWS measurements from {station_id} processed to level 2. Hourly average.",
            )

            t0 = datetime.datetime.utcnow()
            for dataset in [dataset_hour, dataset_10min]:
                self.assertEqual(dataset.attrs["format"], "raw")
                self.assertEqual(dataset.attrs["station_id"], station_id)
                self.assertIsInstance(dataset.attrs["date_created"], str)
                date_created = pd.to_datetime(dataset.attrs["date_created"])
                self.assertLess(t0 - date_created, datetime.timedelta(seconds=1))
                self.assertEqual(
                    dataset.attrs["date_issued"], dataset.attrs["date_created"]
                )
                self.assertEqual(
                    dataset.attrs["date_modified"], dataset.attrs["date_created"]
                )
                self.assertEqual(
                    dataset.attrs["processing_level"],
                    "Level 2",
                )
                self.assertEqual(
                    dataset.attrs["institution"],
                    "Geological Survey of Denmark and Greenland (GEUS)",
                )
                source_decoded = json.loads(dataset.attrs["source"])
                self.assertSetEqual(
                    {"pypromice", "l0_config_file", "l0_data_root"},
                    set(source_decoded.keys()),
                )
                self.assertEqual(
                    source_decoded["pypromice"],
                    metadata.version("pypromice"),
                )
                config_file_name, config_hash = source_decoded["l0_config_file"].rsplit(
                    ":", 1
                )
                self.assertEqual(
                    config_file_name,
                    config_file.name,
                )
                data_root_name, data_root_hash = source_decoded["l0_data_root"].rsplit(":", 1)
                self.assertEqual(
                    data_root_name,
                    TEST_DATA_ROOT_PATH.name,
                )
                self.assertNotEquals(config_hash, 'unknown', 'This test will fail while the commit is dirty')
                self.assertNotEquals(data_root_hash, 'unknown', 'This test will fail while the commit is dirty')
