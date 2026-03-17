import tempfile
from pathlib import Path
from unittest import TestCase

import pandas as pd
import xarray as xr

from pypromice.ingest import l0
from pypromice.ingest.l0 import load_data_file, load_data_files

DATA_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent / "tests" / "data"


class L0IngestTestCase(TestCase):

    def test_load_config(self):
        filename = "FOO_300534063816570_1.txt"
        config_str = f"""
        station_id = 'FOO'
        logger_type = 'CR1000X'
        number_of_booms = 2
        site_type = 'accumulation'
        nodata     = ['-999', 'NAN']
        modem = [['300534063816570', '2024-06-17 01:00:00']]

        ['{filename}']
        format     = 'TX'
        skiprows = 0
        latitude =  78.02
        longitude = 33.97
        dsr_eng_coef = 11.09
        usr_eng_coef = 12.61
        dlr_eng_coef = 7.83
        ulr_eng_coef = 9.63
        tilt_y_factor = -1
        boom_azimuth = 0
        columns = ['time','rec','p_l','p_u']
        """
        with tempfile.TemporaryDirectory() as dir:
            config_file_path = Path(dir) / "config.toml"
            inpath = Path(dir)
            with config_file_path.open("w") as f:
                f.write(config_str)

            config = l0.load_config(config_file_path.as_posix(), inpath=inpath)

        # The global configuration entries are dynamically added to the file config
        self.assertListEqual(list(config.keys()), [filename])
        file_config = config[filename]
        self.assertEqual(file_config["station_id"], "FOO")
        self.assertEqual(file_config["logger_type"], "CR1000X")
        self.assertEqual(file_config["number_of_booms"], 2)
        self.assertEqual(file_config["site_type"], "accumulation")
        self.assertEqual(file_config["nodata"], ["-999", "NAN"])
        self.assertEqual(
            file_config["modem"], [["300534063816570", "2024-06-17 01:00:00"]]
        )

        # The config parser appends msg_lat and msg_lon to the column list as "default columns"
        expected_columns = ["time", "rec", "p_l", "p_u", "msg_lat", "msg_lon"]
        self.assertListEqual(file_config["columns"], expected_columns)

        # Dynamically generated attributes
        self.assertEqual(file_config["file"], (inpath / filename).as_posix())
        self.assertEqual(file_config["conf"], config_file_path.as_posix())

    def test_load_data_files(self):
        """Test a basic case where the data file contains the msg_lat and msg_lon columns and skiprows is set to 1."""
        with tempfile.NamedTemporaryFile() as f:
            data_str = """
            2024-05-15 17:00:00,1084640400,805.0,-20.03,42,77.53670,-65.88580
            2024-05-15 18:00:00,1084644000,806.0,-19.67,42,77.22598,-61.26020
            2024-05-15 19:00:00,1084647600,806.0,-19.57,42,77.21205,-61.47997
            2024-05-15 20:00:00,1084651200,806.0,-19.55,42,77.20913,-61.11830
            """
            f.write(data_str.encode())
            f.seek(0)
            config = {
                "a_file_name_key": {
                    "format": "raw",
                    "nodata": ["-999", "NAN"],
                    "skiprows": 1,
                    "columns": [
                        "time",
                        "rec",
                        "p_l",
                        "t_l",
                        "msg_lat",
                        "msg_lon",
                    ],
                    "file": f.name,
                }
            }

            datasets = load_data_files(config)
            self.assertEqual(len(datasets), 1)
            dataset = datasets[0]
            self.assertDictEqual(
                dataset.attrs,
                {
                    "detected_file_type": "csv_default",
                    "filename": Path(f.name).name,
                    "format": "raw",
                    "level": "L0",
                },
            )
            self.assertListEqual(
                list(dataset.data_vars),
                ["rec", "p_l", "t_l", "msg_lat", "msg_lon"],
            )

    def test_load_data_files_without_mgs_coords(self):
        """
        Read data_files handles a special case where the data file does not contain the msg_lat and msg_lon columns.
        The config reader automatically adds the columns to the config file.
        """
        with tempfile.NamedTemporaryFile() as f:

            data_str = """
            2024-05-15 17:00:00,1084640400,805.0,-20.03
            2024-05-15 18:00:00,1084644000,806.0,-19.67
            2024-05-15 19:00:00,1084647600,806.0,-19.57
            2024-05-15 20:00:00,1084651200,806.0,-19.55
            """
            f.write(data_str.encode())
            f.seek(0)
            config = {
                "a_file_name_key": {
                    "format": "raw",
                    "nodata": ["-999", "NAN"],
                    "skiprows": 0,
                    "columns": [
                        "time",
                        "rec",
                        "p_l",
                        "t_l",
                        "msg_lat",
                        "msg_lon",
                    ],
                    "file": f.name,
                }
            }

            datasets = load_data_files(config)
            self.assertEqual(len(datasets), 1)
            dataset = datasets[0]
            self.assertListEqual(
                list(dataset.data_vars),
                ["rec", "p_l", "t_l"],
                "The data file does not contain the msg_lat and msg_lon columns.",
            )

    def test_load_tx_data_files(self):
        config_file_path = DATA_ROOT / "test_config1_tx.toml"
        config = l0.load_config(config_file_path, inpath=DATA_ROOT)

        datasets = l0.load_data_files(config)
        self.assertEqual(len(datasets), 1)
        dataset: xr.Dataset = datasets[0]

        self.assertEqual(dataset.attrs["detected_file_type"], "csv_default")
        self.assertEqual(
            dataset.attrs["filename"],
            "test_raw_transmitted1.txt",
        )
        self.assertEqual(
            dataset.attrs["station_id"],
            "TEST1",
        )
        self.assertEqual(
            dataset.attrs["logger_type"],
            "CR1000X",
        )

        self.assertIn("time", dataset.coords)
        self.assertEqual(dataset.coords["time"].dtype, "datetime64[ns]")
        # The file has newlines in the header and footer, so the number of rows is 15412
        self.assertEqual(len(dataset.coords["time"]), 15410)

    def test_load_raw_data_files(self):
        config_file_path = DATA_ROOT / "test_config1_raw.toml"
        config = l0.load_config(config_file_path, inpath=DATA_ROOT)

        datasets = l0.load_data_files(config)
        self.assertEqual(len(datasets), 2)
        dataset: xr.Dataset = datasets[0]

        self.assertIn("time", dataset.coords)
        self.assertEqual(dataset.coords["time"].dtype, "datetime64[ns]")

    def test_load_data_files_time_offset(self):
        config_file_path = DATA_ROOT / "test_config1_raw.toml"
        config = l0.load_config(config_file_path, inpath=DATA_ROOT)
        config["test_raw1.txt"]["time_offset"] = 2
        dataset = l0.load_data_files(config)[0]

        expected_timestamps = pd.to_datetime(
            [
                "2016-08-01 02:00:00",
                "2016-08-01 02:10:00",
                "2016-08-01 02:20:00",
                "2016-08-01 02:30:00",
                "2016-08-01 02:40:00",
            ]
        )

        self.assertListEqual(
            list(dataset.coords["time"].head(5)),
            list(expected_timestamps),
        )
