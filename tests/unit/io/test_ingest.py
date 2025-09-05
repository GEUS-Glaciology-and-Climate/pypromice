import tempfile
from pathlib import Path
from unittest import TestCase

import pandas as pd
import xarray as xr

from pypromice.io.ingest import l0
from pypromice.io.ingest.l0 import read_data_file

DATA_ROOT = Path(__file__).parent.parent.parent / "data"


class L0IngestTestCase(TestCase):

    def test_load_data_files(self):
        config_file_path = DATA_ROOT / "test_config1_raw.toml"
        config = l0.load_config(config_file_path, inpath=DATA_ROOT)

        datasets = l0.load_data_files(config)
        self.assertEqual(len(datasets), 2)
        dataset: xr.Dataset = datasets[0]

        # Testing global attributes
        expected_attributes = dict(
            level="L0",
            detected_file_type="toa5",
            filename="test_raw1.txt",
            station_id="TEST1",
            logger_type="CR1000X",
            number_of_booms=1,
            site_type="ablation",
            format="raw",
            latitude=79.91,
            longitude=24.09,
            hygroclip_t_offset=40,
            dsr_eng_coef=14.01,
            usr_eng_coef=12.72,
            dlr_eng_coef=11.08,
            ulr_eng_coef=11.42,
            pt_z_coef=0.39571,
            pt_z_p_coef=1022.5,
            pt_z_factor=2.5,
            pt_antifreeze=50,
            boom_azimuth=0,
        )
        self.assertLessEqual(
            set(expected_attributes.keys()),
            set(dataset.attrs.keys()),
        )
        self.assertDictEqual(
            # {**dataset.attrs, **expected_attributes},
            expected_attributes,
            dataset.attrs,
        )

        # Testing data variables
        # The SKIP columns are removed during ingestion
        expected_data_vars = [
            "rec",
            #'SKIP_3',
            "p_u",
            "t_u",
            #'SKIP_6',
            "rh_u",
            "wspd_u",
            "wdir_u",
            "wdir_std_u",
            "dsr",
            "usr",
            "dlr",
            "ulr",
            "t_rad",
            "z_boom_u",
            "z_boom_q_u",
            "z_stake",
            "z_stake_q",
            "z_pt",
            "t_i_1",
            "t_i_2",
            "t_i_3",
            "t_i_4",
            "t_i_5",
            "t_i_6",
            "t_i_7",
            "t_i_8",
            "tilt_x",
            "tilt_y",
            "gps_time",
            "gps_lat",
            "gps_lon",
            "gps_alt",
            "gps_geoid",
            #'SKIP_36',
            "gps_q",
            "gps_numsat",
            "gps_hdop",
            "t_log",
            "fan_dc_u",
            "batt_v_ini",
            "batt_v",
        ]
        self.assertListEqual(list(dataset.data_vars.keys()), expected_data_vars)

        # Testing time coordinates
        expected_timestamps = pd.to_datetime(
            [
                "2016-08-01 00:00:00",
                "2016-08-01 00:10:00",
                "2016-08-01 00:20:00",
                "2016-08-01 00:30:00",
                "2016-08-01 00:40:00",
            ]
        )
        self.assertListEqual(
            list(dataset.coords["time"].head(5)),
            list(expected_timestamps),
        )

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

    def test_csv_v1(self):
        with tempfile.NamedTemporaryFile() as f:

            config = dict(
                station_id="NUK_L",
                logger_type="CR1000",
                number_of_booms=1,
                site_type="ablation",
                nodata=["-999", "NAN"],
                format="raw",
                skiprows=0,
                latitude=64.48,
                longitude=49.53,
                hygroclip_t_offset=0,
                dsr_eng_coef=8.51,
                usr_eng_coef=8.51,
                dlr_eng_coef=8.51,
                ulr_eng_coef=8.51,
                pt_z_coef=0.4702,
                pt_z_p_coef=1023.5,
                pt_z_factor=2.5,
                pt_antifreeze=50,
                boom_azimuth=0,
                file_version=1,
                columns=[
                    "SKIP_1",
                    "year",
                    "doy",
                    "hhmm",
                    "p_u",
                    "t_u",
                    "SKIP_7",
                    "rh_u",
                    "wspd_u",
                    "wdir_u",
                    "dsr",
                    "usr",
                    "dlr",
                    "ulr",
                    "t_rad",
                    "z_boom_u",
                    "z_stake",
                    "z_pt",
                    "t_i_1",
                    "t_i_2",
                    "t_i_3",
                    "t_i_4",
                    "t_i_5",
                    "t_i_6",
                    "t_i_7",
                    "t_i_8",
                    "SKIP_27",
                    "tilt_x",
                    "tilt_y",
                    "gps_lat",
                    "gps_lon",
                    "gps_alt",
                    "gps_hdop",
                    "t_log",
                    "batt_v",
                ],
            )
        with open(f.name, "w") as fp:
            fp.write(
                """
                101,2010,207,130,0,0,939.7955,6.36351,5.34553,77.98921,0,0,0,1.658184,2.321458,-44.84436,-16.47955,5.101608,2.385,220,0.709,189,1.068,-0.0838623,-0.05529785,-0.04577637,-0.1029053,-0.06478882,0.04940796,-0.1029053,-0.1219177,3.992,3.289,"","","","","","","","","",4.384979,0,12.78455,12.77899
                101,2010,207,140,0,298178,939.8408,6.174957,5.352204,75.3913,2.577,62.05,0.02,0.6632738,0.9949107,-32.05946,-15.28068,4.704056,2.636,203,0.708,187,1.068,-0.1702881,-0.1322021,-0.1417236,-0.1417236,-0.1607666,-0.05603027,-0.1512451,-0.1322021,-0.024,-0.31,"GT013910.20","NH6428.91374","WH04931.90166","551.6","32.5","M","1","09","0.89",4.470833,0.2,12.78444,12.71152
                101,2010,207,150,0,298180,940.1732,4.227051,4.197676,81.39323,2.747,69.88,0,-0.6632738,-0.9949107,-18.27676,-12.88362,3.951935,2.645,218,0.709,186,1.068,-0.1895447,-0.1419373,-0.1514587,-0.1514587,-0.1705322,-0.07531738,-0.1610107,-0.1324158,5.018,4.111,"GT013910.20","NH6428.91374","WH04931.90166","551.6","32.5","M","1","09","0.89",4.299129,128.2,12.74443,12.69999
                101,2010,207,200,0,298190,940.5785,4.60434,4.271085,79.61841,2.78,60.79,0,0.6632738,0.3316369,-12.18447,-12.88358,3.993774,2.645,209,0.707,186,14.6,-0.1896667,-0.1420593,-0.1515808,-0.1420593,-0.1705933,-0.07537842,-0.1515808,-0.1325073,2.275,1.409,"GT020315.60","NH6428.91402","WH04931.89954","548.1","32.5","M","1","09","0.89",4.150021,127.6,12.73882,12.64423
                101,2010,207,210,0,298200,940.3727,3.95517,3.730526,81.39198,3.108,60.31,0.014,-0.3316369,-0.6632738,-11.38548,-12.68382,3.659042,2.648,205,0.707,194,14.64,-0.1896667,-0.1420593,-0.1516113,-0.1516113,-0.1706543,-0.07543945,-0.1611328,-0.1325378,1.242,0.959,"GT020315.60","NH6428.91402","WH04931.89954","548.1","32.5","M","1","09","0.89",4.064167,120.4,12.72196,12.67751
                """
            )
        config["file"] = f.name
        dataset = read_data_file(config)

        self.assertEquals(
            dataset.attrs["detected_file_type"],
            "csv_v1",
        )
        expected_timestamps = pd.to_datetime(
            [
                "2010-07-26T01:30:00",
                "2010-07-26T01:40:00",
                "2010-07-26T01:50:00",
                "2010-07-26T02:00:00",
                "2010-07-26T02:10:00",
            ]
        )

        self.assertListEqual(
            list(dataset.coords["time"].head(5)),
            list(expected_timestamps),
        )
