"""
Automatic test to verify get_bufr generates exactly the same output files given the same parameters.
"""

import datetime
import hashlib
import logging
import pickle
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Dict
from unittest import TestCase

import numpy as np
import pandas as pd

from pypromice.postprocess import get_bufr
from pypromice.postprocess.get_bufr import (
    DEFAULT_STATION_CONFIGURATION_PATH,
    StationConfiguration,
    write_station_configuration_mapping,
)

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
    level=logging.INFO,
)

DATA_DIR = Path(__file__).parent.absolute()


def run_get_bufr(
    l3_data: pd.DataFrame,
    stid: str,
    latest_timestamps: Optional[Dict[str, datetime.datetime]],
    station_configuration_mapping=None,
    **get_bufr_kwargs,
) -> Dict[str, str]:
    """
    Run get_bufr using a temporary folder structure for input and output data
    The output bufr files can be verified using the sha256 hashes.

    Parameters
    ----------
    l3_data : Data frame with level 3 data
    stid : Station id for input data
    latest_timestamps

    Returns
    -------
    mapping from station id to sha256 hashes

    """
    with TemporaryDirectory() as output_path:
        output_path = Path(output_path)
        bufr_out = output_path.joinpath("BUFR_out")
        timestamps_pickle_filepath = output_path.joinpath("latest_timestamps.pickle")
        positions_filepath = output_path.joinpath("AWS_latest_locations.csv")
        station_configuration_path = output_path.joinpath("station_configuration.toml")
        l3_filepath = output_path.joinpath(f"{stid}_hour.csv")
        l3_data.to_csv(l3_filepath)

        if station_configuration_mapping is None:
            shutil.copy(
                DEFAULT_STATION_CONFIGURATION_PATH,
                station_configuration_path,
            )
        else:
            with station_configuration_path.open("w") as fp:
                write_station_configuration_mapping(
                    station_configuration_mapping,
                    fp,
                )

        if latest_timestamps is not None:
            with timestamps_pickle_filepath.open("wb") as fp:
                pickle.dump(latest_timestamps, fp)

        get_bufr.get_bufr(
            bufr_out=bufr_out,
            input_files=[l3_filepath],
            timestamps_pickle_filepath=timestamps_pickle_filepath,
            positions_filepath=positions_filepath,
            station_configuration_path=station_configuration_path,
            **get_bufr_kwargs,
        )

        output_bufr_files = bufr_out.glob("*.bufr")
        file_hashes = dict()
        for p in output_bufr_files:
            with p.open("rb") as fp:
                file_hashes[p.stem] = hashlib.sha256(fp.read()).hexdigest()

        return file_hashes


class PreRefactoringBufrTestCase(TestCase):
    @staticmethod
    def get_station_configuration_mapping(
        stid: str,
        wmo_id: str,
        station_site: Optional[str] = None,
        station_type: str = "mobile",
        barometer_from_gps: float = 0,
        anemometer_from_sonic_ranger: float = 0.4,
        temperature_from_sonic_ranger: float = -0.1,
        height_of_gps_from_station_ground: float = 0,
        skipped_variables=(),
        comment=None,
        export_bufr=True,
        positions_update_timestamp_only=False,
    ) -> dict:
        return {
            stid: StationConfiguration(
                stid=stid,
                station_site=stid if station_type is None else station_site,
                station_type=station_type,
                wmo_id=wmo_id,
                barometer_from_gps=barometer_from_gps,
                anemometer_from_sonic_ranger=anemometer_from_sonic_ranger,
                temperature_from_sonic_ranger=temperature_from_sonic_ranger,
                height_of_gps_from_station_ground=height_of_gps_from_station_ground,
                skipped_variables=skipped_variables,
                comment=comment,
                export_bufr=export_bufr,
                positions_update_timestamp_only=positions_update_timestamp_only,
            ),
        }

    def test_get_bufr_has_new_data(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"DY2": datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 8)
        expected_file_hashes = {
            stid: "2b94d2ef611cfddb6dd537ca63d0ec4fb5d8e880943f81a6d5e724c042ac8971"
        }
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )

    def test_get_bufr_has_new_data_dont_store_position(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"DY2": datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 8)
        expected_file_hashes = {
            stid: "2b94d2ef611cfddb6dd537ca63d0ec4fb5d8e880943f81a6d5e724c042ac8971"
        }
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=False,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )

    def test_get_bufr_stid_to_skip(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"DY2": datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 6)
        expected_file_hashes = {}
        mapping = self.get_station_configuration_mapping(
            stid, wmo_id="04464", export_bufr=False
        )
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )

    def test_get_bufr_has_no_data_newer_than_latests_timestamps(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {stid: datetime.datetime(2023, 12, 7, 23, 00)}
        now_timestamp = datetime.datetime(2023, 12, 8)
        expected_file_hashes = {}

        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )

    def test_get_bufr_includes_datasets_not_in_latests_timestamps(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        latest_timestamps = {}
        now_timestamp = datetime.datetime(2023, 12, 8)
        expected_file_hashes = {
            stid: "2b94d2ef611cfddb6dd537ca63d0ec4fb5d8e880943f81a6d5e724c042ac8971"
        }

        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )

    def test_get_bufr_has_old_data_compared_to_now(self):
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        latest_timestamps = {stid: datetime.datetime(2023, 12, 6)}
        now_timestamp = datetime.datetime(2023, 12, 20)
        expected_file_hashes = {}

        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )

    def test_invalid_value_at_last_index(self):
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        # Set some of instantanous values to nan
        l3_src.loc[140:, "p_i"] = np.nan
        latest_timestamps = {stid: datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 8)
        expected_file_hashes = {
            stid: "bb951e0245ce3f6fe656b9bb5c85f097753a6969cc60b2cf8b34e0764495e627"
        }

        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )

    def test_multiple_last_valid_indices_all_instantaneous_timestamps_are_none(self):
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        # Set all instantanous values to nan
        l3_src.loc[
            :,
            [
                "t_i",
                "p_i",
                "rh_i",
                "wspd_i",
                "wdir_i",
            ],
        ] = np.nan
        latest_timestamps = {stid: datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 6)
        expected_file_hashes = {}

        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )

    def test_multiple_last_valid_indices_all_older_than_2days(self):
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        # Set some of instantanous values to nan
        l3_src.loc[140:, "p_i"] = np.nan
        latest_timestamps = {stid: datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 10)
        expected_file_hashes = {}

        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )

    def test_min_data_wx_failed(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        l3_src.loc[:, ["p_i", "t_i"]] = np.nan
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"DY2": datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 6)
        expected_file_hashes = {}
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )

    def test_min_data_pos_failed(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        l3_src.loc[:, ["gps_lat"]] = np.nan
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"DY2": datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 6)
        expected_file_hashes = {}
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )

    def test_ignore_newer_data_than_now_input(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {stid: datetime.datetime(2023, 12, 1)}
        # New is before the latest data
        now_timestamp = datetime.datetime(
            2023,
            12,
            6,
        )
        expected_file_hashes = {
            stid: "976a24edef2d0e6e2f29fa13d6242419fa05b24905db715fe351c19a1aa1d577"
        }
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )

    def test_land_station_export(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "WEG_B"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"WEG_B": datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 8)
        expected_file_hashes = {
            stid: "eb42044f38326a295bcd18bd42fba5ed88800c5a688f885b87147aacaa5f5001"
        }

        mapping = self.get_station_configuration_mapping(
            stid, wmo_id="460", station_type="land"
        )
        file_hashes = run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertDictEqual(
            expected_file_hashes,
            file_hashes,
        )
