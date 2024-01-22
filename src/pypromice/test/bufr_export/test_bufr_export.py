import datetime
import hashlib
import logging
import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Dict
from unittest import TestCase

import numpy as np
import pandas as pd

from pypromice.postprocess import get_bufr, wmo_config

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
    level=logging.WARNING,
)

DATA_DIR = Path(__file__).parent.parent.joinpath("data").absolute()


class GetBufrTestCase(TestCase):

    def _test_get_bufr(
        self,
        l3_data: pd.DataFrame,
        stid: str,
        now_timestamp: datetime.datetime,
        latest_timestamps: Optional[Dict[str, datetime.datetime]],
        expected_file_hashes: Dict[str, str],
        **get_bufr_kwargs,
    ):
        with TemporaryDirectory() as output_path:
            output_path = Path(output_path)
            bufr_out = output_path.joinpath("BUFR_out")
            timestamps_pickle_filepath = output_path.joinpath(
                "latest_timestamps.pickle"
            )
            positions_filepath = output_path.joinpath("AWS_latest_locations.csv")
            l3_filepath = output_path.joinpath(f"{stid}_hour.csv")
            l3_data.to_csv(l3_filepath)

            # Newest measurement in DY2_hour: 2023-12-07 23:00:00
            if latest_timestamps is not None:
                with timestamps_pickle_filepath.open("wb") as fp:
                    pickle.dump(latest_timestamps, fp)

            get_bufr.get_bufr(
                bufr_out=bufr_out.as_posix() + "/",
                l3_filepath=l3_filepath.as_posix(),
                timestamps_pickle_filepath=timestamps_pickle_filepath.as_posix(),
                positions_filepath=positions_filepath.as_posix(),
                now_timestamp=now_timestamp,
                **get_bufr_kwargs,
            )

            output_bufr_files = bufr_out.glob("*.bufr")
            file_hashes = dict()
            for p in output_bufr_files:
                with p.open("rb") as fp:
                    file_hashes[p.stem] = hashlib.sha256(fp.read()).hexdigest()
            self.assertDictEqual(
                expected_file_hashes,
                file_hashes,
            )

    def test_get_bufr_has_new_data(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"DY2": datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 6)
        expected_file_hashes = {
            stid: "2b94d2ef611cfddb6dd537ca63d0ec4fb5d8e880943f81a6d5e724c042ac8971"
        }
        self._test_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            expected_file_hashes=expected_file_hashes,
            stid=stid,
            store_positions=True,
            time_limit="3M",
            stid_to_skip=wmo_config.stid_to_skip,
        )

    def test_get_bufr_has_new_data_dont_store_position(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"DY2": datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 6)
        expected_file_hashes = {
            stid: "2b94d2ef611cfddb6dd537ca63d0ec4fb5d8e880943f81a6d5e724c042ac8971"
        }
        self._test_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            expected_file_hashes=expected_file_hashes,
            stid=stid,
            store_positions=False,
            time_limit="3M",
            stid_to_skip=wmo_config.stid_to_skip,
        )

    def test_get_bufr_stid_to_skip(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"DY2": datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 6)
        expected_file_hashes = {}
        self._test_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            expected_file_hashes=expected_file_hashes,
            stid=stid,
            store_positions=True,
            time_limit="3M",
            stid_to_skip={'TEST': [stid]},
        )

    def test_get_bufr_has_no_data_newer_than_latests_timestamps(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {stid: datetime.datetime(2023, 12, 7, 23, 00)}
        now_timestamp = datetime.datetime(2023, 12, 8)
        expected_file_hashes = {}

        self._test_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            expected_file_hashes=expected_file_hashes,
            stid=stid,
            store_positions=True,
            time_limit="3M",
            stid_to_skip=wmo_config.stid_to_skip,
        )

    def test_get_bufr_ignores_datasets_not_in_latests_timestamps(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        latest_timestamps = {}
        now_timestamp = datetime.datetime(2023, 12, 7)
        expected_file_hashes = {}

        self._test_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            expected_file_hashes=expected_file_hashes,
            stid=stid,

            store_positions=True,
            time_limit="3M",
            stid_to_skip=wmo_config.stid_to_skip,
        )


    def test_get_bufr_has_old_data_compared_to_now(self):
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        latest_timestamps = {stid: datetime.datetime(2023, 12, 6)}
        now_timestamp = datetime.datetime(2023, 12, 20)
        expected_file_hashes = {}

        self._test_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            expected_file_hashes=expected_file_hashes,
            stid=stid,

            store_positions=True,
            time_limit="3M",
            stid_to_skip=wmo_config.stid_to_skip,
        )

    def test_multiple_last_valid_indices(self):
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        # Set some of instantanous values to nan
        l3_src.loc[140:, "p_i"] = np.nan
        latest_timestamps = {stid: datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 6)
        expected_file_hashes = {
            stid: "bb951e0245ce3f6fe656b9bb5c85f097753a6969cc60b2cf8b34e0764495e627"
        }

        self._test_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            expected_file_hashes=expected_file_hashes,
            stid=stid,

            store_positions=True,
            time_limit="3M",
            stid_to_skip=wmo_config.stid_to_skip,
        )

    def test_multiple_last_valid_indices_all_instantaneous_timestamps_are_none(self):
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        # Set all instantanous values to nan
        l3_src.loc[:, ["t_i", "p_i", "rh_i", "wspd_i", "wdir_i", ]] = np.nan
        latest_timestamps = {stid: datetime.datetime(2023, 12, 1)}
        now_timestamp = datetime.datetime(2023, 12, 6)
        expected_file_hashes = {}

        self._test_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            expected_file_hashes=expected_file_hashes,
            stid=stid,
            store_positions=True,
            time_limit="3M",
            stid_to_skip=wmo_config.stid_to_skip,
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

        self._test_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            expected_file_hashes=expected_file_hashes,
            stid=stid,
            store_positions=True,
            time_limit="3M",
            stid_to_skip=wmo_config.stid_to_skip,
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
        self._test_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            expected_file_hashes=expected_file_hashes,
            stid=stid,
            store_positions=True,
            time_limit="3M",
            stid_to_skip=wmo_config.stid_to_skip,
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
        self._test_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=latest_timestamps,
            expected_file_hashes=expected_file_hashes,
            stid=stid,
            store_positions=True,
            time_limit="3M",
            stid_to_skip=wmo_config.stid_to_skip,
        )