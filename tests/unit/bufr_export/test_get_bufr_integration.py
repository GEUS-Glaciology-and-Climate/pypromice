"""
Automatic test to verify get_bufr generates exactly the same output files given the same parameters.
"""

import datetime
import logging
import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Dict
from unittest import TestCase

import numpy as np
import pandas as pd

from pypromice.postprocess import get_bufr
from pypromice.postprocess.bufr_utilities import read_bufr_message, BUFRVariables
from pypromice.station_configuration import (
    StationConfiguration,
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
    station_configuration_mapping: Dict[str, StationConfiguration],
    **get_bufr_kwargs,
) -> Optional[BUFRVariables]:
    """
    Run get_bufr using a temporary folder structure for input and output data

    Parameters
    ----------
    l3_data : Data frame with level 3 data
    stid : Station id for input data
    latest_timestamps

    Returns
    -------
    Optional[BUFRVariables]
        BUFR variables if the output file was generated successfully

    """
    with TemporaryDirectory() as output_path:
        output_path = Path(output_path)
        bufr_out = output_path.joinpath("BUFR_out")
        timestamps_pickle_filepath = output_path.joinpath("latest_timestamps.pickle")
        positions_filepath = output_path.joinpath("AWS_latest_locations.csv")
        l3_filepath = output_path.joinpath(f"{stid}_hour.csv")
        l3_data.to_csv(l3_filepath)

        if latest_timestamps is not None:
            with timestamps_pickle_filepath.open("wb") as fp:
                pickle.dump(latest_timestamps, fp)

        get_bufr.get_bufr(
            bufr_out=bufr_out,
            input_files=[l3_filepath],
            timestamps_pickle_filepath=timestamps_pickle_filepath,
            positions_filepath=positions_filepath,
            station_configuration_mapping=station_configuration_mapping,
            **get_bufr_kwargs,
        )

        output_path = bufr_out.joinpath(f"{stid}.bufr")
        if not output_path.exists():
            return None

        with output_path.open("rb") as fp:
            return read_bufr_message(fp)


class PreRefactoringBufrTestCase(TestCase):
    @staticmethod
    def get_station_configuration_mapping(
        stid: str,
        wmo_id: str,
        station_site: Optional[str] = None,
        station_type: str = "mobile",
        barometer_from_gps: float = 0.0,
        anemometer_from_sonic_ranger: float = 0.4,
        temperature_from_sonic_ranger: float = -0.1,
        height_of_gps_from_station_ground: float = 1.0,
        sonic_ranger_from_gps: float = 1.5,
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
                sonic_ranger_from_gps=sonic_ranger_from_gps,
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
        target_timestamp = datetime.datetime(2023, 12, 8)
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )
        expected_bufr_variables = BUFRVariables(
            wmo_id="04464",
            station_type="mobile",
            # Newest measurement in tx_l3_test1.csv: 2023-12-07 23:00:00
            timestamp=datetime.datetime(2023, 12, 7, 23, 00),
            relativeHumidity=69,
            airTemperature=255.95,
            pressure=77300.0,
            windDirection=149,
            windSpeed=14.9,
            latitude=66.48249,
            longitude=-46.29427,
            heightOfStationGroundAboveMeanSeaLevel=2123.7,
            heightOfBarometerAboveMeanSeaLevel=2124.7,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=4.09,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=4.59,
        )
        pd.testing.assert_series_equal(
            bufr_data.as_series(),
            expected_bufr_variables.as_series(),
        )

    def test_get_bufr_has_new_data_dont_store_position(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"DY2": datetime.datetime(2023, 12, 1)}
        target_timestamp = datetime.datetime(2023, 12, 8)
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=False,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )
        expected_bufr_variables = BUFRVariables(
            wmo_id="04464",
            station_type="mobile",
            # Newest measurement in tx_l3_test1.csv: 2023-12-07 23:00:00
            timestamp=datetime.datetime(2023, 12, 7, 23, 00),
            relativeHumidity=69,
            airTemperature=255.95,
            pressure=77300.0,
            windDirection=149,
            windSpeed=14.9,
            latitude=66.48249,
            longitude=-46.29427,
            heightOfStationGroundAboveMeanSeaLevel=2123.7,
            heightOfBarometerAboveMeanSeaLevel=2124.7,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=4.09,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=4.59,
        )
        pd.testing.assert_series_equal(
            bufr_data.as_series(),
            expected_bufr_variables.as_series(),
        )

    def test_get_bufr_stid_to_skip(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"DY2": datetime.datetime(2023, 12, 1)}
        target_timestamp = datetime.datetime(2023, 12, 6)
        mapping = self.get_station_configuration_mapping(
            stid, wmo_id="04464", export_bufr=False
        )
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertIsNone(bufr_data)

    def test_get_bufr_has_no_data_newer_than_latests_timestamps(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {stid: datetime.datetime(2023, 12, 7, 23, 00)}
        target_timestamp = datetime.datetime(2023, 12, 8)

        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertIsNone(bufr_data)

    def test_get_bufr_includes_datasets_not_in_latests_timestamps(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        latest_timestamps = {}
        target_timestamp = datetime.datetime(2023, 12, 8)
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )

        expected_bufr_variables = BUFRVariables(
            wmo_id="04464",
            station_type="mobile",
            # Newest measurement in tx_l3_test1.csv: 2023-12-07 23:00:00
            timestamp=datetime.datetime(2023, 12, 7, 23, 00),
            relativeHumidity=69,
            airTemperature=255.95,
            pressure=77300.0,
            windDirection=149,
            windSpeed=14.9,
            latitude=66.48249,
            longitude=-46.29427,
            heightOfStationGroundAboveMeanSeaLevel=2123.7,
            heightOfBarometerAboveMeanSeaLevel=2124.7,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=4.09,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=4.59,
        )
        pd.testing.assert_series_equal(
            bufr_data.as_series(),
            expected_bufr_variables.as_series(),
        )

    def test_get_bufr_has_old_data_compared_to_now(self):
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        latest_timestamps = {stid: datetime.datetime(2023, 12, 6)}
        target_timestamp = datetime.datetime(2023, 12, 20)

        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertIsNone(bufr_data)

    def test_invalid_value_at_last_index(self):
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        # Set some of instantanous values to nan
        l3_src.loc[140:, "p_i"] = np.nan
        latest_timestamps = {stid: datetime.datetime(2023, 12, 1)}
        target_timestamp = datetime.datetime(2023, 12, 8)
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )
        expected_bufr_variables = BUFRVariables(
            wmo_id="04464",
            station_type="mobile",
            # Newest measurement in tx_l3_test1.csv: 2023-12-07 23:00:00
            timestamp=datetime.datetime(2023, 12, 7, 23, 00),
            relativeHumidity=69,
            airTemperature=255.95,
            pressure=np.nan,
            windDirection=149,
            windSpeed=14.9,
            latitude=66.48249,
            longitude=-46.29427,
            heightOfStationGroundAboveMeanSeaLevel=2123.7,
            heightOfBarometerAboveMeanSeaLevel=2124.7,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=4.09,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=4.59,
        )
        pd.testing.assert_series_equal(
            bufr_data.as_series(),
            expected_bufr_variables.as_series(),
        )

    def test_invalid_position_data(self):
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        # Set some of instantanous values to nan
        l3_src.loc[:, "gps_lat"] = np.nan
        latest_timestamps = {stid: datetime.datetime(2023, 12, 1)}
        target_timestamp = datetime.datetime(2023, 12, 8)
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertIsNone(bufr_data)

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
        target_timestamp = datetime.datetime(2023, 12, 6)
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )

        self.assertIsNone(bufr_data)

    def test_multiple_last_valid_indices_all_older_than_2days(self):
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        # Set some of instantanous values to nan
        l3_src.loc[140:, "p_i"] = np.nan
        latest_timestamps = {stid: datetime.datetime(2023, 12, 1)}
        target_timestamp = datetime.datetime(2023, 12, 10)

        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertIsNone(bufr_data)

    def test_min_data_wx_failed(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        l3_src.loc[:, ["p_i", "t_i"]] = np.nan
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"DY2": datetime.datetime(2023, 12, 1)}
        target_timestamp = datetime.datetime(2023, 12, 6)
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )

        self.assertIsNone(bufr_data)

    def test_min_data_pos_failed(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        l3_src.loc[:, ["gps_lat"]] = np.nan
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"DY2": datetime.datetime(2023, 12, 1)}
        target_timestamp = datetime.datetime(2023, 12, 6)
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )
        self.assertIsNone(bufr_data)

    def test_ignore_newer_data_than_now_input(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "DY2"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {stid: datetime.datetime(2023, 12, 1)}
        # New is before the latest data
        target_timestamp = datetime.datetime(
            2023,
            12,
            6,
        )
        mapping = self.get_station_configuration_mapping(stid, wmo_id="04464")
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )
        expected_bufr_variables = BUFRVariables(
            wmo_id="04464",
            station_type="mobile",
            # Newest measurement in tx_l3_test1.csv: 2023-12-07 23:00:00 but target_timestamp is 2023-12-06
            timestamp=datetime.datetime(2023, 12, 6, 0, 0),
            relativeHumidity=82,
            airTemperature=250.85,
            pressure=77370.0,
            windDirection=153,
            windSpeed=10.4,
            latitude=66.48249,
            longitude=-46.29426,
            heightOfStationGroundAboveMeanSeaLevel=2123.3,
            heightOfBarometerAboveMeanSeaLevel=2124.3,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=4.09,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=4.59,
        )
        pd.testing.assert_series_equal(
            bufr_data.as_series(),
            expected_bufr_variables.as_series(),
        )

    def test_land_station_export(self):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        stid = "WEG_B"
        # Newest measurement in DY2_hour: 2023-12-07 23:00:00
        latest_timestamps = {"WEG_B": datetime.datetime(2023, 12, 1)}
        target_timestamp = datetime.datetime(2023, 12, 8)
        mapping = self.get_station_configuration_mapping(
            stid, wmo_id="460", station_type="land"
        )
        bufr_data = run_get_bufr(
            l3_data=l3_src,
            target_timestamp=target_timestamp,
            latest_timestamps=latest_timestamps,
            stid=stid,
            store_positions=True,
            linear_regression_time_limit="91d",
            station_configuration_mapping=mapping,
        )
        expected_bufr_variables = BUFRVariables(
            wmo_id="460",
            station_type="land",
            # Newest measurement in tx_l3_test1.csv: 2023-12-07 23:00:00
            timestamp=datetime.datetime(2023, 12, 7, 23, 00),
            relativeHumidity=69,
            airTemperature=255.95,
            pressure=77300.0,
            windDirection=149,
            windSpeed=14.9,
            latitude=66.48249,
            longitude=-46.29427,
            heightOfStationGroundAboveMeanSeaLevel=2123.7,
            heightOfBarometerAboveMeanSeaLevel=2124.7,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=4.09,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=4.59,
        )
        pd.testing.assert_series_equal(
            bufr_data.as_series(),
            expected_bufr_variables.as_series(),
        )
