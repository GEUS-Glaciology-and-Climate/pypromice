import datetime
import logging
import pickle
import sys
import unittest
import uuid
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np
import pandas as pd

from pypromice.postprocess.bufr_utilities import BUFRVariables
from pypromice.postprocess.get_bufr import (
    process_station,
    StationConfiguration,
    get_bufr,
    get_bufr_variables,
    write_station_configuration_mapping,
    load_station_configuration_mapping,
)
from tests.unit.bufr_export.test_get_bufr_integration import (
    DATA_DIR,
    run_get_bufr,
)

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
    level=logging.WARNING,
)

MOCK_BASE_STR = "pypromice.postprocess.get_bufr.{}"


class StationConfigurationTestCase(TestCase):
    def test_read(self):
        source_lines = [
            "[UPE_L]\n",
            'stid = "UPE_L"\n',
            'station_site = "UPE_L"\n',
            'project = "Promice"\n',
            'station_type = "mobile"\n',
            'wmo_id = "04423"\n',
            "barometer_from_gps = -0.25\n",
            "anemometer_from_sonic_ranger = 0.4\n",
            "temperature_from_sonic_ranger = 0.0\n",
            "height_of_gps_from_station_ground = 0.9\n",
            "sonic_ranger_from_gps = 1.3\n",
            "export_bufr = true\n",
            "skipped_variables = []\n",
            "positions_update_timestamp_only = false\n",
        ]
        source_io = StringIO()
        source_io.writelines(source_lines)
        source_io.seek(0)
        expected_configuration_mapping = {
            "UPE_L": StationConfiguration(
                stid="UPE_L",
                station_site="UPE_L",
                project="Promice",
                station_type="mobile",
                wmo_id="04423",
                barometer_from_gps=-0.25,
                anemometer_from_sonic_ranger=0.4,
                temperature_from_sonic_ranger=0.0,
                height_of_gps_from_station_ground=0.9,
                sonic_ranger_from_gps=1.3,
                export_bufr=True,
                comment=None,
                skipped_variables=[],
                positions_update_timestamp_only=False,
            )
        }

        station_configuration_mapping = load_station_configuration_mapping(source_io)

        self.assertDictEqual(
            expected_configuration_mapping,
            station_configuration_mapping,
        )

    def test_write_read(self):
        station_config = StationConfiguration(
            stid="UPE_L",
            station_site="UPE_L",
            project="Promice",
            station_type="mobile",
            wmo_id="04423",
            barometer_from_gps=-0.25,
            anemometer_from_sonic_ranger=0.4,
            temperature_from_sonic_ranger=0.0,
            height_of_gps_from_station_ground=0.9,
            sonic_ranger_from_gps=1.3,
            export_bufr=True,
            comment=None,
            skipped_variables=[],
            positions_update_timestamp_only=False,
        )
        config_mapping = {station_config.stid: station_config}
        source_io = StringIO()

        write_station_configuration_mapping(config_mapping, source_io)
        source_io.seek(0)
        read_mapping = load_station_configuration_mapping(source_io)

        self.assertDictEqual(
            config_mapping,
            read_mapping,
        )

    def test_write_read_minimal_config(self):
        station_config = StationConfiguration(stid="UPE_L")
        config_mapping = {station_config.stid: station_config}
        source_io = StringIO()

        write_station_configuration_mapping(config_mapping, source_io)
        source_io.seek(0)
        read_mapping = load_station_configuration_mapping(source_io)

        self.maxDiff = None
        self.assertEqual(
            station_config,
            config_mapping[station_config.stid],
        )
        self.assertDictEqual(
            config_mapping,
            read_mapping,
        )

    def test_write_read_empty_mapping(self):
        config_mapping = {}
        source_io = StringIO()

        write_station_configuration_mapping(config_mapping, source_io)
        source_io.seek(0)
        read_mapping = load_station_configuration_mapping(source_io)

        self.assertDictEqual(
            config_mapping,
            read_mapping,
        )


class BufrVariablesTestCase(TestCase):
    def test_bufr_variables_gcnet(self):
        self._test_bufr_variables(
            stid="DY2",
            wmo_id="04464",
            station_type="mobile",
            relativeHumidity=69.0,
            airTemperature=256.0,
            pressure=77300.0,
            windDirection=149.0,
            windSpeed=14.9,
            latitude=66.482488,
            longitude=-46.294266,
            heightOfStationGroundAboveMeanSeaLevel=2123.2,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=4.6,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=4.6,
            heightOfBarometerAboveMeanSeaLevel=2125.25,
        )

    def test_bufr_variables_promice_v2(self):
        self._test_bufr_variables(
            stid="NUK_L",
            wmo_id="04403",
            station_type="mobile",
            relativeHumidity=69.0,
            airTemperature=256.0,
            pressure=77300.0,
            windDirection=149.0,
            windSpeed=14.9,
            latitude=66.482488,
            longitude=-46.294266,
            heightOfStationGroundAboveMeanSeaLevel=2123.8,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=4.2,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=4.6,
            heightOfBarometerAboveMeanSeaLevel=2124.45,
        )

    def test_bufr_variables_promice_v3(self):
        self._test_bufr_variables(
            stid="QAS_Mv3",
            wmo_id="04441",
            station_type="mobile",
            relativeHumidity=69.0,
            airTemperature=256.0,
            pressure=77300.0,
            windDirection=149.0,
            windSpeed=14.9,
            latitude=66.482488,
            longitude=-46.294266,
            heightOfStationGroundAboveMeanSeaLevel=2123.8,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=4.2,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=4.6,
            heightOfBarometerAboveMeanSeaLevel=2126,
        )

    def test_none_values_in_config(self):
        timestamp = datetime.datetime.now()
        data = pd.Series(
            data=dict(
                rh_i=0.93,
                t_i=-21,
                name="",
                p_i=993,
                wdir_i=32.1,
                wspd_i=5.3,
                gps_lat_fit=66.0,
                gps_lon_fit=-46.0,
                gps_alt_fit=1094,
                z_boom_u_smooth=2.1,
            ),
            name=timestamp,
        )
        station_config = StationConfiguration(
            stid="A_STID",
            station_type="land",
            wmo_id="4201",
            barometer_from_gps=0.2,
            anemometer_from_sonic_ranger=0.1,
            temperature_from_sonic_ranger=1.3,
            height_of_gps_from_station_ground=2.1,
        )

        output = get_bufr_variables(
            data,
            station_configuration=station_config,
        )

        self.assertEqual(
            BUFRVariables(
                wmo_id=station_config.wmo_id,
                station_type=station_config.station_type,
                timestamp=timestamp,
                relativeHumidity=1.0,
                airTemperature=252.2,  # Converted to kelvin
                pressure=199300.0,
                windDirection=32.0,
                windSpeed=5.3,
                latitude=66.0,
                longitude=-46.0,
                heightOfStationGroundAboveMeanSeaLevel=1091.9,
                heightOfBarometerAboveMeanSeaLevel=1094.2,
                heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=3.4,
                heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=2.2,
            ),
            output,
        )

    @mock.patch("pypromice.postprocess.get_bufr.write_bufr_message")
    def _test_bufr_variables(
        self,
        write_bufr_message_mock: mock.MagicMock,
        stid: str,
        wmo_id: str,
        relativeHumidity: float,
        airTemperature: float,
        pressure: float,
        windDirection: float,
        windSpeed: float,
        latitude: float,
        longitude: float,
        heightOfStationGroundAboveMeanSeaLevel: float,
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH: float,
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD: float,
        heightOfBarometerAboveMeanSeaLevel: float,
        station_type: str,
    ):
        l3_src_filepath = DATA_DIR.joinpath("tx_l3_test1.csv")
        l3_src = pd.read_csv(l3_src_filepath)
        now_timestamp = datetime.datetime(2023, 12, 8)

        timestamps = {}
        run_get_bufr(
            l3_data=l3_src,
            now_timestamp=now_timestamp,
            latest_timestamps=timestamps,
            stid=stid,
            store_positions=True,
            time_limit="91d",
        )

        write_bufr_message_mock.assert_called_once()
        call = write_bufr_message_mock.call_args_list[0]
        expected_time = datetime.datetime(year=2023, month=12, day=7, hour=23)
        expected_bufr_variables = BUFRVariables(
            wmo_id=wmo_id,
            station_type=station_type,
            timestamp=expected_time,
            relativeHumidity=relativeHumidity,
            airTemperature=airTemperature,
            pressure=pressure,
            windDirection=windDirection,
            windSpeed=windSpeed,
            latitude=latitude,
            longitude=longitude,
            heightOfStationGroundAboveMeanSeaLevel=heightOfStationGroundAboveMeanSeaLevel,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD,
            heightOfBarometerAboveMeanSeaLevel=heightOfBarometerAboveMeanSeaLevel,
        )
        pd.testing.assert_series_equal(
            pd.Series(expected_bufr_variables),
            pd.Series(call.kwargs["variables"]),
        )


@mock.patch(MOCK_BASE_STR.format("get_bufr_variables"))
@mock.patch(MOCK_BASE_STR.format("write_bufr_message"))
@mock.patch(MOCK_BASE_STR.format("get_latest_data"))
@mock.patch(MOCK_BASE_STR.format("load_data"))
class ProcessStationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.file_path = mock.create_autospec(Path)
        self.output_path = mock.create_autospec(Path)
        self.now_timestamp = mock.create_autospec(datetime.datetime)
        self.time_limit = mock.create_autospec(str)
        self.stid = str(uuid.uuid4())
        self.station_configuration = mock.MagicMock()
        self.earliest_timestamp = mock.MagicMock()

    def test_process_station_no_new_data(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
    ):
        self.earliest_timestamp = datetime.datetime(2023, 10, 3)
        latest_data_datetime = datetime.datetime(2023, 10, 3)
        get_latest_data_mock.return_value = pd.Series(
            data={
                "p_i": -227.1,
                "t_i": -16.7,
                "rh_i": 84.6,
                "wspd_i": 14.83,
                "wdir_i": 142.2,
                "gps_lat": 66.482469,
                "gps_lon": -46.294232,
                "gps_alt": 2116.0,
                "z_boom_u": 4.1901,
                "gps_lat_fit": 66.482474,
                "gps_lon_fit": -46.294261,
                "gps_alt_fit": 2119.6,
                "z_boom_u_smooth": 4.2,
            },
            name=latest_data_datetime,
        )
        expected_output = {
            "timestamp": latest_data_datetime,
            "lat": 66.482474,
            "lon": -46.294261,
            "alt": 2119.6,
        }

        output = process_station(
            file_path=self.file_path,
            output_path=self.output_path,
            now_timestamp=self.now_timestamp,
            latest_timestamp=self.earliest_timestamp,
            time_limit=self.time_limit,
            stid=self.stid,
            station_configuration=self.station_configuration,
        )

        self.assertDictEqual(
            output,
            expected_output,
        )
        get_bufr_variables_mock.assert_not_called()
        write_bufr_message_mock.assert_not_called()

    def test_process_station_has_new_data(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
    ):
        self.earliest_timestamp = datetime.datetime(2023, 10, 2)
        latest_data_datetime = datetime.datetime(2023, 10, 3)
        get_latest_data_mock.return_value = pd.Series(
            data={
                "p_i": -227.1,
                "t_i": -16.7,
                "rh_i": 84.6,
                "wspd_i": 14.83,
                "wdir_i": 142.2,
                "gps_lat": 66.482469,
                "gps_lon": -46.294232,
                "gps_alt": 2116.0,
                "z_boom_u": 4.1901,
                "gps_lat_fit": 66.482474,
                "gps_lon_fit": -46.294261,
                "gps_alt_fit": 2119.6,
                "z_boom_u_smooth": 4.2,
            },
            name=latest_data_datetime,
        )
        expected_output = {
            "timestamp": latest_data_datetime,
            "lat": 66.482474,
            "lon": -46.294261,
            "alt": 2119.6,
        }

        output = process_station(
            file_path=self.file_path,
            output_path=self.output_path,
            now_timestamp=self.now_timestamp,
            latest_timestamp=self.earliest_timestamp,
            time_limit=self.time_limit,
            stid=self.stid,
            station_configuration=self.station_configuration,
        )

        self.assertDictEqual(
            output,
            expected_output,
        )
        get_bufr_variables_mock.assert_called_once_with(
            data=get_latest_data_mock.return_value,
            station_configuration=self.station_configuration,
        )
        write_bufr_message_mock.assert_called_once_with(
            variables=get_bufr_variables_mock.return_value,
            file=self.output_path.open().__enter__(),
        )

    def test_min_data_wx_failed(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
    ):
        self.earliest_timestamp = datetime.datetime(2023, 10, 2)
        latest_data_datetime = datetime.datetime(2023, 10, 3)
        get_latest_data_mock.return_value = pd.Series(
            data={
                "p_i": np.nan,
                "t_i": np.nan,
                "rh_i": 84.6,
                "wspd_i": 14.83,
                "wdir_i": 142.2,
                "gps_lat": 66.482469,
                "gps_lon": -46.294232,
                "gps_alt": 2116.0,
                "z_boom_u": 4.1901,
                "gps_lat_fit": 66.482474,
                "gps_lon_fit": -46.294261,
                "gps_alt_fit": 2119.6,
                "z_boom_u_smooth": 4.2,
            },
            name=latest_data_datetime,
        )
        expected_output = {
            "timestamp": latest_data_datetime,
            "lat": 66.482474,
            "lon": -46.294261,
            "alt": 2119.6,
        }

        output = process_station(
            file_path=self.file_path,
            output_path=self.output_path,
            now_timestamp=self.now_timestamp,
            latest_timestamp=self.earliest_timestamp,
            time_limit=self.time_limit,
            stid=self.stid,
            station_configuration=self.station_configuration,
        )

        # The BUFR export step shall be skipped
        get_bufr_variables_mock.assert_not_called()
        write_bufr_message_mock.assert_not_called()
        self.assertDictEqual(
            output,
            expected_output,
        )

    def test_min_data_pos_failed(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
    ):
        self.earliest_timestamp = datetime.datetime(2023, 10, 2)
        latest_data_datetime = datetime.datetime(2023, 10, 3)
        get_latest_data_mock.return_value = pd.Series(
            data={
                "p_i": -227.1,
                "t_i": -16.7,
                "rh_i": 84.6,
                "wspd_i": 14.83,
                "wdir_i": 142.2,
                "gps_lat": 66.482469,
                "gps_lon": -46.294232,
                "gps_alt": 2116.0,
                "z_boom_u": 4.1901,
                "gps_lat_fit": 66.482474,
                "gps_lon_fit": -46.294261,
                "gps_alt_fit": np.nan,
                "z_boom_u_smooth": 4.2,
            },
            name=latest_data_datetime,
        )
        expected_output = {
            "timestamp": latest_data_datetime,
            "lat": None,
            "lon": None,
            "alt": None,
        }

        output = process_station(
            file_path=self.file_path,
            output_path=self.output_path,
            now_timestamp=self.now_timestamp,
            latest_timestamp=self.earliest_timestamp,
            time_limit=self.time_limit,
            stid=self.stid,
            station_configuration=self.station_configuration,
        )

        # The BUFR export step shall be skipped
        get_bufr_variables_mock.assert_not_called()
        write_bufr_message_mock.assert_not_called()
        self.assertDictEqual(
            output,
            expected_output,
        )

    def test_no_valid_data(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
    ):
        get_latest_data_mock.return_value = None

        output = process_station(
            file_path=self.file_path,
            output_path=self.output_path,
            now_timestamp=self.now_timestamp,
            latest_timestamp=self.earliest_timestamp,
            time_limit=self.time_limit,
            stid=self.stid,
            station_configuration=self.station_configuration,
        )

        load_data_mock.assert_called_once()
        get_latest_data_mock.assert_called_once()
        write_bufr_message_mock.assert_not_called()
        get_bufr_variables_mock.assert_not_called()
        self.assertIsNone(output)

    def test_skipped_variables(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
    ):
        self.earliest_timestamp = datetime.datetime(2023, 10, 2)
        latest_data_datetime = datetime.datetime(2023, 10, 3)
        original_p_i = 42.0
        get_latest_data_mock.return_value = pd.Series(
            data={
                "p_i": original_p_i,
                "t_i": -16.7,
                "rh_i": 84.6,
                "wspd_i": 14.83,
                "wdir_i": 142.2,
                "gps_lat": 66.482469,
                "gps_lon": -46.294232,
                "gps_alt": 2116.0,
                "z_boom_u": 4.1901,
                "gps_lat_fit": 66.482474,
                "gps_lon_fit": -46.294261,
                "gps_alt_fit": 2119.6,
                "z_boom_u_smooth": 4.2,
            },
            name=latest_data_datetime,
        )
        self.station_configuration = StationConfiguration(
            stid="A_STID",
            station_site="A_STATION_SITE",
            station_type="mobile",
            wmo_id="04242",
            skipped_variables=["p_i"],
            height_of_gps_from_station_ground=1.4,
            barometer_from_gps=0.1,
            anemometer_from_sonic_ranger=0.1,
            temperature_from_sonic_ranger=0.2,
            export_bufr=True,
        )
        expected_output = {
            "timestamp": latest_data_datetime,
            "lat": 66.482474,
            "lon": -46.294261,
            "alt": 2119.6,
        }
        self.assertEqual(
            original_p_i,
            get_latest_data_mock.return_value["p_i"],
        )

        output = process_station(
            file_path=self.file_path,
            output_path=self.output_path,
            now_timestamp=self.now_timestamp,
            latest_timestamp=self.earliest_timestamp,
            time_limit=self.time_limit,
            stid=self.stid,
            station_configuration=self.station_configuration,
        )

        self.assertTrue(
            np.isnan(get_latest_data_mock.return_value["p_i"]),
            "p_i shall be set to nan since it is in skipped_variables",
        )
        self.assertDictEqual(
            output,
            expected_output,
        )
        get_bufr_variables_mock.assert_called_once_with(
            data=get_latest_data_mock.return_value,
            station_configuration=self.station_configuration,
        )
        write_bufr_message_mock.assert_called_once_with(
            variables=get_bufr_variables_mock.return_value,
            file=self.output_path.open().__enter__(),
        )


class GetBufrTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_root = TemporaryDirectory()
        self.root_path = Path(self.temporary_root.name)
        self.l3_data_root = self.root_path / "l3"
        self.l3_data_root.mkdir()
        self.bufr_root = self.root_path / "bufr"
        self.bufr_root.mkdir()

        self.positions_file_path = self.root_path / "positions.csv"
        self.positions_seed_path = self.root_path / "positions_seed.csv"
        self.timestamps_pickle_filepath = self.root_path / "latest_timestamps.pickle"
        self.station_configuration_path = self.root_path / "station_configuration.toml"

    def tearDown(self) -> None:
        self.temporary_root.cleanup()

    @mock.patch(MOCK_BASE_STR.format("process_station"))
    def test_process_station_raises_exception(
        self, process_station_mock: mock.MagicMock
    ):
        """
        get_bufr should skip stations where process_station raises exception
        """
        timestamps_pickle_filepath = self.root_path / "timestamps.pickle"
        stid = "THE_STID_FOR_A_STATION"
        input_file_path = self.root_path / f"{stid}_hourly.csv"
        process_station_mock.side_effect = Exception("Test exception")
        now_timestamp = datetime.datetime.now()
        self.assertFalse(self.positions_file_path.exists())
        self.assertFalse(timestamps_pickle_filepath.exists())

        get_bufr(
            store_positions=True,
            bufr_out=self.bufr_root,
            input_files=[input_file_path],
            positions_filepath=self.positions_file_path,
            station_configuration_path=None,
            timestamps_pickle_filepath=timestamps_pickle_filepath,
            now_timestamp=now_timestamp,
        )

        self.assertTrue(self.positions_file_path.exists())
        self.assertTrue(timestamps_pickle_filepath.exists())

    @mock.patch(MOCK_BASE_STR.format("process_station"))
    def test_multiple_stations(self, process_station_mock: mock.MagicMock):
        station_config01 = StationConfiguration(stid="station_01", export_bufr=True)
        station_config02 = StationConfiguration(stid="station_02", export_bufr=True)
        station_config03 = StationConfiguration(stid="station_03", export_bufr=False)
        process_station_return_values = {
            station_config01.stid: dict(
                timestamp=datetime.datetime(2023, 2, 1, 10), lat=1, lon=3, alt=31
            ),
            station_config02.stid: dict(
                timestamp=datetime.datetime(2023, 2, 1, 10), lat=2, lon=3, alt=31
            ),
            station_config03.stid: dict(
                timestamp=datetime.datetime(2023, 2, 1, 10), lat=3, lon=3, alt=31
            ),
        }
        process_station_mock.side_effect = (
            lambda **kwargs: process_station_return_values[
                kwargs["station_configuration"].stid
            ]
        )
        input_files = [
            self.root_path / f"{station_config01.stid}_hourly.csv",
            self.root_path / f"{station_config02.stid}_hourly.csv",
            self.root_path / f"{station_config03.stid}_hourly.csv",
        ]
        station_configs = {
            station_config01.stid: station_config01,
            station_config02.stid: station_config02,
            station_config03.stid: station_config03,
        }
        with self.station_configuration_path.open("w") as fp:
            write_station_configuration_mapping(
                station_configs,
                fp,
            )

        get_bufr(
            store_positions=True,
            bufr_out=self.bufr_root,
            input_files=input_files,
            positions_filepath=self.positions_file_path,
            station_configuration_path=self.station_configuration_path,
            timestamps_pickle_filepath=self.timestamps_pickle_filepath,
            positions_seed_path=None,
            now_timestamp=datetime.datetime.now(),
        )

        self.assertEqual(3, process_station_mock.call_count)
        read_positions = pd.read_csv(
            self.positions_file_path, index_col=0, parse_dates=["timestamp"]
        ).to_dict(orient="index")
        self.assertDictEqual(
            read_positions,
            process_station_return_values,
        )

    def test_no_stations(self):
        now_timestamp = datetime.datetime.now()
        self.assertFalse(self.positions_file_path.exists())
        self.assertFalse(self.timestamps_pickle_filepath.exists())

        get_bufr(
            store_positions=True,
            bufr_out=self.bufr_root,
            input_files=(),
            positions_filepath=self.positions_file_path,
            station_configuration_path=None,
            timestamps_pickle_filepath=self.timestamps_pickle_filepath,
            now_timestamp=now_timestamp,
        )

        self.assertTrue(self.positions_file_path.exists())
        self.assertTrue(self.timestamps_pickle_filepath.exists())
        positions = pd.read_csv(self.positions_file_path)
        pd.testing.assert_frame_equal(
            positions,
            pd.DataFrame(columns=["stid", "timestamp", "lat", "lon", "alt"], data=[]),
        )
        with self.timestamps_pickle_filepath.open("br") as fp:
            timestamps = pickle.load(fp)
        self.assertDictEqual(dict(), timestamps)

    @mock.patch(MOCK_BASE_STR.format("process_station"))
    def test_single_station(self, process_station_mock: mock.MagicMock):
        now_timestamp = datetime.datetime.now()
        stid = "THE_STID_FOR_A_STATION"
        input_file_path = self.root_path / f"{stid}_hourly.csv"
        station_configuration = StationConfiguration(stid=stid, export_bufr=True)
        with self.station_configuration_path.open("w") as fp:
            write_station_configuration_mapping(
                dict(stid=station_configuration),
                fp,
            )
        expected_output_path = self.bufr_root / f"{stid}.bufr"
        expected_latest_timestamp = now_timestamp - datetime.timedelta(days=2)
        expected_station_configuration = StationConfiguration(
            stid=stid, export_bufr=True
        )

        get_bufr(
            store_positions=True,
            bufr_out=self.bufr_root,
            input_files=[input_file_path],
            positions_filepath=self.positions_file_path,
            station_configuration_path=self.station_configuration_path,
            timestamps_pickle_filepath=self.timestamps_pickle_filepath,
            positions_seed_path=None,
            now_timestamp=now_timestamp,
        )

        process_station_mock.assert_called_once_with(
            file_path=input_file_path,
            output_path=expected_output_path,
            now_timestamp=now_timestamp,
            latest_timestamp=expected_latest_timestamp,
            time_limit="91d",
            stid=stid,
            station_configuration=expected_station_configuration,
        )

    @mock.patch(MOCK_BASE_STR.format("process_station"))
    def test_station_without_configuration(self, process_station_mock: mock.MagicMock):
        now_timestamp = datetime.datetime.now()
        stid = "STATION_ID"
        input_file_path = self.root_path / f"{stid}_hourly.csv"
        expected_station_configuration = StationConfiguration(stid=stid)
        expected_output_path = self.bufr_root / f"{stid}.bufr"

        get_bufr(
            store_positions=True,
            bufr_out=self.bufr_root,
            input_files=[input_file_path],
            positions_filepath=self.positions_file_path,
            station_configuration_path=None,
            timestamps_pickle_filepath=self.timestamps_pickle_filepath,
            positions_seed_path=None,
            now_timestamp=now_timestamp,
        )

        process_station_mock.assert_called_once_with(
            file_path=input_file_path,
            output_path=expected_output_path,
            now_timestamp=now_timestamp,
            latest_timestamp=now_timestamp - datetime.timedelta(days=2),
            time_limit="91d",
            stid=stid,
            station_configuration=expected_station_configuration,
        )

    @mock.patch(MOCK_BASE_STR.format("process_station"))
    def test_latest_timestamp(self, process_station_mock: mock.MagicMock):
        stid = "STATION_ID"
        now_timestamp = datetime.datetime(2022, 1, 5, 10, 21)
        latest_timestamp = datetime.datetime(2022, 1, 5, 10, 0)
        # Save latest timestamp to pickle file
        with self.timestamps_pickle_filepath.open("wb") as fp:
            pickle.dump({stid: latest_timestamp}, fp)
        input_file_path = self.root_path / f"{stid}_hourly.csv"

        get_bufr(
            store_positions=True,
            bufr_out=self.bufr_root,
            input_files=[input_file_path],
            positions_filepath=self.positions_file_path,
            station_configuration_path=None,
            timestamps_pickle_filepath=self.timestamps_pickle_filepath,
            positions_seed_path=None,
            now_timestamp=now_timestamp,
        )

        process_station_mock.assert_called_once_with(
            file_path=input_file_path,
            output_path=self.bufr_root / f"{stid}.bufr",
            now_timestamp=now_timestamp,
            latest_timestamp=latest_timestamp,
            time_limit="91d",
            stid=stid,
            station_configuration=StationConfiguration(stid=stid),
        )

    @mock.patch(MOCK_BASE_STR.format("process_station"))
    def test_update_timestamp_only(self, process_station_mock: mock.MagicMock):
        stid = "STATION_ID"
        # Prepare station config
        station_config = StationConfiguration(
            stid=stid, positions_update_timestamp_only=True
        )
        with self.station_configuration_path.open("w") as fp:
            write_station_configuration_mapping(
                config_mapping={station_config.stid: station_config},
                fp=fp,
            )
        input_file_path = self.root_path / f"{stid}_hourly.csv"
        seed_timestamp = datetime.datetime(2021, 10, 2, 10, 0)
        now_timestamp = datetime.datetime(2023, 3, 3, 5, 0)
        positions_seed = pd.DataFrame(
            columns=["stid", "timestamp", "lat", "lon", "alt"],
            data=[
                [stid, seed_timestamp, 65.0, -40.0, 800],
            ],
        )
        positions_seed.to_csv(self.positions_seed_path, index=False)
        process_station_mock.return_value = {
            "timestamp": now_timestamp,
            # All position values should be ignored
            "lat": None,
            "lot": np.nan,
            "alt": 2414.0,
        }
        # Only timestamp should be updated
        expected_positions = positions_seed.copy()
        expected_positions["timestamp"] = now_timestamp

        get_bufr(
            store_positions=True,
            bufr_out=self.bufr_root,
            input_files=[input_file_path],
            positions_filepath=self.positions_file_path,
            station_configuration_path=self.station_configuration_path,
            timestamps_pickle_filepath=self.timestamps_pickle_filepath,
            positions_seed_path=self.positions_seed_path,
            now_timestamp=now_timestamp,
        )

        positions = pd.read_csv(self.positions_file_path, parse_dates=["timestamp"])
        self.assertEqual(1, len(positions))
        pd.testing.assert_series_equal(
            positions.iloc[0],
            expected_positions.iloc[0],
        )

    def test_position_seed(self):
        """
        There are no data files available. get_bufr should use the position_seed for output positions.
        """
        positions_seed = pd.DataFrame(
            columns=["stid", "timestamp", "lat", "lon", "alt"],
            data=[
                ["STATION_A", datetime.datetime(2021, 10, 2), 65.0, -40.0, 800],
                ["STATION_B", datetime.datetime(2023, 11, 12), 66.0, -50.0, 1100],
            ],
        )
        positions_seed.to_csv(self.positions_seed_path, index=False)

        get_bufr(
            store_positions=True,
            bufr_out=self.bufr_root,
            input_files=(),
            positions_filepath=self.positions_file_path,
            station_configuration_path=None,
            timestamps_pickle_filepath=self.timestamps_pickle_filepath,
            positions_seed_path=self.positions_seed_path,
            now_timestamp=datetime.datetime.now(),
        )

        for p in self.root_path.glob("*"):
            print(p)

        positions = pd.read_csv(self.positions_file_path, parse_dates=["timestamp"])
        pd.testing.assert_frame_equal(positions, positions_seed)
