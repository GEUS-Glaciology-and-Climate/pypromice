import datetime
import logging
import random
import sys
import tempfile
from io import BufferedWriter
from pathlib import Path
from unittest import TestCase, mock

import pandas as pd

from pypromice.io.bufr.bufr_utilities import BUFRVariables
from pypromice.io.bufr.get_bufr import (
    get_station_positions,
    get_bufr_variables,
    REQUIRED_KEYS,
    get_bufr,
)
from pypromice.io.bufr.station_configuration import StationConfiguration
from tests.utilities import get_station_configuration

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
    level=logging.WARNING,
)


class GetStationPositionsTestCase(TestCase):
    def test_all_data_available(self):
        """
        Test the get_station_positions function
        """
        timestamp = pd.to_datetime("2024-03-01 00:00:00")
        latest_data = pd.Series(
            name=timestamp,
            data={
                "gps_lat_fit": 78.52901,
                "gps_lon_fit": -56.8450358,
                "gps_alt_fit": 1968.561,
            },
        )

        positions = get_station_positions(latest_data=latest_data)

        self.assertDictEqual(
            positions,
            dict(
                timestamp=timestamp,
                lat=78.52901,
                lon=-56.8450358,
                alt=1968.561,
            ),
        )

    def test_missing_data(self):
        """
        Test the get_station_positions function with missing data
        """
        timestamp = pd.to_datetime("2024-03-01 00:00:00")
        latest_data = pd.Series(
            name=timestamp,
            data={
                "gps_lat_fit": 78.52901,
                "gps_lon_fit": -56.8450358,
            },
        )

        with self.assertRaises(KeyError):
            get_station_positions(latest_data=latest_data)

    def test_nan_latitude(self):
        """
        get_station_positions shall discard all position data if latitude is NaN
        """
        timestamp = pd.to_datetime("2024-03-01 00:00:00")
        latest_data = pd.Series(
            name=timestamp,
            data={
                "gps_lat_fit": float("nan"),
                "gps_lon_fit": -56.8450358,
                "gps_alt_fit": 1968.561,
            },
        )

        positions = get_station_positions(latest_data=latest_data)

        self.assertDictEqual(
            positions,
            dict(
                timestamp=timestamp,
                lat=None,
                lon=None,
                alt=None,
            ),
        )

    def test_nan_altitude(self):
        """
        get_station_positions shall discard all position data if altitude is NaN
        """
        timestamp = pd.to_datetime("2024-03-01 00:00:00")
        latest_data = pd.Series(
            name=timestamp,
            data={
                "gps_lat_fit": 78.52901,
                "gps_lon_fit": -56.8450358,
                "gps_alt_fit": float("nan"),
            },
        )

        positions = get_station_positions(latest_data=latest_data)

        self.assertDictEqual(
            positions,
            dict(
                timestamp=timestamp,
                lat=None,
                lon=None,
                alt=None,
            ),
        )


class TestGetBufrVariablesTestCase(TestCase):
    def test_bufr_variables_gcnet(self):
        config = StationConfiguration(
            stid="DY2",
            station_site="DY2",
            project="GC-Net",
            wmo_id="04464",
            station_type="mobile",
            barometer_from_gps=0.55,
            anemometer_from_sonic_ranger=0.4,
            temperature_from_sonic_ranger=0.4,
            height_of_gps_from_station_ground=1.5,
            sonic_ranger_from_gps=0.15,
            export_bufr=True,
        )
        timestamp = pd.to_datetime("2024-03-01 00:00:00")
        data = pd.Series(
            name=timestamp,
            data={
                "t_i": -12.5,
                "p_i": 3.1,
                "rh_i": 0.5,
                "wspd_i": 2.5,
                "wdir_i": 182.1,
                "z_boom_u_smooth": 1.6,
                "gps_lat_fit": 78.52901,
                "gps_lon_fit": -56.8450358,
                "gps_alt_fit": 1968.561,
            },
        )
        expected_bufr_variables = BUFRVariables(
            wmo_id=config.wmo_id,
            station_type=config.station_type,
            timestamp=timestamp,
            relativeHumidity=data.rh_i,
            airTemperature=data.t_i + 273.15,
            nonCoordinatePressure=100310,
            windDirection=data.wdir_i,
            windSpeed=data.wspd_i,
            latitude=data.gps_lat_fit,
            longitude=data.gps_lon_fit,
            heightOfStationGroundAboveMeanSeaLevel=data.gps_alt_fit
            - config.height_of_gps_from_station_ground,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=data.z_boom_u_smooth
            + config.temperature_from_sonic_ranger,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=data.z_boom_u_smooth
            + config.anemometer_from_sonic_ranger,
            heightOfBarometerAboveMeanSeaLevel=data.gps_alt_fit
            + config.barometer_from_gps,
        )

        bufr_variables = get_bufr_variables(
            data=data,
            station_configuration=config,
        )

        pd.testing.assert_series_equal(
            bufr_variables.as_series(),
            expected_bufr_variables.as_series(),
        )

    def test_bufr_variables_static_gps_elevation(self):
        timestamp = pd.to_datetime("2024-03-01 00:00:00")
        data = pd.Series(
            data=dict(
                rh_i=0.93,
                t_i=-21,
                name="",
                p_i=993,
                wdir_i=32.1,
                wspd_i=5.3,
                gps_lon_fit=-46.0,
                gps_lat_fit=66.0,
                # This is a erroneous value that should be overridden by the static value
                gps_alt_fit=142.1,
                z_boom_u_smooth=2.1,
            ),
            name=timestamp,
        )
        config = StationConfiguration(
            stid="A_STID",
            station_type="land",
            wmo_id="4201",
            export_bufr=True,
            barometer_from_gps=1.3,
            height_of_gps_from_station_ground=0.9,
            static_height_of_gps_from_mean_sea_level=17.5,
            anemometer_from_sonic_ranger=None,
            temperature_from_sonic_ranger=None,
            sonic_ranger_from_gps=None,
        )
        # The elevations should be determined from the static variable
        expected_station_ground_elevation = 17.5 - 0.9
        expected_barometer_elevation = 17.5 + 1.3
        expected_bufr_variables = BUFRVariables(
            wmo_id=config.wmo_id,
            station_type=config.station_type,
            timestamp=timestamp,
            relativeHumidity=1.0,
            airTemperature=252.15,  # Converted to kelvin
            nonCoordinatePressure=199300.0,
            windDirection=32.0,
            windSpeed=5.3,
            latitude=66.0,
            longitude=-46.0,
            heightOfStationGroundAboveMeanSeaLevel=expected_station_ground_elevation,
            heightOfBarometerAboveMeanSeaLevel=expected_barometer_elevation,
            # The sensor heights are ignored since the necessary dimension values are missing
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=float("nan"),
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=float("nan"),
        )

        bufr_variables = get_bufr_variables(
            data=data,
            station_configuration=config,
        )

        pd.testing.assert_series_equal(
            bufr_variables.as_series(),
            expected_bufr_variables.as_series(),
        )

    def test_fails_on_missing_dimension_values(self):
        """
        Test that get_bufr_variables raises an AttributeError if the data is missing
        """
        timestamp = pd.to_datetime("2024-03-01 00:00:00")
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
        config = StationConfiguration(
            stid="A_STID",
            station_type="land",
            wmo_id="4201",
            export_bufr=True,
        )

        with self.assertRaises(AttributeError):
            get_bufr_variables(
                data,
                station_configuration=config,
            )

    def test_nan_location_yields_none(self):
        config = get_station_configuration(export_bufr=True)
        data = pd.Series(
            name=pd.to_datetime("2024-03-01 00:00:00"),
            data={
                "t_i": -12.5,
                "p_i": 1003.1,
                "rh_i": 0.5,
                "wspd_i": 2.5,
                "wdir_i": 182.1,
                "z_boom_u_smooth": 1.6,
                "gps_lat_fit": 78.52901,
                "gps_lon_fit": float("nan"),
                "gps_alt_fit": 1968.561,
            },
        )

        return_value = get_bufr_variables(
            data,
            station_configuration=config,
        )

        self.assertIsNone(return_value)

    def test_nan_t_i_and_p_i_yields_none(self):
        config = get_station_configuration(export_bufr=True)
        data = pd.Series(
            name=pd.to_datetime("2024-03-01 00:00:00"),
            data={
                "t_i": float("nan"),
                "p_i": float("nan"),
                "rh_i": 0.5,
                "wspd_i": 2.5,
                "wdir_i": 182.1,
                "z_boom_u_smooth": 1.6,
                "gps_lat_fit": 78.52901,
                "gps_lon_fit": -56.8450358,
                "gps_alt_fit": 1968.561,
            },
        )

        return_value = get_bufr_variables(
            data,
            station_configuration=config,
        )

        self.assertIsNone(return_value)

    def test_missing_keys(self):
        config = get_station_configuration(export_bufr=True)
        for key in REQUIRED_KEYS:
            data = pd.Series(
                name=pd.to_datetime("2024-03-01 00:00:00"),
                data={
                    "t_i": -12.5,
                    "p_i": 1003.1,
                    "rh_i": 0.5,
                    "wspd_i": 2.5,
                    "wdir_i": 182.1,
                    "z_boom_u_smooth": 1.6,
                    "gps_lat_fit": 78.52901,
                    "gps_lon_fit": -56.8450358,
                    "gps_alt_fit": 1968.561,
                },
            )
            del data[key]

            with self.assertRaises(ValueError, msg=f"Key: {key}"):
                get_bufr_variables(
                    data=data,
                    station_configuration=config,
                )


MOCK_BASE_STR = "pypromice.postprocess.get_bufr.{}"


@mock.patch(MOCK_BASE_STR.format("get_station_positions"))
@mock.patch(MOCK_BASE_STR.format("get_bufr_variables"))
@mock.patch(MOCK_BASE_STR.format("write_bufr_message"))
@mock.patch(MOCK_BASE_STR.format("get_latest_data"))
@mock.patch(MOCK_BASE_STR.format("load_data"))
class TestGetBufrTestCase(TestCase):
    def test_has_new_data(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
        get_station_positions_mock: mock.MagicMock,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            output_path = root_path / "bufr_out"
            station_config = get_station_configuration(
                export_bufr=True,
                positions_update_timestamp_only=False,
            )
            input_file = root_path / "input" / f"{station_config.stid}_hour.csv"
            positions_filepath = root_path / "positions.csv"
            timestamps_pickle_filepath = root_path / "timestamps.pickle"
            now_timestamp = pd.to_datetime("2024-03-01 00:12:00")
            latest_timestamp = pd.to_datetime("2024-03-01 00:01:00")
            get_latest_data_mock.return_value.name = latest_timestamp
            get_station_positions_mock.return_value = dict(
                timestamp=latest_timestamp,
                lat=78.52901,
                lon=-56.8450358,
                alt=1968.561,
            )

            get_bufr(
                input_files=[input_file],
                station_configuration_mapping={station_config.stid: station_config},
                break_on_error=True,
                bufr_out=output_path,
                target_timestamp=now_timestamp,
                positions_filepath=positions_filepath,
                store_positions=True,
                timestamps_pickle_filepath=timestamps_pickle_filepath,
            )

            load_data_mock.assert_called_once_with(input_file, now_timestamp)
            get_latest_data_mock.assert_called_once_with(
                load_data_mock.return_value,
                lin_reg_time_limit="91d",
                vars_to_skip=station_config.skipped_variables,
            )
            get_station_positions_mock.assert_called_once_with(
                get_latest_data_mock.return_value
            )
            get_bufr_variables_mock.assert_called_once_with(
                get_latest_data_mock.return_value,
                station_config,
            )
            write_bufr_message_mock.assert_called_once_with(
                get_bufr_variables_mock.return_value,
                mock.ANY,
            )
            # Write bufr is invoked with an open file object. It is therefore necessary to check the path of the file
            expected_output_file_path = output_path / f"{station_config.stid}.bufr"
            output_file = write_bufr_message_mock.call_args[0][1]
            self.assertIsInstance(output_file, BufferedWriter)
            self.assertEqual(Path(output_file.name), expected_output_file_path)
            written_positions = pd.read_csv(
                positions_filepath, index_col=0, parse_dates=["timestamp"]
            )
            self.assertDictEqual(
                get_station_positions_mock.return_value,
                dict(written_positions.loc[station_config.stid]),
            )
            self.assertTrue(timestamps_pickle_filepath.exists())
            timestamps = pd.read_pickle(timestamps_pickle_filepath)
            self.assertDictEqual(
                timestamps,
                {station_config.stid: latest_timestamp},
            )

    def test_no_new_data(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
        get_station_positions_mock: mock.MagicMock,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            output_path = root_path / "bur_out"
            station_config = get_station_configuration(
                export_bufr=True,
                positions_update_timestamp_only=False,
            )
            input_file = root_path / "input" / f"{station_config.stid}_hour.csv"
            positions_filepath = root_path / "positions.csv"
            now_timestamp = pd.to_datetime("2024-03-01 00:12:00")
            # The latest data is two month old
            latest_timestamp = pd.to_datetime("2024-01-01 00:12:00")
            get_latest_data_mock.return_value.name = latest_timestamp
            get_station_positions_mock.return_value = dict(
                timestamp=latest_timestamp,
                lat=78.52901,
                lon=-56.8450358,
                alt=1968.561,
            )

            get_bufr(
                input_files=[input_file],
                station_configuration_mapping={station_config.stid: station_config},
                break_on_error=True,
                bufr_out=output_path,
                target_timestamp=now_timestamp,
                positions_filepath=positions_filepath,
                store_positions=True,
                timestamps_pickle_filepath=None,
                time_window_length=pd.to_timedelta("2d"),
            )

            get_latest_data_mock.assert_called_once_with(
                load_data_mock.return_value,
                lin_reg_time_limit="91d",
                vars_to_skip=station_config.skipped_variables,
            )
            get_station_positions_mock.assert_called_once_with(
                get_latest_data_mock.return_value
            )
            get_bufr_variables_mock.assert_not_called()
            write_bufr_message_mock.assert_not_called()
            written_positions = pd.read_csv(
                positions_filepath, index_col=0, parse_dates=["timestamp"]
            )
            self.assertDictEqual(
                get_station_positions_mock.return_value,
                dict(written_positions.loc[station_config.stid]),
            )

    def test_position_seed(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
        get_station_positions_mock: mock.MagicMock,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            positions_filepath = root_path / "positions.csv"
            positions_seed_path = root_path / "positions_seed.csv"
            positions_seed = pd.DataFrame(
                columns=["stid", "timestamp", "lat", "lon", "alt"],
                data=[
                    ["STATION_A", datetime.datetime(2021, 10, 2), 65.0, -40.0, 800],
                    ["STATION_B", datetime.datetime(2023, 11, 12), 66.0, -50.0, 1100],
                ],
            ).set_index("stid")
            positions_seed.to_csv(positions_seed_path, index=True)

            get_bufr(
                input_files=[],
                station_configuration_mapping=dict(),
                break_on_error=True,
                bufr_out=mock.create_autospec(Path),
                target_timestamp=mock.create_autospec(datetime.timedelta),
                positions_filepath=positions_filepath,
                positions_seed_path=positions_seed_path,
                store_positions=True,
                timestamps_pickle_filepath=None,
                time_window_length=pd.to_timedelta("2d"),
            )

            written_positions = pd.read_csv(
                positions_filepath, index_col="stid", parse_dates=["timestamp"]
            )
            pd.testing.assert_frame_equal(
                positions_seed,
                written_positions,
            )

    def test_no_input_paths(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
        get_station_positions_mock: mock.MagicMock,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            positions_filepath = root_path / "positions.csv"
            get_bufr(
                input_files=[],
                station_configuration_mapping=dict(),
                break_on_error=True,
                bufr_out=mock.create_autospec(Path),
                target_timestamp=mock.create_autospec(datetime.timedelta),
                positions_filepath=positions_filepath,
                store_positions=True,
                timestamps_pickle_filepath=None,
                time_window_length=pd.to_timedelta("2d"),
            )

            load_data_mock.assert_not_called()
            get_latest_data_mock.assert_not_called()
            get_station_positions_mock.assert_not_called()
            get_bufr_variables_mock.assert_not_called()
            write_bufr_message_mock.assert_not_called()
            # The positions file should be created, but empty
            self.assertTrue(positions_filepath.exists())
            written_positions = pd.read_csv(
                positions_filepath, index_col=0, parse_dates=["timestamp"]
            )
            self.assertEqual(0, len(written_positions))

    def test_get_latest_data_fails(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
        get_station_positions_mock: mock.MagicMock,
    ):
        """
        get_latest_data returns None when there are no valid data available for the staiton
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            positions_filepath = root_path / "positions.csv"
            station_config = get_station_configuration(
                export_bufr=True,
                positions_update_timestamp_only=False,
            )
            input_file = root_path / "input" / f"{station_config.stid}_hour.csv"
            target_timestamp = mock.create_autospec(datetime.timedelta)
            get_latest_data_mock.return_value = None
            get_bufr(
                input_files=[input_file],
                station_configuration_mapping={station_config.stid: station_config},
                break_on_error=True,
                bufr_out=mock.create_autospec(Path),
                target_timestamp=target_timestamp,
                positions_filepath=positions_filepath,
                store_positions=True,
                timestamps_pickle_filepath=None,
                time_window_length=pd.to_timedelta("2d"),
            )

            load_data_mock.assert_called_once_with(input_file, target_timestamp)
            get_latest_data_mock.assert_called_once_with(
                load_data_mock.return_value,
                lin_reg_time_limit="91d",
                vars_to_skip=station_config.skipped_variables,
            )
            get_station_positions_mock.assert_not_called()
            get_bufr_variables_mock.assert_not_called()
            write_bufr_message_mock.assert_not_called()
            self.assertTrue(positions_filepath.exists())

    def test_already_existing_in_latest_timestamps(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
        get_station_positions_mock: mock.MagicMock,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            output_path = root_path / "bufr_out"
            positions_filepath = root_path / "positions.csv"
            station_config = get_station_configuration(
                export_bufr=True,
                positions_update_timestamp_only=False,
            )
            now_timestamp = pd.to_datetime("2024-03-01 00:12:00")
            latest_timestamp = pd.to_datetime("2024-03-01 00:01:00")
            input_file = root_path / "input" / f"{station_config.stid}_hour.csv"
            timestamps_pickle_filepath = root_path / "timestamps.pickle"
            latest_timestamps = {station_config.stid: latest_timestamp}
            with timestamps_pickle_filepath.open("wb") as f:
                pd.to_pickle(latest_timestamps, f)
            get_latest_data_mock.return_value.name = latest_timestamp
            get_station_positions_mock.return_value = dict(
                timestamp=latest_timestamp,
                lat=78.52901,
                lon=-56.8450358,
                alt=1968.561,
            )

            get_bufr(
                input_files=[input_file],
                station_configuration_mapping={station_config.stid: station_config},
                break_on_error=True,
                bufr_out=output_path,
                target_timestamp=now_timestamp,
                positions_filepath=positions_filepath,
                store_positions=True,
                timestamps_pickle_filepath=timestamps_pickle_filepath,
            )

            get_station_positions_mock.assert_called_once()
            # The BUFR export should be skipped since the latest timestamp is already in the timestamps
            get_bufr_variables_mock.assert_not_called()
            write_bufr_message_mock.assert_not_called()
            self.assertTrue(positions_filepath.exists())

    def test_no_station_configuration(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
        get_station_positions_mock: mock.MagicMock,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            output_path = root_path / "bufr_out"
            positions_filepath = root_path / "positions.csv"
            station_id = "A_STID"
            now_timestamp = pd.to_datetime("2024-03-01 00:12:00")
            latest_timestamp = pd.to_datetime("2024-03-01 00:01:00")
            input_file = root_path / "input" / f"{station_id}_hour.csv"
            get_latest_data_mock.return_value.name = latest_timestamp
            get_station_positions_mock.return_value = dict(
                timestamp=latest_timestamp,
                lat=78.52901,
                lon=-56.8450358,
                alt=1968.561,
            )

            get_bufr(
                input_files=[input_file],
                station_configuration_mapping=dict(),
                break_on_error=True,
                bufr_out=output_path,
                target_timestamp=now_timestamp,
                positions_filepath=positions_filepath,
                store_positions=True,
                timestamps_pickle_filepath=None,
            )

            get_station_positions_mock.assert_called_once()
            get_bufr_variables_mock.assert_not_called()
            write_bufr_message_mock.assert_not_called()
            self.assertTrue(positions_filepath.exists())

    def test_update_timestamps_only(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
        get_station_positions_mock: mock.MagicMock,
    ):
        pass

    def test_cleans_up_when_on_exception(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
        get_station_positions_mock: mock.MagicMock,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            positions_filepath = root_path / "positions.csv"
            station_config = get_station_configuration(
                export_bufr=True,
                positions_update_timestamp_only=False,
            )
            input_file = root_path / "input" / f"{station_config.stid}_hour.csv"
            target_timestamp = mock.create_autospec(datetime.timedelta)
            get_latest_data_mock.side_effect = Exception("Test exception")

            get_bufr(
                input_files=[input_file],
                station_configuration_mapping={station_config.stid: station_config},
                break_on_error=False,
                bufr_out=mock.create_autospec(Path),
                target_timestamp=target_timestamp,
                positions_filepath=positions_filepath,
                store_positions=True,
                timestamps_pickle_filepath=None,
                time_window_length=pd.to_timedelta("2d"),
            )

            load_data_mock.assert_called_once_with(input_file, target_timestamp)
            get_latest_data_mock.assert_called_once_with(
                load_data_mock.return_value,
                lin_reg_time_limit="91d",
                vars_to_skip=station_config.skipped_variables,
            )
            get_station_positions_mock.assert_not_called()
            get_bufr_variables_mock.assert_not_called()
            write_bufr_message_mock.assert_not_called()
            self.assertTrue(positions_filepath.exists())

    def test_multiple_stations(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
        get_station_positions_mock: mock.MagicMock,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            positions_filepath = root_path / "positions.csv"
            output_path = root_path / "bufr_out"
            station_config1 = StationConfiguration(stid="station_01", export_bufr=True)
            station_config2 = StationConfiguration(stid="station_02", export_bufr=True)
            station_config3 = StationConfiguration(stid="station_03", export_bufr=False)
            station_configs = [station_config1, station_config2, station_config3]
            station_configuration_mapping = {
                config.stid: config for config in station_configs
            }
            input_files = [
                root_path / "input" / f"{config.stid}_hour.csv"
                for config in station_configs
            ]
            target_timestamp = pd.to_datetime("2024-03-01 00:12:00")
            latest_timestamp = pd.to_datetime("2024-03-01 00:01:00")
            get_latest_data_mock.return_value.name = latest_timestamp
            station_positions = [
                dict(
                    timestamp=latest_timestamp,
                    lat=random.random() * 180 - 90,
                    lon=random.random() * 360 - 180,
                    alt=2000 * random.random(),
                )
                for _ in range(3)
            ]
            get_station_positions_mock.side_effect = station_positions

            get_bufr(
                input_files=input_files,
                station_configuration_mapping=station_configuration_mapping,
                break_on_error=True,
                bufr_out=output_path,
                target_timestamp=target_timestamp,
                positions_filepath=positions_filepath,
                store_positions=True,
                timestamps_pickle_filepath=None,
                time_window_length=pd.to_timedelta("2d"),
            )

            self.assertTrue(positions_filepath.exists())
            self.assertEqual(3, get_station_positions_mock.call_count)
            self.assertEqual(2, write_bufr_message_mock.call_count)
            written_positions = pd.read_csv(
                positions_filepath, index_col=0, parse_dates=["timestamp"]
            )
            self.assertSetEqual(
                set(written_positions.index),
                {config.stid for config in station_configs},
            )

    def test_station_without_configuration(
        self,
        load_data_mock: mock.MagicMock,
        get_latest_data_mock: mock.MagicMock,
        write_bufr_message_mock: mock.MagicMock,
        get_bufr_variables_mock: mock.MagicMock,
        get_station_positions_mock: mock.MagicMock,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            positions_filepath = root_path / "positions.csv"
            output_path = root_path / "bufr_out"
            target_timestamp = datetime.datetime.now()
            stid = "STATION_ID"
            input_file_path = root_path / f"{stid}_hourly.csv"
            get_station_positions_mock.return_value = dict(
                timestamp=target_timestamp,
                lat=78.52901,
                lon=-56.8450358,
                alt=1968.561,
            )

            get_bufr(
                input_files=[input_file_path],
                station_configuration_mapping={},
                break_on_error=True,
                bufr_out=output_path,
                target_timestamp=target_timestamp,
                positions_filepath=positions_filepath,
                store_positions=True,
                timestamps_pickle_filepath=None,
                time_window_length=pd.to_timedelta("2d"),
            )

            get_latest_data_mock.assert_called_once()
            get_station_positions_mock.assert_called_once()
            get_bufr_variables_mock.assert_not_called()
            written_positions = pd.read_csv(
                positions_filepath, index_col=0, parse_dates=["timestamp"]
            )
            self.assertDictEqual(
                get_station_positions_mock.return_value,
                dict(written_positions.loc[stid]),
            )
