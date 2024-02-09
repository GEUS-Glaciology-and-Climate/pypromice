import datetime
import logging
import sys
from unittest import TestCase, mock

import pandas as pd

from pypromice.postprocess import wmo_config
from pypromice.postprocess.bufr_utilities import BUFRVariables
from pypromice.test.bufr_export.test_get_bufr_prerefactoring import DATA_DIR, run_get_bufr

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
    level=logging.WARNING,
)


class BufrVariablesTestCase(TestCase):

    def test_bufr_variables_gcnet(self):
        self._test_bufr_variables(
            stid="DY2",
            wmo_id='04464',
            station_type='mobile',
            relativeHumidity=69.,
            airTemperature=256.,
            pressure=77300.0,
            windDirection=149.,
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
            wmo_id='04403',
            station_type='mobile',
            relativeHumidity=69.,
            airTemperature=256.,
            pressure=77300.0,
            windDirection=149.,
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
            wmo_id='04441',
            station_type='mobile',
            relativeHumidity=69.,
            airTemperature=256.,
            pressure=77300.0,
            windDirection=149.,
            windSpeed=14.9,
            latitude=66.482488,
            longitude=-46.294266,
            heightOfStationGroundAboveMeanSeaLevel=2123.8,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=4.2,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=4.6,
            heightOfBarometerAboveMeanSeaLevel=2126,
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
            station_type: str):
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
            time_limit="3M",
            stid_to_skip=wmo_config.stid_to_skip,
        )

        write_bufr_message_mock.assert_called_once()
        call = write_bufr_message_mock.call_args_list[0]
        expected_time = datetime.datetime(year=2023, month=12, day=7, hour=23)
        expected_bufr_variables = BUFRVariables(
            stid=stid,
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
            pd.Series(call.kwargs['variables']),
        )


