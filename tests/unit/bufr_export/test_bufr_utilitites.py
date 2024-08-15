import datetime
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


from pypromice.postprocess.bufr_utilities import (
    write_bufr_message,
    read_bufr_message,
    BUFRVariables,
)


class MockException(Exception):
    pass


class BUFRExportTestCase(unittest.TestCase):
    @staticmethod
    def get_valid_variables() -> BUFRVariables:
        return BUFRVariables(
            wmo_id="04439",
            station_type="mobile",
            timestamp=datetime.datetime(2021, 10, 14, 6, 0),
            relativeHumidity=43.0,
            airTemperature=256.0,
            pressure=95400.0,
            windDirection=12.0,
            windSpeed=2.2,
            latitude=66,
            longitude=-48,
            heightOfStationGroundAboveMeanSeaLevel=1450,
            heightOfBarometerAboveMeanSeaLevel=1452,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=4.1,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=4.6,
        )

    @mock.patch("pypromice.postprocess.bufr_utilities.codes_write")
    def test_bufr_file_are_deleted_on_exception(self, codes_write_mock: mock.MagicMock):
        codes_write_mock.side_effect = MockException()
        with tempfile.TemporaryFile("w+b") as file:
            self.assertRaises(
                MockException,
                write_bufr_message,
                variables=self.get_valid_variables(),
                file=file,
            )
            self.assertEqual(
                0,
                file.tell(),
                "There shall not be written anything the the output file.",
            )

    def test_land_station(self):
        # Test write and read with a ground truth BUFR file for verification.
        ground_truth_path = Path(__file__).parent.joinpath("WEG_B.bufr")
        variables_src = BUFRVariables(
            wmo_id="460",
            station_type="land",
            timestamp=datetime.datetime(2023, 12, 19, 10, 0),
            relativeHumidity=41,
            airTemperature=263,
            pressure=100570,
            windDirection=108,
            windSpeed=8.2,
            latitude=71.14146,
            longitude=-51.22206,
            heightOfStationGroundAboveMeanSeaLevel=14,
            heightOfBarometerAboveMeanSeaLevel=16.5,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=2.4,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=2.9,
        )

        with tempfile.TemporaryFile("w+b") as fp:
            write_bufr_message(variables=variables_src, file=fp)
            fp.seek(0)
            variables_read = read_bufr_message(
                fp=fp,
            )
            fp.seek(0)
            file_hash = hashlib.sha256(fp.read()).hexdigest()

        # Test if we can write data to the file and read it back
        self.assertEqual(
            variables_src,
            variables_read,
        )

        # Test if the written data file are equal to a ground truth reference file
        with ground_truth_path.open("br") as fp:
            ground_truth_read = read_bufr_message(
                fp=fp,
            )
            fp.seek(0)
            ground_truth_hash = hashlib.sha256(fp.read()).hexdigest()

        pd.testing.assert_series_equal(
            pd.Series(ground_truth_read).sort_index(),
            pd.Series(variables_read).sort_index(),
        )
        self.assertEqual(ground_truth_hash, file_hash)

    def test_mobile_station(self):
        # Test write and read with a ground truth BUFR file for verification.
        ground_truth_path = Path(__file__).parent.joinpath("DY2.bufr")
        variables_src = BUFRVariables(
            wmo_id="04464",
            station_type="mobile",
            timestamp=datetime.datetime(2023, 12, 19, 10, 0),
            relativeHumidity=84,
            airTemperature=251.8,
            pressure=76360,
            windDirection=218,
            windSpeed=9.6,
            latitude=66.48248,
            longitude=-46.29428,
            heightOfStationGroundAboveMeanSeaLevel=2121.7,
            heightOfBarometerAboveMeanSeaLevel=2125.9,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=4.1,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=4.6,
        )

        with tempfile.TemporaryFile("w+b") as fp:
            write_bufr_message(variables=variables_src, file=fp)
            fp.seek(0)
            variables_read = read_bufr_message(
                fp=fp,
            )
            fp.seek(0)
            file_hash = hashlib.sha256(fp.read()).hexdigest()

        # Test if we can write data to the file and read it back
        pd.testing.assert_series_equal(
            pd.Series(variables_src).sort_index(),
            pd.Series(variables_read).sort_index(),
        )

        # Test if the written data file are equal to a ground truth reference file
        with ground_truth_path.open("br") as fp:
            ground_truth_read = read_bufr_message(
                fp=fp,
            )
            fp.seek(0)
            ground_truth_hash = hashlib.sha256(fp.read()).hexdigest()

        pd.testing.assert_series_equal(
            pd.Series(ground_truth_read).sort_index(),
            pd.Series(variables_read).sort_index(),
        )
        self.assertEqual(ground_truth_hash, file_hash)

    def test_nan_value_serialization(self):
        variables_src = BUFRVariables(
            wmo_id="04464",
            station_type="mobile",
            timestamp=datetime.datetime(2023, 12, 19, 10, 0),
            relativeHumidity=np.nan,
            airTemperature=np.nan,
            pressure=np.nan,
            windDirection=np.nan,
            windSpeed=np.nan,
            latitude=np.nan,
            longitude=np.nan,
            heightOfStationGroundAboveMeanSeaLevel=np.nan,
            heightOfBarometerAboveMeanSeaLevel=np.nan,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=np.nan,
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=np.nan,
        )

        with tempfile.TemporaryFile("br+") as fp:
            write_bufr_message(variables_src, fp)
            fp.seek(0)
            variables_read = read_bufr_message(fp)

        self.assertEqual(
            variables_src,
            variables_read,
        )

    def test_precision(self):
        """
        Test if the BUFRVariable rounding configurations aligns with the BUFR format.

        Use np.random.random() to generate high precision random values.
        """
        variables_src = BUFRVariables(
            wmo_id="04464",
            station_type="mobile",
            timestamp=datetime.datetime(2023, 12, 19, 10, 0),
            relativeHumidity=np.random.random(),
            airTemperature=np.random.random(),
            pressure=1000 * np.random.random(),
            windDirection=np.random.random(),
            windSpeed=np.random.random(),
            latitude=np.random.random(),
            longitude=np.random.random(),
            heightOfStationGroundAboveMeanSeaLevel=np.random.random(),
            heightOfBarometerAboveMeanSeaLevel=np.random.random(),
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=np.random.random(),
            heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=np.random.random(),
        )
        with tempfile.TemporaryFile("w+b") as fp:
            write_bufr_message(variables=variables_src, file=fp)
            fp.seek(0)
            variables_read = read_bufr_message(
                fp=fp,
            )

        self.assertEqual(
            variables_src,
            variables_read,
        )
