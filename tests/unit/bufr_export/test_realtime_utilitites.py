import datetime
import unittest
from io import StringIO

import numpy as np
import pandas as pd

from pypromice.postprocess.real_time_utilities import get_latest_data


class GetLatestDataTestCase(unittest.TestCase):
    def get_data(self) -> pd.DataFrame:
        # There has to be >=15 rows for calculating the gps fits
        csv_lines = [
            "time,p_i,t_i,rh_i,wspd_i,wdir_i,gps_lat,gps_lon,gps_alt,z_boom_u",
            "2023-12-06 14:00:00,-227.3,-15.5,87.5,15.01,127.4,66.482481,-46.294234,2127.0,4.1871",
            "2023-12-06 15:00:00,-227.7,-15.4,87.4,15.58,139.2,66.482488,-46.294176,2120.0,4.1877",
            "2023-12-06 16:00:00,-227.7,-15.2,87.6,15.41,129.4,66.48253,-46.294227,2142.0,4.1873",
            "2023-12-06 17:00:00,-227.5,-15.4,87.1,13.66,138.6,66.482494,-46.294307,2132.0,4.1907",
            "2023-12-06 18:00:00,-228.0,-15.1,87.7,15.7,141.5,66.482497,-46.294308,2129.0,4.1907",
            "2023-12-06 19:00:00,-227.6,-15.1,87.6,15.78,132.5,66.482497,-46.294204,2124.0,4.193",
            "2023-12-06 20:00:00,-226.5,-14.9,87.5,13.3,138.0,66.482467,-46.294334,2116.0,4.1857",
            "2023-12-06 21:00:00,-226.8,-15.0,87.4,13.94,135.1,66.482485,-46.294188,2127.0,4.1884",
            "2023-12-06 22:00:00,-226.6,-15.2,88.0,11.55,139.0,66.482503,-46.294225,2126.0,4.1873",
            "2023-12-06 23:00:00,-227.6,-15.5,87.8,12.48,166.9,66.482519,-46.294191,2123.0,4.1875",
            "2023-12-07 00:00:00,-227.8,-15.5,87.2,17.62,151.0,66.48254,-46.294238,2146.0,4.185",
            "2023-12-07 01:00:00,-227.3,-15.8,86.5,14.63,140.5,66.482461,-46.294258,2123.0,4.185",
            "2023-12-07 02:00:00,-227.6,-15.9,86.5,15.45,143.0,66.482492,-46.294182,2120.0,4.1885",
            "2023-12-07 03:00:00,-227.3,-15.9,85.2,15.22,148.4,66.482505,-46.294319,2126.0,4.1802",
            "2023-12-07 04:00:00,-226.9,-16.2,85.4,13.1,151.6,66.482458,-46.294284,2116.0,4.1893",
            "2023-12-07 05:00:00,-227.4,-16.4,85.5,15.53,144.2,66.48246,-46.294335,2125.0,4.1844",
            "2023-12-07 06:00:00,-227.1,-16.7,84.6,14.83,142.2,66.482469,-46.294232,2116.0,4.1901",
        ]
        return pd.read_csv(
            StringIO("\n".join(csv_lines)),
            parse_dates=["time"],
            index_col=0,
        )

    def test_1(self):
        data = self.get_data()
        expected_output = pd.Series(
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
                "gps_lat_fit": 66.4824788,
                "gps_lon_fit": -46.2942685,
                "gps_alt_fit": 2121.4118,
                "z_boom_u_smooth": 4.188,
            },
            name=datetime.datetime(2023, 12, 7, 6),
        )

        latest_data = get_latest_data(
            df=data,
            lin_reg_time_limit="1w",
        )

        pd.testing.assert_series_equal(latest_data, expected_output, rtol=1e-8)

    def test_has_no_data(self):
        data = self.get_data()
        # Remove all rows but keep columns
        data = data.iloc[0:0]

        latest_data = get_latest_data(
            df=data,
            lin_reg_time_limit="1w",
        )
        self.assertIsNone(latest_data)

    def test_latest_data_row_is_invalid(self):
        """
        The last line is invalid. get_latest_data shall therefore return the second last line
        """
        data = self.get_data()
        data.loc["2023-12-07 06:00:00", :] = np.nan
        expected_output_timestamp = datetime.datetime(2023, 12, 7, 5)
        expected_output = pd.Series(
            data={
                "p_i": -227.4,
                "t_i": -16.4,
                "rh_i": 85.5,
                "wspd_i": 15.53,
                "wdir_i": 144.2,
                "gps_lat": 66.48246,
                "gps_lon": -46.294335,
                "gps_alt": 2125.0,
                "z_boom_u": 4.1844,
                "gps_lat_fit": 66.4824828,
                "gps_lon_fit": -46.2942753,
                "gps_alt_fit": 2123.3088,
                "z_boom_u_smooth": 4.187,
            },
            name=expected_output_timestamp,
        )

        latest_data = get_latest_data(
            df=data,
            lin_reg_time_limit="1w",
        )

        pd.testing.assert_series_equal(latest_data, expected_output, rtol=1e-8)

    def test_latest_data_has_some_invalid_values(self):
        """
        Return the latest data where there are some valid values.
        """
        data = self.get_data()
        # p_i is set to None to test the case multiple last valid indices
        data.loc["2023-12-07 06:00:00", "p_i"] = None
        expected_output = pd.Series(
            data={
                "p_i": np.nan,  # p_i shall be selected from the previous hour
                "t_i": -16.7,
                "rh_i": 84.6,
                "wspd_i": 14.83,
                "wdir_i": 142.2,
                "gps_lat": 66.482469,
                "gps_lon": -46.294232,
                "gps_alt": 2116.0,
                "z_boom_u": 4.1901,
                "gps_lat_fit": 66.4824788,
                "gps_lon_fit": -46.2942685,
                "gps_alt_fit": 2121.4118,
                "z_boom_u_smooth": 4.188,
            },
            name=datetime.datetime(2023, 12, 7, 6),
        )

        latest_data = get_latest_data(
            df=data,
            lin_reg_time_limit="1w",
        )

        pd.testing.assert_series_equal(latest_data, expected_output, rtol=1e-8)

    def test_all_instantaneous_timestamps_values_are_nan(self):
        data = self.get_data()
        data.loc[
            :,
            [
                "t_i",
                "p_i",
                "rh_i",
                "wspd_i",
                "wdir_i",
            ],
        ] = np.nan

        latest_data = get_latest_data(
            df=data,
            lin_reg_time_limit="1w",
        )

        self.assertIsNone(latest_data)

    def test_auxiliary_input_data(self):
        data = self.get_data()
        data["auxiliary_data"] = np.random.random(len(data))
        expected_output = data["auxiliary_data"].iloc[-1]

        latest_data = get_latest_data(
            df=data,
            lin_reg_time_limit="1w",
        )

        self.assertEqual(expected_output, latest_data["auxiliary_data"])
