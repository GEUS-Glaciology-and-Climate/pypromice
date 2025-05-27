import unittest

import pandas as pd
import xarray as xr
import numpy as np

from pypromice.process.wind import filter_wind_direction, calculate_directional_wind_speed


class DirectionalWindSpeedTestCase(unittest.TestCase):
    def test_filter_wind_direction(self):
        n_entries = 4
        df_in = pd.DataFrame(
            data={
                "wspd_u": np.random.rand(n_entries) * 10,
                "wdir_u": np.random.rand(n_entries) * 360,
            },
            index=pd.date_range("2021-01-01", periods=n_entries, freq="H"),
        )

        ds = xr.Dataset(df_in)
        ds_out = ds.copy()

        # Calculate directional wind speed for upper boom
        ds_out['wdir_u_filtered'] = filter_wind_direction(ds_out['wdir_u'],
                                                          ds_out['wspd_u'])

        # Convert to dataframe for easier comparison
        df_out = ds_out.to_dataframe()
        new_columns = set(df_out.columns) - set(df_in.columns)
        expected_new_columns = {
            "wdir_u_filtered",
        }
        self.assertSetEqual(new_columns, expected_new_columns)

    def test_calculate_directional_wind_speed(self):
        n_entries = 4
        df_in = pd.DataFrame(
            data={
                "wspd_u": np.random.rand(n_entries) * 10,
                "wdir_u": np.random.rand(n_entries) * 360,
            },
            index=pd.date_range("2021-01-01", periods=n_entries, freq="H"),
        )

        ds = xr.Dataset(df_in)
        ds_out = ds.copy()

        # Calculate directional wind speed for upper boom
        ds_out['wspd_x_u'], ds_out['wspd_y_u'] = calculate_directional_wind_speed(ds_out['wspd_u'],
                                                                                  ds_out['wdir_u'])

        # Convert to dataframe for easier comparison
        df_out = ds_out.to_dataframe()
        new_columns = set(df_out.columns) - set(df_in.columns)
        expected_new_columns = {
            "wspd_x_u",
            "wspd_y_u",
        }
        self.assertSetEqual(new_columns, expected_new_columns)