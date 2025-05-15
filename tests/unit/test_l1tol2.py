import unittest

import pandas as pd
import xarray as xr
import numpy as np

from pypromice.process.wind import filter_wind_direction, calculate_directional_wind_speed


class DirectionalWindSpeedTestCase(unittest.TestCase):
    def test_get_directional_wind_speed_one_boom(self):
        n_entries = 4

        df_in = pd.DataFrame(
            data={
                "wspd_u": np.random.rand(n_entries) * 10,
                "wspd_i": np.random.rand(n_entries) * 10,
                "wdir_u": np.random.rand(n_entries) * 360,
                "wdir_i": np.random.rand(n_entries) * 360,
            },
            index=pd.date_range("2021-01-01", periods=n_entries, freq="H"),
        )
        ds = xr.Dataset(df_in)
        ds.attrs["number_of_booms"] = 1

        ds_out = ds.copy()

        # Calculate directional wind speed for upper boom
        ds_out['wdir_u'] = filter_wind_direction(ds_out['wdir_u'],
                                                 ds_out['wspd_u'])
        ds_out['wspd_x_u'], ds_out['wspd_y_u'] = calculate_directional_wind_speed(ds_out['wspd_u'],
                                                                                  ds_out['wdir_u'])

        # Calculate directional wind speed for lower boom
        if ds_out.attrs['number_of_booms'] == 2:
            ds_out['wdir_l'] = filter_wind_direction(ds_out['wdir_l'],
                                                      ds_out['wspd_l'])
            ds_out['wspd_x_l'], ds_out['wspd_y_l'] = calculate_directional_wind_speed(ds_out['wspd_l'],
                                                                                      ds_out['wdir_l'])

        # Calculate directional wind speed for instantaneous measurements
        if hasattr(ds_out, 'wdir_i'):
            if ~ds_out['wdir_i'].isnull().all() and ~ds_out['wspd_i'].isnull().all():
                ds_out['wdir_i'] = filter_wind_direction(ds_out['wdir_i'], ds_out['wspd_i'])
                ds_out['wspd_x_i'], ds_out['wspd_y_i'] = calculate_directional_wind_speed(ds_out['wspd_i'],
                                                                                          ds_out['wdir_i'])

        # Convert to dataframe for easier comparison
        df_out = ds_out.to_dataframe()
        new_columns = set(df_out.columns) - set(df_in.columns)
        expected_new_columns = {
            "wspd_x_u",
            "wspd_y_u",
            "wspd_x_i",
            "wspd_y_i",
        }
        self.assertSetEqual(new_columns, expected_new_columns)

    def test_get_directional_wind_speed_two_booms(self):
        n_entries = 4

        df_in = pd.DataFrame(
            data={
                "wspd_u": np.random.rand(n_entries) * 10,
                "wspd_l": np.random.rand(n_entries) * 10,
                "wspd_i": np.random.rand(n_entries) * 10,
                "wdir_u": np.random.rand(n_entries) * 360,
                "wdir_l": np.random.rand(n_entries) * 360,
                "wdir_i": np.random.rand(n_entries) * 360,
            },
            index=pd.date_range("2021-01-01", periods=n_entries, freq="H"),
        )
        ds = xr.Dataset(df_in)
        ds.attrs["number_of_booms"] = 2

        ds_out = ds.copy()

        # Calculate directional wind speed for upper boom
        ds_out['wdir_u'] = filter_wind_direction(ds_out['wdir_u'],
                                                 ds_out['wspd_u'])
        ds_out['wspd_x_u'], ds_out['wspd_y_u'] = calculate_directional_wind_speed(ds_out['wspd_u'],
                                                                                  ds_out['wdir_u'])

        # Calculate directional wind speed for lower boom
        if ds_out.attrs['number_of_booms'] == 2:
            ds_out['wdir_l'] = filter_wind_direction(ds_out['wdir_l'],
                                                      ds_out['wspd_l'])
            ds_out['wspd_x_l'], ds_out['wspd_y_l'] = calculate_directional_wind_speed(ds_out['wspd_l'],
                                                                                      ds_out['wdir_l'])

        # Calculate directional wind speed for instantaneous measurements
        if hasattr(ds_out, 'wdir_i'):
            if ~ds_out['wdir_i'].isnull().all() and ~ds_out['wspd_i'].isnull().all():
                ds_out['wdir_i'] = filter_wind_direction(ds_out['wdir_i'], ds_out['wspd_i'])
                ds_out['wspd_x_i'], ds_out['wspd_y_i'] = calculate_directional_wind_speed(ds_out['wspd_i'],
                                                                                          ds_out['wdir_i'])

        # Convert to dataframe for easier comparison
        df_out = ds_out.to_dataframe()
        new_columns = set(df_out.columns) - set(df_in.columns)
        expected_new_columns = {
            "wspd_x_u",
            "wspd_y_u",
            "wspd_x_i",
            "wspd_y_i",
            "wspd_x_l",
            "wspd_y_l",
        }
        self.assertSetEqual(new_columns, expected_new_columns)

    def test_get_directional_wind_speed_without_instantaneous_data(self):
        n_entries = 4

        df_in = pd.DataFrame(
            data={
                "wspd_u": np.random.rand(n_entries) * 10,
                "wdir_u": np.random.rand(n_entries) * 360,
            },
            index=pd.date_range("2021-01-01", periods=n_entries, freq="H"),
        )
        ds = xr.Dataset(df_in)
        ds.attrs["number_of_booms"] = 1

        ds_out = ds.copy()

        # Calculate directional wind speed for upper boom
        ds_out['wdir_u'] = filter_wind_direction(ds_out['wdir_u'],
                                                 ds_out['wspd_u'])
        ds_out['wspd_x_u'], ds_out['wspd_y_u'] = calculate_directional_wind_speed(ds_out['wspd_u'],
                                                                                  ds_out['wdir_u'])

        # Calculate directional wind speed for lower boom
        if ds_out.attrs['number_of_booms'] == 2:
            ds_out['wdir_l'] = filter_wind_direction(ds_out['wdir_l'],
                                                      ds_out['wspd_l'])
            ds_out['wspd_x_l'], ds_out['wspd_y_l'] = calculate_directional_wind_speed(ds_out['wspd_l'],
                                                                                      ds_out['wdir_l'])

        # Calculate directional wind speed for instantaneous measurements
        if hasattr(ds_out, 'wdir_i'):
            if ~ds_out['wdir_i'].isnull().all() and ~ds_out['wspd_i'].isnull().all():
                ds_out['wdir_i'] = filter_wind_direction(ds_out['wdir_i'], ds_out['wspd_i'])
                ds_out['wspd_x_i'], ds_out['wspd_y_i'] = calculate_directional_wind_speed(ds_out['wspd_i'],
                                                                                          ds_out['wdir_i'])

        # Convert to dataframe for easier comparison
        df_out = ds_out.to_dataframe()
        new_columns = set(df_out.columns) - set(df_in.columns)
        # The function should ignore instantaneous data if it is not present
        expected_new_columns = {
            "wspd_x_u",
            "wspd_y_u",
        }
        self.assertSetEqual(new_columns, expected_new_columns)
