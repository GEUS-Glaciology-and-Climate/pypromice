import unittest

import numpy as np
import pandas as pd
import xarray as xr

import pypromice.resources
from pypromice.process.L1toL2 import get_directional_wind_speed
from pypromice.process.value_clipping import clip_values


class ClipValuesTestCase(unittest.TestCase):
    def test_flag_wdir_on_nan_wspd(self):
        n_entries = 4
        df_in = pd.DataFrame(
            data={
                "wspd_u": n_entries * [np.nan],
                "wspd_i": n_entries * [np.nan],
                "wspd_l": n_entries * [np.nan],
                "wdir_u": np.random.rand(n_entries) * 360,
                "wdir_i": np.random.rand(n_entries) * 360,
                "wdir_l": np.random.rand(n_entries) * 360,
            },
            index=pd.date_range("2021-01-01", periods=n_entries, freq="H"),
        )
        ds = xr.Dataset(df_in)
        ds.attrs["number_of_booms"] = 2
        ds_out = get_directional_wind_speed(ds.copy())
        vars = pypromice.resources.load_variables(None)

        ds_out = clip_values(ds_out, vars)
        # Convert to dataframe for easier comparison
        df_out = ds_out.to_dataframe()

        # Assert all dir values are set nan
        self.assertTrue(df_out["wdir_u"].isna().all())
        self.assertTrue(df_out["wdir_l"].isna().all())
        self.assertTrue(df_out["wdir_i"].isna().all())

    def test_flag_wdir_zero_wspd(self):
        n_entries = 4
        df_in = pd.DataFrame(
            data={
                "wspd_u": n_entries * [0],
                "wspd_i": n_entries * [0],
                "wspd_l": n_entries * [0],
                "wdir_u": np.random.rand(n_entries) * 360,
                "wdir_i": np.random.rand(n_entries) * 360,
                "wdir_l": np.random.rand(n_entries) * 360,
            },
            index=pd.date_range("2021-01-01", periods=n_entries, freq="H"),
        )
        ds = xr.Dataset(df_in)
        ds.attrs["number_of_booms"] = 2
        ds_out = get_directional_wind_speed(ds.copy())
        vars = pypromice.resources.load_variables(None)

        ds_out = clip_values(ds_out, vars)

        # Convert to dataframe for easier comparison
        df_out = ds_out.to_dataframe()
        # Assert all dir values are set nan
        self.assertTrue(df_out["wdir_u"].isna().all())
        self.assertTrue(df_out["wdir_l"].isna().all())
        self.assertTrue(df_out["wdir_i"].isna().all())

    def test_flagging_depended_on_wspd(self):
        # This unit test tests variable conditions for flagging wdir values
        # Nan, 0, and negative values should be flagged while positive values should not
        n_entries = 4
        df_in = pd.DataFrame(
            data={
                "wspd_u": [np.nan, 0, 10, -3],
                "wdir_u": [0, 180, 90, 270],
            },
            index=pd.date_range("2021-01-01", periods=n_entries, freq="H"),
        )
        ds = xr.Dataset(df_in)
        ds.attrs["number_of_booms"] = 1
        ds_out = get_directional_wind_speed(ds.copy())
        vars = pypromice.resources.load_variables(None)

        ds_out = clip_values(ds_out, vars)

        # Convert to dataframe for easier comparison
        df_out = ds_out.to_dataframe()

        expected_dataframe = pd.DataFrame(
            data={
                "wspd_u": [np.nan, 0, 10, np.nan],
                "wdir_u": [np.nan, np.nan, 90, np.nan],
                "wspd_x_u": [np.nan, np.nan, 10, np.nan],
                "wspd_y_u": [np.nan, np.nan, 0, np.nan],
            },
            index=df_out.index,
        )
        self.assertEqual(
            df_out.columns.tolist(),
            expected_dataframe.columns.tolist(),
        )
        pd.testing.assert_frame_equal(
            df_out,
            expected_dataframe,
            check_dtype=True,
            check_like=True,
        )

    def test_recursive_flagging(self):
        fields = ["a", "b", "c"]
        variable_config = pd.DataFrame(
            columns=["field", "lo", "hi", "OOL"],
            data=[
                ["a", 0, 10, "b"],
                ["b", 100, 110, ""],
                ["c", 200, 210, "a"],
            ],
        ).set_index("field")
        data_index = pd.RangeIndex(4)
        data = pd.DataFrame(
            columns=fields,
            data=[
                [0, 100, 215],  # c is out of range
                [5, 115, 200],  # b is out of range
                [10, 100, 200],  # All a withing range
                [15, 100, 200],  # a is out of range
            ],
            dtype=float,
            index=data_index,
        )
        expected_output = pd.DataFrame(
            columns=fields,
            data=[
                [np.nan, np.nan, np.nan],  # c is nan -> a -> b
                [5, np.nan, 200],  # b is nan
                [10, 100, 200],
                [np.nan, np.nan, 200],  # a is nan -> b
            ],
            dtype=float,
            index=data.index,
        )

        data_set = xr.Dataset(data)

        data_set_out = clip_values(data_set, variable_config)
        data_frame_out = data_set_out.to_dataframe()

        pd.testing.assert_frame_equal(
            data_frame_out,
            expected_output,
            check_names=False,
            check_dtype=True,
        )

    def test_circular_dependencies(self):
        fields = ["a", "b", "c"]
        variable_config = pd.DataFrame(
            columns=["field", "lo", "hi", "OOL"],
            data=[
                ["a", 0, 10, "b"],
                ["b", 100, 110, "c"],
                ["c", 200, 210, "a"],
            ],
        ).set_index("field")
        data_index = pd.RangeIndex(4)
        data = pd.DataFrame(
            columns=fields,
            data=[
                [0, 100, 215],  # c is out of range
                [5, 115, 200],  # b is out of range
                [10, 100, 200],  # All a withing range
                [15, 100, 200],  # a is out of range
            ],
            dtype=float,
            index=data_index,
        )
        # All variables a dependent due to circular dependency
        expected_output = pd.DataFrame(
            columns=fields,
            data=[
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [10, 100, 200],
                [np.nan, np.nan, np.nan],
            ],
            dtype=float,
            index=data.index,
        )

        data_set = xr.Dataset(data)

        data_set_out = clip_values(data_set, variable_config)
        data_frame_out = data_set_out.to_dataframe()

        pd.testing.assert_frame_equal(
            data_frame_out,
            expected_output,
            check_names=False,
            check_dtype=True,
        )

    def test_rh_corrected_case(self):
        """
        The rh corrected variables are treated differently in the clipping function.
        """
        data_index = pd.RangeIndex(2)
        rh_u = pd.Series(index=data_index, data=[0, 54], name="rh_u")
        rh_u_cor = pd.Series(index=data_index, data=[0, np.nan], name="rh_u_cor")
        rh_l = pd.Series(index=data_index, data=[-20, 54], name="rh_l")
        rh_l_cor = pd.Series(index=data_index, data=[0, 254], name="rh_l_cor")
        data = pd.concat([rh_u, rh_u_cor, rh_l, rh_l_cor], axis=1)
        variable_config = pd.DataFrame(
            columns=["field", "lo", "hi", "OOL"],
            data=[
                ["rh_u", 0, 100, "rh_u_cor"],
                ["rh_u_cor", 0, 150, ""],
                ["rh_l_cor", np.nan, np.nan, ""],
                ["rh_l", 0, 100, "rh_l_cor"],
            ],
        ).set_index("field")
        data_set = xr.Dataset(data)

        data_set_out = clip_values(data_set, variable_config)

        # Convert to dataframe for easier comparison
        data_frame_out = data_set_out.to_dataframe()
        # The value of rh_u_cor should be nan since the input was nan
        self.assertTrue(np.isnan(data_frame_out.iloc[1]["rh_u_cor"]))
        # The value of rh_l_cor should not be changed since the hi threshold is not nan
        self.assertEqual(data_frame_out.iloc[1]["rh_l_cor"], 254)
        # The value of rh_l_cor should be nan since rh_l is below its threshold
        self.assertTrue(np.isnan(data_frame_out.iloc[0]["rh_l_cor"]))

    def test_nan_input(self):
        """
        Test that the function handles the case where nan input should cascade to child variables.
        """
        fields = ["a", "b"]
        variable_config = pd.DataFrame(
            columns=["field", "lo", "hi", "OOL"],
            data=[
                ["a", 0, 10, "b"],
                ["b", 100, 110, ""],
            ],
        ).set_index("field")
        data_index = pd.RangeIndex(2)
        data = pd.DataFrame(
            columns=fields,
            data=[
                [0, 100],  # All a withing range
                [np.nan, 100],  # a is nan
            ],
            dtype=float,
            index=data_index,
        )
        expected_output = pd.DataFrame(
            columns=fields,
            data=[
                [0, 100],
                [np.nan, np.nan],  # a is nan -> b
            ],
            dtype=float,
            index=data.index,
        )

        data_set = xr.Dataset(data)

        data_set_out = clip_values(data_set, variable_config)
        data_frame_out = data_set_out.to_dataframe()

        pd.testing.assert_frame_equal(
            data_frame_out,
            expected_output,
            check_names=False,
            check_dtype=True,
        )
