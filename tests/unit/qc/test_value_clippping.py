import unittest

import numpy as np
import pandas as pd
import xarray as xr

import pypromice.resources
from pypromice.core.variables.wind import (
    filter_wind_direction,
    calculate_directional_wind_speed,
)
from pypromice.core.qc.value_clipping import clip_values


def _make_ds(df: pd.DataFrame) -> xr.Dataset:
    df = df.copy()
    df.index.name = "time"
    ds = xr.Dataset.from_dataframe(df)

    n = ds.sizes["time"]
    for v in list(ds.data_vars):
        if not v.endswith("_qc"):
            ds[f"{v}_qc"] = xr.DataArray(
                np.full(n, "OK", dtype=object),
                dims=("time",),
                coords={"time": ds["time"]},
            )
    return ds



def _prep_wind_vars(ds: xr.Dataset) -> xr.Dataset:
    ds_out = ds.copy()

    # filter_wind_direction now returns Dataset with QC updated
    ds_out = filter_wind_direction(ds_out, tag="_u")
    ds_out["wspd_x_u"], ds_out["wspd_y_u"] = calculate_directional_wind_speed(
        ds_out["wspd_u"], ds_out["wdir_u"]
    )

    if ds_out.attrs.get("number_of_booms", 1) == 2 and ("wdir_l" in ds_out) and ("wspd_l" in ds_out):
        ds_out = filter_wind_direction(ds_out, tag="_l")
        ds_out["wspd_x_l"], ds_out["wspd_y_l"] = calculate_directional_wind_speed(
            ds_out["wspd_l"], ds_out["wdir_l"]
        )

    if ("wdir_i" in ds_out) and ("wspd_i" in ds_out):
        if (~ds_out["wdir_i"].isnull().all()) and (~ds_out["wspd_i"].isnull().all()):
            ds_out = filter_wind_direction(ds_out, tag="_i")
            ds_out["wspd_x_i"], ds_out["wspd_y_i"] = calculate_directional_wind_speed(
                ds_out["wspd_i"], ds_out["wdir_i"]
            )

    return ds_out



def _run_clip(ds: xr.Dataset, variable_config: pd.DataFrame | None = None) -> xr.Dataset:
    vars_cfg = variable_config if variable_config is not None else pypromice.resources.load_variables(None)
    return clip_values(ds, vars_cfg)


def _assert_qc_all(ds: xr.Dataset, var: str, expected: str):
    qc = ds[f"{var}_qc"].to_series()
    assert (qc == expected).all()


def _assert_qc_equals(ds: xr.Dataset, var: str, expected: list[str]):
    qc = ds[f"{var}_qc"].to_series().tolist()
    assert qc == expected


class ClipValuesTestCase(unittest.TestCase):
    def test_flag_wdir_on_nan_wspd(self):
        n_entries = 4
        df_in = pd.DataFrame(
            data={
                "wspd_u": n_entries * [np.nan],
                "wdir_u": np.random.rand(n_entries) * 360,
            },
            index=pd.date_range("2021-01-01", periods=n_entries, freq="h"),
        )
        ds = _make_ds(df_in)
        ds_out = _run_clip(_prep_wind_vars(ds))

        assert (ds_out["wdir_u_qc"].to_series() == "DEPENDENCY").all()

    def test_flag_wdir_zero_wspd(self):
        n_entries = 4
        df_in = pd.DataFrame(
            data={
                "wspd_u": n_entries * [0.0],
                "wdir_u": np.random.rand(n_entries) * 360,
            },
            index=pd.date_range("2021-01-01", periods=n_entries, freq="h"),
        )
        ds = _make_ds(df_in)
        ds_out = _run_clip(_prep_wind_vars(ds))

        assert (ds_out["wdir_u_qc"].to_series() == "ZERO_WSPD").all()

    def test_flagging_depended_on_wspd(self):
        n_entries = 4
        df_in = pd.DataFrame(
            data={
                "wspd_u": [np.nan, 0, 10, -3],
                "wdir_u": [0, 180, 90, 270],
            },
            index=pd.date_range("2021-01-01", periods=n_entries, freq="h"),
        )
        ds = _make_ds(df_in)
        ds.attrs["number_of_booms"] = 1

        ds_out = _run_clip(_prep_wind_vars(ds))
        df_out = ds_out[["wspd_u", "wdir_u", "wspd_x_u", "wspd_y_u", "wdir_u_qc", "wspd_x_u_qc", "wspd_y_u_qc"]].to_dataframe()

        expected_qc = ["DEPENDENCY", "ZERO_WSPD", "OK", "DEPENDENCY"]
        assert df_out["wdir_u_qc"].tolist() == expected_qc
        expected_qc = ["OK", "DEPENDENCY", "OK", "DEPENDENCY"]
        assert df_out["wspd_x_u_qc"].tolist() == expected_qc
        assert df_out["wspd_y_u_qc"].tolist() == expected_qc

    def test_recursive_flagging(self):
        fields = ["a", "b", "c"]
        variable_config = pd.DataFrame(
            columns=["field", "lo", "hi", "dependent_variables"],
            data=[
                ["a", 0, 10, "b"],
                ["b", 100, 110, ""],
                ["c", 200, 210, "a"],
            ],
        ).set_index("field")

        data = pd.DataFrame(
            columns=fields,
            data=[
                [0, 100, 215],
                [5, 115, 200],
                [10, 100, 200],
                [15, 100, 200],
            ],
            dtype=float,
            index=pd.RangeIndex(4),
        )
        ds = _make_ds(data)
        ds_out = _run_clip(ds, variable_config)

        assert ds_out["c_qc"].to_series().tolist() == ["OOL", "OK", "OK", "OK"]
        assert ds_out["a_qc"].to_series().tolist() == ["DEPENDENCY", "OK", "OK", "OOL"]
        assert ds_out["b_qc"].to_series().tolist() == ["DEPENDENCY", "OOL", "OK", "DEPENDENCY"]

    def test_circular_dependencies(self):
        fields = ["a", "b", "c"]
        variable_config = pd.DataFrame(
            columns=["field", "lo", "hi", "dependent_variables"],
            data=[
                ["a", 0, 10, "b"],
                ["b", 100, 110, "c"],
                ["c", 200, 210, "a"],
            ],
        ).set_index("field")

        data = pd.DataFrame(
            columns=fields,
            data=[
                [0, 100, 215],
                [5, 115, 200],
                [10, 100, 200],
                [15, 100, 200],
            ],
            dtype=float,
            index=pd.RangeIndex(4),
        )
        ds = _make_ds(data)
        ds_out = _run_clip(ds, variable_config)

        assert ds_out["a_qc"].to_series().tolist() == ["DEPENDENCY", "DEPENDENCY", "OK", "OOL"]
        assert ds_out["b_qc"].to_series().tolist() == ["DEPENDENCY", "OOL", "OK", "DEPENDENCY"]
        assert ds_out["c_qc"].to_series().tolist() == ["OOL", "DEPENDENCY", "OK", "DEPENDENCY"]

    def test_rh_adjusted(self):
        variable_config = pd.DataFrame(
            columns=["field", "lo", "hi", "dependent_variables"],
            data=[
                ["rh_u", 0, 150, "rh_u_wrt_ice_or_water"],
                ["rh_u_wrt_ice_or_water", 0, 150, ""],
            ],
        ).set_index("field")

        rows_input = [
            dict(rh_u=42, rh_u_wrt_ice_or_water=43),
            dict(rh_u=-10, rh_u_wrt_ice_or_water=3),
            dict(rh_u=54, rh_u_wrt_ice_or_water=-4),
            dict(rh_u=160, rh_u_wrt_ice_or_water=120),
            dict(rh_u=100, rh_u_wrt_ice_or_water=255),
        ]
        df_input = pd.DataFrame(rows_input, dtype=float)
        ds = _make_ds(df_input)

        ds_out = _run_clip(ds, variable_config)

        assert ds_out["rh_u_qc"].to_series().tolist() == ["OK", "OOL", "OK", "OOL", "OK"]
        assert ds_out["rh_u_wrt_ice_or_water_qc"].to_series().tolist() == [
            "OK", "DEPENDENCY", "OOL", "DEPENDENCY", "OOL"
        ]

    def test_nan_input(self):
        fields = ["a", "b"]
        variable_config = pd.DataFrame(
            columns=["field", "lo", "hi", "dependent_variables"],
            data=[
                ["a", 0, 10, "b"],
                ["b", 100, 110, ""],
            ],
        ).set_index("field")

        data = pd.DataFrame(
            columns=fields,
            data=[
                [0, 100],
                [np.nan, 100],
            ],
            dtype=float,
            index=pd.RangeIndex(2),
        )
        ds = _make_ds(data)
        ds_out = _run_clip(ds, variable_config)

        assert ds_out["a_qc"].to_series().tolist() == ["OK", "OK"]
        assert ds_out["b_qc"].to_series().tolist() == ["OK", "DEPENDENCY"]


if __name__ == "__main__":
    unittest.main()
