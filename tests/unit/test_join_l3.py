from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr

from pypromice.pipeline.join_l3 import get_valid_time_block, resolve_block_overlap


def _make_dataset(time_index, t_u_values):
    return xr.Dataset(
        {"t_u": ("time", t_u_values), "dsr": ("time", t_u_values)},
        coords={"time": time_index},
    )


class GetValidTimeBlockTestCase(TestCase):
    def test_splits_on_long_gap(self):
        # 100 days of hourly data, with t_u/dsr missing for days 20-60 (>30D gap)
        time_index = pd.date_range("2020-01-01", periods=100 * 24, freq="h")
        values = np.ones(len(time_index))
        gap_mask = (time_index >= "2020-01-20") & (time_index < "2020-03-01")
        values[gap_mask] = np.nan

        ds = _make_dataset(time_index, values)

        with TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / "TEST_STATION_mixed.nc"
            ds.to_netcdf(filepath)

            blocks = get_valid_time_block(
                {"stid": "TEST_STATION"}, str(filepath), isNead=False
            )

        self.assertEqual(len(blocks), 2)
        self.assertEqual(pd.to_datetime(blocks[0]["start_time"]), time_index[0])
        self.assertTrue(pd.to_datetime(blocks[0]["end_time"]) < pd.Timestamp("2020-01-20"))
        self.assertTrue(pd.to_datetime(blocks[1]["start_time"]) >= pd.Timestamp("2020-03-01"))
        self.assertEqual(pd.to_datetime(blocks[1]["end_time"]), time_index[-1])


class ResolveBlockOverlapTestCase(TestCase):
    def test_falls_back_to_older_station_during_newer_gap(self):
        time_index = pd.date_range("2020-01-01", periods=100 * 24, freq="h")

        # older station ("v2"): continuous coverage, value = 2
        old_values = np.full(len(time_index), 2.0)
        old_ds = _make_dataset(time_index, old_values)

        # newer station ("v3"): value = 3, but missing/failed for days 20-60
        new_values = np.full(len(time_index), 3.0)
        gap_mask = (time_index >= "2020-01-20") & (time_index < "2020-03-01")
        new_values[gap_mask] = np.nan
        new_ds = _make_dataset(time_index, new_values)

        old_blocks = [{
            "stid": "TEST_v2",
            "start_time": np.datetime64(time_index[0]),
            "end_time": np.datetime64(time_index[-1]),
            "dataset": old_ds,
        }]
        # split new_ds the same way get_valid_time_block would (two blocks around the gap)
        new_blocks = [
            {
                "stid": "TEST_v3",
                "start_time": np.datetime64(time_index[0]),
                "end_time": np.datetime64(pd.Timestamp("2020-01-19 23:00")),
                "dataset": new_ds.sel(time=slice(time_index[0], "2020-01-19 23:00")),
            },
            {
                "stid": "TEST_v3",
                "start_time": np.datetime64(pd.Timestamp("2020-03-01")),
                "end_time": np.datetime64(time_index[-1]),
                "dataset": new_ds.sel(time=slice("2020-03-01", time_index[-1])),
            },
        ]

        resolved = resolve_block_overlap(old_blocks + new_blocks)

        # newest-first ordering
        self.assertEqual([b["stid"] for b in resolved], ["TEST_v3", "TEST_v2", "TEST_v3"])

        def block_covering(timestamp):
            for b in resolved:
                if b["start_time"] <= np.datetime64(timestamp) <= b["end_time"]:
                    return b
            self.fail(f"no resolved block covers {timestamp}")

        # outside the newer station's gap, its data (v3, value 3) wins
        self.assertEqual(block_covering("2020-01-10")["stid"], "TEST_v3")
        self.assertEqual(block_covering("2020-01-10")["dataset"]["t_u"].sel(time="2020-01-10T00:00").item(), 3.0)
        self.assertEqual(block_covering("2020-03-15")["stid"], "TEST_v3")

        # during the newer station's gap, the older station (v2, value 2) fills in
        gap_block = block_covering("2020-02-01")
        self.assertEqual(gap_block["stid"], "TEST_v2")
        self.assertEqual(gap_block["dataset"]["t_u"].sel(time="2020-02-01T00:00").item(), 2.0)
