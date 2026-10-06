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

    def test_backfills_same_station_gap_for_untested_variables(self):
        """A tested_vars-only outage splits one station's record into two
        blocks; the gap between them must not silently drop other variables
        that kept reporting through it (station_dataset backfill)."""
        time_index = pd.date_range("2020-01-01", periods=100 * 24, freq="h")

        t_u = np.full(len(time_index), -10.0)
        gap_mask = (time_index >= "2020-01-20") & (time_index < "2020-03-01")
        t_u[gap_mask] = np.nan

        # z_boom_u is not a tested variable and keeps reporting through the gap
        z_boom_u = np.full(len(time_index), 2.5)

        full_ds = xr.Dataset(
            {"t_u": ("time", t_u), "dsr": ("time", t_u), "z_boom_u": ("time", z_boom_u)},
            coords={"time": time_index},
        )

        blocks = [
            {
                "stid": "TEST",
                "start_time": np.datetime64(time_index[0]),
                "end_time": np.datetime64(pd.Timestamp("2020-01-19 23:00")),
                "dataset": full_ds.sel(time=slice(time_index[0], "2020-01-19 23:00")),
                "station_dataset": full_ds,
            },
            {
                "stid": "TEST",
                "start_time": np.datetime64(pd.Timestamp("2020-03-01")),
                "end_time": np.datetime64(time_index[-1]),
                "dataset": full_ds.sel(time=slice("2020-03-01", time_index[-1])),
                "station_dataset": full_ds,
            },
        ]

        resolved = resolve_block_overlap(blocks)

        self.assertEqual(len(resolved), 1)
        merged = resolved[0]["dataset"]

        # the merged block should span the full original range...
        self.assertEqual(pd.to_datetime(merged.time.values[0]), time_index[0])
        self.assertEqual(pd.to_datetime(merged.time.values[-1]), time_index[-1])

        # ...and z_boom_u, which was never missing, must not have lost the
        # samples that fell inside the tested_vars gap.
        self.assertEqual(
            int(merged["z_boom_u"].notnull().sum()),
            int(full_ds["z_boom_u"].notnull().sum()),
        )
        self.assertEqual(
            merged["z_boom_u"].sel(time="2020-02-01T00:00").item(), 2.5
        )


def _station_file(tmp_dir, name, time_index, t_u, extra=None):
    """Write a station file with t_u/dsr (the tested variables) and extras."""
    data = {"t_u": ("time", np.asarray(t_u, float)), "dsr": ("time", np.asarray(t_u, float))}
    for k, v in (extra or {}).items():
        data[k] = ("time", np.asarray(v, float))
    ds = xr.Dataset(data, coords={"time": time_index})
    path = Path(tmp_dir) / f"{name}_mixed.nc"
    ds.to_netcdf(path)
    return str(path)


class EdgeRowsTestCase(TestCase):
    """Rows before the first / after the last valid t_u/dsr that still hold
    data in other variables (e.g. the half-empty latest transmission)."""

    def setUp(self):
        self.time = pd.date_range("2020-01-01", periods=10 * 24, freq="h")
        n = len(self.time)
        # t_u/dsr valid only from day 2 to the end of day 8
        self.t_u = np.full(n, np.nan)
        self.t_u[24:8 * 24] = -10.0
        # battery-only rows at the start, instantaneous-only rows at the end
        self.batt = np.full(n, np.nan)
        self.batt[:24] = 12.0
        self.t_i = np.full(n, np.nan)
        self.t_i[8 * 24:] = -5.0
        self.extra = {"batt_v": self.batt, "t_i": self.t_i}

    def test_edge_rows_kept_when_nothing_overlaps(self):
        with TemporaryDirectory() as tmp:
            fp = _station_file(tmp, "A", self.time, self.t_u, self.extra)
            blocks = get_valid_time_block({"stid": "A"}, fp, isNead=False)
        self.assertEqual(sum(bool(b.get("is_edge")) for b in blocks), 2)  # head and tail

        resolved = resolve_block_overlap(blocks)
        self.assertEqual(len(resolved), 1)
        merged = resolved[0]["dataset"]
        self.assertEqual(pd.to_datetime(merged.time.values[0]), self.time[0])
        self.assertEqual(pd.to_datetime(merged.time.values[-1]), self.time[-1])
        self.assertEqual(merged.time.size, self.time.size)          # no gap, no duplicate
        self.assertTrue(merged.time.to_index().is_unique)
        self.assertEqual(int(merged["t_i"].notnull().sum()), int(np.isfinite(self.t_i).sum()))
        self.assertEqual(int(merged["batt_v"].notnull().sum()), 24)

    def test_regular_block_range_is_still_set_by_tested_variables(self):
        with TemporaryDirectory() as tmp:
            fp = _station_file(tmp, "A", self.time, self.t_u, self.extra)
            blocks = get_valid_time_block({"stid": "A"}, fp, isNead=False)
        regular = [b for b in blocks if not b.get("is_edge")]
        self.assertEqual(len(regular), 1)
        self.assertEqual(pd.to_datetime(regular[0]["start_time"]), self.time[24])
        self.assertEqual(pd.to_datetime(regular[0]["end_time"]), self.time[8 * 24 - 1])

    def test_edge_rows_never_replace_valid_data_of_another_station(self):
        """The newer station B only has battery rows (head edge) while the
        older station A still reports valid t_u: A must keep that period."""
        time = pd.date_range("2020-01-01", periods=10 * 24, freq="h")
        n = len(time)
        t_a = np.full(n, 1.0)                              # A: valid the whole time
        t_b = np.full(n, np.nan); t_b[5 * 24:] = 2.0      # B: t_u from day 5
        batt_b = np.full(n, np.nan); batt_b[:5 * 24] = 12.0  # B: battery rows from day 0
        with TemporaryDirectory() as tmp:
            fa = _station_file(tmp, "A", time, t_a)
            fb = _station_file(tmp, "B", time, t_b, {"batt_v": batt_b})
            blocks = (get_valid_time_block({"stid": "A"}, fa, isNead=False)
                      + get_valid_time_block({"stid": "B"}, fb, isNead=False))
        resolved = resolve_block_overlap(blocks)

        def at(ts):
            for b in resolved:
                if b["start_time"] <= np.datetime64(ts) <= b["end_time"]:
                    return b
            self.fail(f"nothing covers {ts}")

        # before B's t_u starts, A's data is used (not B's battery-only rows)
        self.assertEqual(at("2020-01-02")["stid"], "A")
        self.assertEqual(at("2020-01-02")["dataset"]["t_u"].sel(time="2020-01-02T00:00").item(), 1.0)
        # after, B (newer) takes over as before
        self.assertEqual(at("2020-01-08")["stid"], "B")
