import datetime
import random
import unittest
from typing import Union

import numpy as np
import pandas as pd

from pypromice.qc.difference import find_static_regions


def get_random_datetime() -> datetime.datetime:
    # Select random timestamp in the period 1970-2030
    seconds_per_year = 365.25 * 24 * 3600
    timestamp = 60 * seconds_per_year * random.random()
    return datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)


def get_random_timeseries(
    start: Union[str, datetime.datetime],
    period: datetime.timedelta = datetime.timedelta(days=5),
    freq="1h",
    offset: float = -20,
    amplitude: float = 5,
) -> pd.Series:
    end = start + period
    time_range = pd.date_range(start=start, end=end, freq=freq, tz="utc")
    steps_per_day = "1d" / time_range.freq
    phase = np.random.random() * 2 * np.pi
    x = 2 * np.pi * np.arange(len(time_range)) / steps_per_day
    data = amplitude * np.sin(x + phase) + offset
    return pd.Series(index=time_range, data=data, name="data")


class DifferenceQATestCase(unittest.TestCase):
    def test_1_hour_static(self):
        series = get_random_timeseries(
            start=get_random_datetime(),
            period=datetime.timedelta(days=5),
            freq="1h",
        )
        index = 24
        series.iloc[index] = series.iloc[index - 1]

        mask = find_static_regions(series, diff_period=1, static_limit=0.001)

        self.assertEqual(1, mask.sum())
        self.assertTrue(1, mask.iloc[index])

    def test_1_hour_second_index(self):
        series = get_random_timeseries(
            start=get_random_datetime(),
            period=datetime.timedelta(days=5),
            freq="1h",
        )
        index = 1
        series.iloc[index] = series.iloc[index - 1]

        mask = find_static_regions(series, diff_period=1, static_limit=0.001)

        self.assertEqual(1, mask.sum())
        self.assertTrue(1, mask.iloc[index])

    def test_1_hour_last_index(self):
        series = get_random_timeseries(
            start=get_random_datetime(),
            period=datetime.timedelta(days=5),
            freq="1h",
        )
        index = -1
        series.iloc[index] = series.iloc[index - 1]

        mask = find_static_regions(series, diff_period=1, static_limit=0.001)

        self.assertEqual(1, mask.sum())
        self.assertTrue(1, mask.iloc[index])

    def test_no_static_period(self):
        series = get_random_timeseries(
            start=get_random_datetime(),
            period=datetime.timedelta(days=5),
            freq="1h",
        )

        static_mask = find_static_regions(series, diff_period=1, static_limit=0.001)

        pd.testing.assert_series_equal(
            pd.Series(index=static_mask.index, data=False),
            static_mask,
            check_names=False,
        )

    def test_static_period_longer_than_diff(self):
        series = get_random_timeseries(
            start=get_random_datetime(),
            period=datetime.timedelta(days=100),
            freq="1h",
        )
        index_start = 41
        index_length = 30
        index_end = index_start + index_length
        series.iloc[index_start:index_end] = series.iloc[index_start - 1]

        static_mask = find_static_regions(series, diff_period=24, static_limit=0.001)

        self.assertEqual(
            index_length,
            static_mask.sum(),
        )
        self.assertEqual(
            index_length,
            static_mask.iloc[index_start:index_end].sum(),
        )

    def test_diff_period_longer_than_static_period(self):
        series = get_random_timeseries(
            start=get_random_datetime(),
            period=datetime.timedelta(days=100),
            freq="1h",
        )
        index_start = 41
        index_length = 30
        index_end = index_start + index_length
        series.iloc[index_start:index_end] = series.iloc[index_start - 1]

        static_mask = find_static_regions(series, diff_period=31, static_limit=0.001)

        self.assertEqual(0, static_mask.sum())

    def test_static_period_at_the_end(self):
        series = get_random_timeseries(
            start=get_random_datetime(),
            period=datetime.timedelta(days=5),
            freq="1h",
        )
        index_length = 14
        series.iloc[-index_length:] = series.iloc[-index_length - 1]

        static_mask = find_static_regions(series, diff_period=10, static_limit=0.001)

        self.assertEqual(index_length, static_mask.sum())
        self.assertEqual(index_length, static_mask.iloc[-index_length:].sum())
