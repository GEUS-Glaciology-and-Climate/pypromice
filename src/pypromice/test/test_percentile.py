import unittest
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from pypromice.qc.percentiles.outlier_detector import detect_outliers, filter_data


class PercentileQCTestCase(unittest.TestCase):
    def test_column_pattern_matches(self):
        self._test_column_pattern("p_i", True)

    def test_column_pattern_no_match(self):
        self._test_column_pattern("p_l", False)

    def test_column_pattern_with_prefix(self):
        self._test_column_pattern("prefix_p_i", False)

    def test_column_pattern_with_suffix(self):
        self._test_column_pattern("p_i_suffix", False)

    def _test_column_pattern(self, column_name: str, expected_output: bool):
        season_indices = pd.DatetimeIndex(
            [
                datetime(2022, 3, 1),
            ]
        )
        thresholds = pd.DataFrame(
            [
                dict(
                    stid="stid", variable_pattern="p_[iu]", lo=-100, hi=100, season=None
                ),
            ]
        )
        value_outside_range = -325
        input_data = pd.DataFrame(
            index=season_indices, columns=[column_name], data=[value_outside_range]
        )
        if expected_output:
            expected_mask = pd.DataFrame(
                index=season_indices, columns=[column_name], data=[expected_output]
            )
        else:
            expected_mask = pd.DataFrame(index=season_indices, columns=[], data=[])

        mask = detect_outliers(input_data, thresholds)

        pd.testing.assert_frame_equal(expected_mask, mask)

    def test_column_pattern_multicolumns(self):
        thresholds = pd.DataFrame(
            [
                dict(
                    stid="stid", variable_pattern="p_[iu]", lo=-100, hi=100, season=None
                ),
            ]
        )
        date_index = pd.DatetimeIndex([datetime(2022, 3, 1)])
        input_data = pd.DataFrame(
            index=date_index,
            data=[
                dict(
                    p_i=-10,
                    p_u=1000,
                    p_j=1000,
                )
            ],
        )
        # p_j is not in the mask because it doesn't match the pattern
        expected_mask = pd.DataFrame(
            index=date_index,
            data=[
                dict(
                    p_i=False,
                    p_u=True,
                )
            ],
        )

        mask = detect_outliers(input_data, thresholds)

        pd.testing.assert_frame_equal(expected_mask, mask)

    def test_no_season(self):
        season_indices = pd.DatetimeIndex(
            [
                datetime(2022, 3, 1),
                datetime(2022, 8, 1),
            ]
        )
        thresholds = pd.DataFrame(
            [
                dict(stid="stid", variable_pattern="p_i", lo=-100, hi=100, season=None),
            ]
        )
        input_data = pd.DataFrame(index=season_indices, columns=["p_i"], data=[0, -243])
        expected_mask = pd.DataFrame(
            index=season_indices, columns=["p_i"], data=[False, True]
        )

        mask = detect_outliers(input_data, thresholds)

        pd.testing.assert_frame_equal(expected_mask, mask)

    def test_season_filter_invalid_winter_and_spring(self):
        self._test_season_filter(
            input_values=[0, 0, 0, 0], expected_mask=[True, True, False, False]
        )

    def test_season_filter_invalid_summer(self):
        self._test_season_filter(
            input_values=[-10, -10, -10, -10], expected_mask=[False, False, True, False]
        )

    def test_season_filter_valid_season_values(self):
        self._test_season_filter(
            input_values=[-12, -8, -1, -3], expected_mask=[False, False, False, False]
        )

    def _test_season_filter(self, input_values: List[float], expected_mask: List[bool]):
        stid = "A_STID"
        thresholds = pd.DataFrame(
            [
                dict(
                    stid=stid, variable_pattern="t_i", lo=-20, hi=-10, season="winter"
                ),
                dict(stid=stid, variable_pattern="t_i", lo=-10, hi=-1, season="spring"),
                dict(stid=stid, variable_pattern="t_i", lo=-5, hi=5, season="summer"),
                dict(stid=stid, variable_pattern="t_i", lo=-10, hi=0, season="fall"),
            ]
        )
        season_indices = pd.DatetimeIndex(
            [
                datetime(2021, 12, 1),  # winter
                datetime(2022, 3, 1),  # spring
                datetime(2022, 6, 1),  # summer
                datetime(2022, 9, 1),  # fall
            ]
        )
        input_data = pd.DataFrame(
            index=season_indices, columns=["t_i"], data=input_values
        )
        expected_mask = pd.DataFrame(
            index=season_indices, columns=["t_i"], data=expected_mask
        )

        mask = detect_outliers(input_data, thresholds)

        pd.testing.assert_frame_equal(expected_mask, mask)

    def test_remove_outliers(self):
        thresholds = pd.DataFrame(
            columns=[
                "stid",
                "variable_pattern",
                "lo",
                "hi",
                "season",
            ],
            data=[
                ["stid", "t_[iu]", -40, 0, "winter"],
                ["stid", "t_[iu]", -4, 10, "summer"],
            ],
        )
        date_index = pd.DatetimeIndex(
            [
                datetime(2022, 1, 1),
                datetime(2022, 8, 1),
            ]
        )
        input_data = pd.DataFrame(
            index=date_index,
            data=[
                dict(t_i=-10, p_u=994),
                dict(t_i=37, p_u=1024),
            ],
        )
        mask = detect_outliers(input_data, thresholds)
        expected_output_data = input_data.copy()
        expected_output_data[mask] = np.nan

        output_data = filter_data(input_data, thresholds)
        self.assertIsNot(output_data, input_data)
        pd.testing.assert_frame_equal(output_data, expected_output_data)
