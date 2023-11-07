from pathlib import Path

import attrs
import numpy as np
import pandas as pd
import xarray as xr


__all__ = ["ThresholdBasedOutlierDetector"]

season_month_map = {
    "winter": {12, 1, 2},
    "spring": {3, 4, 5},
    "summer": {6, 7, 8},
    "fall": {9, 10, 11},
}


def get_season_index_mask(data_set: pd.DataFrame, season: str) -> np.ndarray:
    season_months = season_month_map.get(
        season, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    )
    return data_set.index.month.isin(season_months)[:, None]


def detect_outliers(data_set: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    masks = []

    season_index_mask = {
        season: get_season_index_mask(data_set, season)
        for season in thresholds["season"].unique()
    }

    for variable_pattern, pattern_configs in thresholds.groupby("variable_pattern"):
        df = data_set.filter(regex=f"^{variable_pattern}$")
        mask = None
        for _, season_config in pattern_configs.iterrows():
            threshold_mask = (df < season_config.lo) | (df > season_config.hi)
            season_mask = threshold_mask & season_index_mask[season_config.season]

            if mask is None:
                mask = season_mask
            else:
                mask |= season_mask
        masks.append(mask)

    return pd.concat(masks, axis=1)


def filter_data(data_set: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    mask = detect_outliers(data_set, thresholds)
    output_data = data_set.copy()
    output_data[mask] = np.nan
    return output_data


@attrs.define
class ThresholdBasedOutlierDetector:
    thresholds: pd.DataFrame = attrs.field()

    def filter_data(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Filter samples across all variables by assigning to nan
        """
        stid = ds.station_id

        stid_thresholds = self.thresholds.query(f"stid == '{stid}'")
        if stid_thresholds.empty:
            return ds

        data_df = ds.to_dataframe()  # Switch to pandas
        data_df = filter_data(
            data_set=data_df,
            thresholds=stid_thresholds,
        )

        ds_out: xr.Dataset = data_df.to_xarray()
        ds_out = ds_out.assign_attrs(ds.attrs)  # Dataset attrs
        for x in ds_out.data_vars:  # variable-specific attrs
            ds_out[x].attrs = ds[x].attrs

        return ds_out

    @classmethod
    def from_csv_config(cls, config_file: Path) -> "ThresholdBasedOutlierDetector":
        """
        Instantiate using explicit csv file with explicit thresholds

        The CSV file shall have the format:

        * Comma separated
        * First row is header
        * Columns
            * stid: Station id
            * variabel_pattern: regular expression filtering the variable name
            * lo: Low threshold
            * hi: High threshold
            * season: The season of the filter: [, winter, spring, summer, fall]. The empty string means all seasons

        """
        return cls(thresholds=pd.read_csv(config_file))

    @classmethod
    def default(cls) -> "ThresholdBasedOutlierDetector":
        """
        Instantiate using aws thresholds stored in the python package.
        Returns
        -------

        """
        default_thresholds_path = Path(__file__).parent.joinpath("thresholds.csv")
        return cls.from_csv_config(default_thresholds_path)
