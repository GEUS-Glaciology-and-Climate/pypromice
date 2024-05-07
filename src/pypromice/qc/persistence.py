import logging
from typing import Mapping, Optional, Union, List, TypedDict

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "persistence_qc",
    "find_persistent_regions",
    "count_consecutive_persistent_values",
    "count_consecutive_true",
]

logger = logging.getLogger(__name__)


class FilterConfig(TypedDict):
    variable: str  # Regular expression for selecting variable
    max_diff: float
    period: int


DEFAULT_VARIABLE_THRESHOLDS: List[FilterConfig] = [
    {"variable": "^t_[uil]$", "max_diff": 0.0001, "period": 2},
    {"variable": "^p_i$", "max_diff": 0.0001, "period": 2},
    {"variable": "^p_[ul]$", "max_diff": 0.0001, "period": 24},
    # Relative humidity can be very stable around 100%.
    # "rh": {"max_diff": 0.0001, "period": 2},
]


def persistence_qc(
    ds: xr.Dataset,
    variable_thresholds: Optional[List[FilterConfig]] = None,
) -> xr.Dataset:
    """
    Detect and filter data points that seems to be persistent within a certain period.

    TODO: It could be nice to have a reference to the logger or description of the behaviour here.
    The AWS logger program is know to return the last successfully read value if it fails reading from the sensor.

    Parameters
    ----------
    ds : xr.Dataset
        Level 1 datset
    variable_thresholds : Mapping
        Define threshold dict to hold limit values, and the difference values.
        Limit values indicate how much a variable has to change to the previous value
        period is how many hours a value can stay the same without being set to NaN
        * are used to calculate and define all limits, which are then applied to *_u, *_l and *_i

    Returns
    -------
    ds_out : xr.Dataset
            Level 1 dataset with difference outliers set to NaN
    """

    # the differenceQC is not done on the Windspeed
    # Optionally examine flagged data by setting make_plots to True
    # This is best done by running aws.py directly and setting 'test_station'
    # Plots will be shown before and after flag removal for each var

    df = ds.to_dataframe()  # Switch to pandas

    if variable_thresholds is None:
        variable_thresholds = DEFAULT_VARIABLE_THRESHOLDS

    logger.debug(f"Running persistence_qc using {variable_thresholds}")

    for config in variable_thresholds:
        var_all = df.filter(regex=config["variable"]).columns
        logger.info('Apply filter "{}" on {}'.format(config, var_all))
        max_diff = config["max_diff"]  # loading persistent limit
        period = config["period"]  # loading diff period

        for v in var_all:
            if v in df:
                mask = find_persistent_regions(df[v], period, max_diff)
                n_masked = mask.sum()
                n_samples = len(mask)
                logger.debug(
                    f"Applying persistent QC in {v}. Filtering {n_masked}/{n_samples} samples"
                )
                # setting outliers to NaN
                df.loc[mask, v] = np.nan

    # Back to xarray, and re-assign the original attrs
    ds_out = df.to_xarray()
    ds_out = ds_out.assign_attrs(ds.attrs)  # Dataset attrs
    for x in ds_out.data_vars:  # variable-specific attrs
        ds_out[x].attrs = ds[x].attrs

    return ds_out


def find_persistent_regions(
    data: pd.Series,
    min_repeats: int,
    max_diff: float,
) -> pd.Series:
    """
    Algorithm that ensures values can stay the same within the outliers_mask
    """
    consecutive_true_df = count_consecutive_persistent_values(data, max_diff)
    persistent_regions = consecutive_true_df >= min_repeats
    # Ignore entries which already nan in the input data
    persistent_regions[data.isna()] = False
    return persistent_regions


def count_consecutive_persistent_values(
    data: pd.Series,
    max_diff: float,
) -> pd.Series:
    diff = data.ffill().diff().abs()  # forward filling all NaNs!
    mask: pd.Series = diff < max_diff
    return count_consecutive_true(mask)


def count_consecutive_true(
    series: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """
    Convert boolean series to integer series where the values represent the number of connective true values.

    Examples
    --------
    >>> count_consecutive_true(pd.Series([False, True, False, False, True, True, True, False, True]))
    pd.Series([0, 1, 0, 0, 1, 2, 3, 0, 1])

    Parameters
    ----------
    series
        Boolean pandas Series or DataFrame

    Returns
    -------
    consecutive_true_count
        Integer pandas Series or DataFrame with values representing the number of connective true values.

    """
    # assert series.dtype == bool
    cumsum = series.cumsum()
    is_first = series.astype("int").diff() == 1
    offset = (is_first * cumsum).replace(0, np.nan).fillna(method="ffill").fillna(0)
    return ((cumsum - offset + 1) * series).astype("int")
