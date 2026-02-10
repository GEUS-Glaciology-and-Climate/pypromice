import logging

import numpy as np
import pandas as pd
import xarray as xr
from typing import Mapping, Optional, Union
from pypromice.core.qc.common import remove_flagged_data, set_flag

__all__ = [
    "persistence_qc",
    "find_persistent_regions",
    "count_consecutive_persistent_values",
    "get_duration_consecutive_true",
]

logger = logging.getLogger(__name__)

# period is given in hours, 2 persistent 10 min values will be flagged if period < 0.333
DEFAULT_VARIABLE_THRESHOLDS = {
    "t_i": {"max_diff": 0.0001, "period": 2},
    "t_u": {"max_diff": 0.0001, "period": 2},
    "t_l": {"max_diff": 0.0001, "period": 2},

    "p_i": {"max_diff": 0.0001, "period": 3},
    "p_u": {"max_diff": 0.0001, "period": 150},
    "p_l": {"max_diff": 0.0001, "period": 150},

    # gets special handling to remove simultaneously constant gps_lat and gps_lon
    "gps_lat_lon": {"max_diff": 0.000001, "period": 6},

    "gps_alt": {"max_diff": 0.0001, "period": 6},
    "t_rad": {"max_diff": 0.0001, "period": 2},

    # gets special handling to allow constant 100%
    "rh_i": {"max_diff": 0.0001, "period": 2},
    "rh_u": {"max_diff": 0.0001, "period": 2},
    "rh_l": {"max_diff": 0.0001, "period": 2},

    "wspd_i": {"max_diff": 0.0001, "period": 6},
    "wspd_u": {"max_diff": 0.0001, "period": 6},
    "wspd_l": {"max_diff": 0.0001, "period": 6},
}


def persistence_qc(
    ds: xr.Dataset,
    variable_thresholds: Optional[Mapping] = None,
) -> xr.Dataset:
    """Flag persistent (frozen) sensor values without altering the data.

    Persistence is evaluated on a working copy where previously flagged samples
    are removed, but the output dataset preserves the original data values.
    Only the corresponding ``<var>_qc`` variables are updated by adding the
    "PERSISTENCE" flag where long periods of near-constant values are detected.

    Args:
        ds: Level 1 dataset containing variables and optional ``*_qc`` variables.
        variable_thresholds: Mapping defining per-variable persistence criteria
            with keys ``max_diff`` and ``period``. Keys like "t", "p", "rh",
            "wspd", "wdir", "z_boom" expand to ``*_u``, ``*_l``, ``*_i``.

    Returns:
        xr.Dataset: Dataset with original data unchanged and updated ``*_qc``
        variables containing additional "PERSISTENCE" flags.
    """
    ds_work = remove_flagged_data(ds)
    df_work = ds_work.to_dataframe()

    if variable_thresholds is None:
        variable_thresholds = DEFAULT_VARIABLE_THRESHOLDS
        logger.debug(f"Running persistence_qc using {variable_thresholds}")
    else:
        logger.info(f"Running persistence_qc using custom thresholds:\n {variable_thresholds}")

    ds_out = ds.copy(deep=True)

    for k in variable_thresholds.keys():
        if k in ["t", "p", "rh", "wspd", "wdir", "z_boom"]:
            var_all = [k + l for l in ["_u", "_l", "_i"]]
        else:
            var_all = [k]

        max_diff = variable_thresholds[k]["max_diff"]
        period = variable_thresholds[k]["period"]

        for v in var_all:
            if v in df_work.columns:
                mask = find_persistent_regions(df_work[v], period, max_diff)
                if "rh" in v:
                    mask = mask & (df_work[v] < 99)

                if mask.any():
                    idx = df_work.index[mask]
                    ds_out = set_flag(
                        ds_out, v, "PERSISTENCE",
                        index_slice={"time": idx},
                    )

            elif v == "gps_lat_lon" and ("gps_lon" in df_work.columns) and ("gps_lat" in df_work.columns):
                mask = find_persistent_regions(df_work["gps_lon"], period, max_diff) & \
                       find_persistent_regions(df_work["gps_lat"], period, max_diff)

                if mask.any():
                    idx = df_work.index[mask]
                    ds_out = set_flag(ds_out, "gps_lon", "PERSISTENCE", index_slice={"time": idx})
                    ds_out = set_flag(ds_out, "gps_lat", "PERSISTENCE", index_slice={"time": idx})

    return ds_out




def find_persistent_regions(
    data: pd.Series,
    min_repeats: int,
    max_diff: float,
) -> pd.Series:
    """
    Algorithm that ensures values can stay the same within the outliers_mask
    """
    consecutive_true_df  = count_consecutive_persistent_values(data, max_diff)
    persistent_regions = consecutive_true_df  >= min_repeats
    for i in range(1, min_repeats):
        persistent_regions |= persistent_regions.shift(-1, fill_value=False)
        # Ignore entries which already nan in the input data
        persistent_regions[data.isna()] = False
    return persistent_regions


def count_consecutive_persistent_values(
    data: pd.Series,
    max_diff: float,
) -> pd.Series:
    diff = data.ffill().diff().abs()  # forward filling all NaNs!
    mask: pd.Series = diff < max_diff
    return get_duration_consecutive_true(mask)


def get_duration_consecutive_true(
    series: pd.Series,
) -> pd.Series:
    """
    From a boolean series, calculates the duration, in hours, of the periods with concecutive true values.

    The first value will be set to NaN, as it is not possible to calculate the duration of a single value.

    Examples
    --------
    >>> get_duration_consecutive_true(pd.Series([False, True, False, False, True, True, True, False, True]))
    pd.Series([np.nan, 1, 0, 0, 1, 2, 3, 0, 1])

    Parameters
    ----------
    pd.Series
        Boolean pandas Series or DataFrame

    Returns
    -------
    pd.Series
        Integer pandas Series or DataFrame with values representing the number of connective true values.

    """
    is_first = series.astype("int").diff() == 1
    delta_time = (series.index.to_series().diff().dt.total_seconds() / 3600)
    cumsum = delta_time.cumsum()
    offset = (is_first * (cumsum - delta_time)).replace(0, np.nan).ffill().fillna(0)

    return (cumsum - offset) * series
