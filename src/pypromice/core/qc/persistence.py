import logging
import numpy as np
import xarray as xr
from typing import Mapping, Optional
from pypromice.core.qc.common import remove_flagged_data, set_flag

__all__ = [
    "persistence_qc",
    "find_persistent_regions",
    "count_consecutive_persistent_values",
    "get_duration_consecutive_true",
]

logger = logging.getLogger(__name__)


DEFAULT_VARIABLE_THRESHOLDS = {

    # Period is given in hours, 2 persistent 10 min values will be flagged if period < 0.333
    "t_i": {"max_diff": 0.0001, "period": 2},
    "t_u": {"max_diff": 0.0001, "period": 2},
    "t_l": {"max_diff": 0.0001, "period": 2},

    "p_i": {"max_diff": 0.0001, "period": 3},
    "p_u": {"max_diff": 0.0001, "period": 150},
    "p_l": {"max_diff": 0.0001, "period": 150},

    # Gets special handling to remove simultaneously constant gps_lat and gps_lon
    "gps_lat_lon": {"max_diff": 0.000001, "period": 6},
    "gps_alt": {"max_diff": 0.0001, "period": 6},

    "t_rad": {"max_diff": 0.0001, "period": 2},

    # Gets special handling to allow constant 100%
    "rh_i": {"max_diff": 0.0001, "period": 2},
    "rh_u": {"max_diff": 0.0001, "period": 2},
    "rh_l": {"max_diff": 0.0001, "period": 2},

    "wspd_i": {"max_diff": 0.0001, "period": 6},
    "wspd_u": {"max_diff": 0.0001, "period": 6},
    "wspd_l": {"max_diff": 0.0001, "period": 6},
}


def persistence_qc(ds: xr.Dataset,
                   variable_thresholds: Optional[Mapping] = None
) -> xr.Dataset:
    """Flag persistent (frozen) sensor values without altering the data.

    Persistence is evaluated on a working copy where previously flagged samples
    are removed, but the output dataset preserves the original data values.
    Only the corresponding ``<var>_qc`` variables are updated by adding the
    "PERSISTENCE" flag where long periods of near-constant values are detected.

    Parameters
    ----------
    ds: xr.Dataset
        Level 1 dataset containing variables and quality control flag variables.
    variable_thresholds: Optional[Mapping]
        Mapping defining per-variable persistence criteria with keys ``max_diff``
        and ``period``. Keys like "t", "p", "rh", "wspd", "wdir", "z_boom"
        expand to ``*_u``, ``*_l``, ``*_i``.

    Returns
    -------
    xr.Dataset
        Dataset with original data unchanged and updated quality control flag
        variables containing additional "PERSISTENCE" flags."""
    if variable_thresholds is None:
        variable_thresholds = DEFAULT_VARIABLE_THRESHOLDS
        logger.debug(f"Running persistence_qc using {variable_thresholds}")
    else:
        logger.info(f"Running persistence_qc using custom thresholds:\n {variable_thresholds}")

    ds_out = ds.copy(deep=True)

    for k, params in variable_thresholds.items():
        if k in ["t", "p", "rh", "wspd", "wdir", "z_boom"]:
            var_all = [k + suffix for suffix in ["_u", "_l", "_i"]]
        else:
            var_all = [k]

        max_diff = params["max_diff"]
        period = params["period"]

        for v in var_all:
            if v in ds:
                mask = find_persistent_regions(ds[v], period, max_diff)

                # Special handling for rh values
                if "rh" in v:
                    mask = mask & (ds[v] < 99)

                if mask.any():
                    v_qc = ds_out[v].attrs.get["ancillary_variables"]
                    ds_out = set_flag(ds_out[v_qc],
                                      mask,
                                      "PERSISTENCE")

            elif v == "gps_lat_lon" and ("gps_lon" in ds) and ("gps_lat" in ds):
                mask_lon = find_persistent_regions(ds["gps_lon"], period, max_diff)
                mask_lat = find_persistent_regions(ds["gps_lat"], period, max_diff)
                mask = mask_lon & mask_lat

                if mask.any():
                    lon_qc = ds_out["gps_lon"].attrs["ancillary_variables"]
                    ds_out[lon_qc] = set_flag(ds_out[lon_qc],
                                              mask,
                                              "PERSISTENCE")

                    lat_qc = ds_out["gps_lat"].attrs["ancillary_variables"]
                    ds_out[lat_qc] = set_flag(ds_out[lat_qc],
                                              mask,
                                              "PERSISTENCE")

    return ds_out


def find_persistent_regions(data: xr.DataArray,
                            min_repeats: int,
                            max_diff: float
) -> xr.DataArray:
    """Identify regions of near-constant values in an xarray DataArray."""
    consecutive_counts = count_consecutive_persistent_values(data, max_diff)
    persistent = consecutive_counts >= min_repeats

    # Extend mask to cover min_repeats length
    for i in range(1, min_repeats):
        persistent = persistent | persistent.shift(time=-1, fill_value=False)

    # Remove NaNs from mask
    persistent = persistent.where(~data.isnull(), False)
    return persistent


def count_consecutive_persistent_values(data: xr.DataArray,
                                        max_diff: float
) -> xr.DataArray:
    """Count consecutive near-constant values in xarray."""
    # Forward-fill NaNs along time
    data_ffill = data.ffill(dim="time")

    # Absolute difference along time
    diff = abs(data_ffill.diff(dim="time", label="upper"))

    # Boolean mask for small differences
    mask = diff < max_diff

    # Compute consecutive durations
    return get_duration_consecutive_true(mask, data["time"])


def get_duration_consecutive_true(mask: xr.DataArray,
                                  time_coord: xr.DataArray
) -> xr.DataArray:
    """Count consecutive True values in a boolean mask and return duration in hours.
    The first value will be set to NaN, as it is not possible to calculate the
    duration of a single value.

    Parameters
    ----------
    mask: xr.DataArray
        Boolean data array
    time_coord: xr.DataArray
        Time coordinate array

    Returns
    -------
    xr. DataArray
        Arrange with values representing the number of connective true values
    """
    # Convert time to hours delta
    dt = time_coord.diff(dim="time") / np.timedelta64(1, "h")

    # Prepend first delta to match mask length
    dt = xr.concat([xr.DataArray([0], coords={"time": [time_coord[0]]}, dims="time"), dt], dim="time")

    # Convert mask to int
    mask_int = mask.astype(int)

    # Identify start of True sequences
    is_first = mask_int.diff(dim="time", label="upper") == 1

    # Cumulative sum of dt
    cumsum = dt.cumsum(dim="time")

    # Compute offset at start of sequences
    offset = (is_first * (cumsum - dt)).where(lambda x: x != 0).ffill(dim="time").fillna(0)

    # Duration in hours
    duration = (cumsum - offset) * mask_int
    return duration
