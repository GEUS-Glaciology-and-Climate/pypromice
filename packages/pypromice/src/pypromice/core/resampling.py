import datetime
import numpy as np
import pandas as pd

DEFAULT_COMPLETENESS_THRESHOLDS = {
    "default": 0.8,
    "albedo": 0.2,
    "p_u": 0.5,
    "p_l": 0.5,
    "z_boom_u": 0.1,
    "z_boom_l": 0.1,
    "z_boom_cor_u": 0.1,
    "z_boom_cor_l": 0.1,
    "z_stake": 0.1,
    "z_stake_cor": 0.1,
    "z_surf_combined": 0.1,
    "t_i_1": 0.1,
    "t_i_2": 0.1,
    "t_i_3": 0.1,
    "t_i_4": 0.1,
    "t_i_5": 0.1,
    "t_i_6": 0.1,
    "t_i_7": 0.1,
    "t_i_8": 0.1,
    "t_i_9": 0.1,
    "t_i_10": 0.1,
    "t_i_11": 0.1,
    "gps_lat": 0.1,
    "gps_lon": 0.1,
    "gps_alt": 0.1,
    "batt_v": 0.1,
}

ALLOWED_TIME_STAMP_DURATIONS = (
    datetime.timedelta(minutes=10),
    datetime.timedelta(minutes=30),
    datetime.timedelta(hours=1),
    datetime.timedelta(hours=6),
    datetime.timedelta(days=1),
)


def classify_timestamp_durations(
        index: pd.DatetimeIndex,
) -> pd.TimedeltaIndex:
    """
    Classifies the durations between consecutive timestamps in a given DatetimeIndex.

    The function computes the time differences between consecutive timestamps and
    checks if these differences belong to a predefined set of allowed durations.
    It performs backward filling to handle missing values

    Parameters
    ----------
    index : pd.DatetimeIndex
        A pandas DatetimeIndex containing the timestamps to classify.

    Returns
    -------
    pd.TimedeltaIndex
        A TimedeltaIndex containing the classified durations for the corresponding
        timestamps in the input index.
    """
    return pd.TimedeltaIndex(
        index.to_series()
        .diff()
        .where(lambda d: d.isin(ALLOWED_TIME_STAMP_DURATIONS))
        .bfill()
    )


def get_completeness_mask(
    data_frame: pd.DataFrame,
    resample_offset: str,
    completeness_thresholds: dict[str, float] = DEFAULT_COMPLETENESS_THRESHOLDS,
    *,
    atol: float = 1e-9,
) -> pd.DataFrame:
    """
    Returns a completeness mask for the given DataFrame based on the specified
    resampling offset, completeness threshold, and tolerance for over-completeness.

    This function evaluates the completeness of timestamped data, ensuring that
    records match the expected durations defined by the `resample_offset`. It
    computes whether each resampled group of data satisfies the completeness
    constraints defined by the `completeness_threshold` and `atol`.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Input data containing a DatetimeIndex and associated values. The index must
        be a DatetimeIndex as the function relies on timestamp durations for
        computations.
    resample_offset : str
        Offset string defining resampling frequency. Examples include 'MS' (month
        start) or other Pandas-compatible offset strings.
    completeness_threshold : float, optional
        Dictionary containing the variable-specific minimum completeness ratio
        required to consider a time period as valid. Must contain a key 'default'
        used for variables not explicitly listed.
        Defaults to the dictionary `DEFAULT_COMPLETENESS_THRESHOLD`.
    atol : float, optional
        Absolute tolerance for over-completeness. Specifies an allowable margin by
        which completeness can exceed 1. Defaults to 1e-9.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing Boolean values, where True indicates that the data
        for the corresponding time period satisfies the completeness constraints,
        while False indicates the data is either under-complete or over-complete.
    """
    if resample_offset in ['MS', 'ME']:
        offset_timedelta = datetime.timedelta(days=30)
        # Increase tolerance for overcomplete values in monthly resampling
        # to handle months with 31 days.
        atol = 1/30 + atol
    else:
        offset_timedelta = pd.to_timedelta(resample_offset)

    index = data_frame.index
    assert isinstance(index, pd.DatetimeIndex)

    timestamp_durations = classify_timestamp_durations(index)
    timestamp_coverage = timestamp_durations / np.array(offset_timedelta)
    data_frame_is_valid = data_frame.notna()

    completeness = (
        data_frame_is_valid
        .mul(timestamp_coverage, axis=0)
        .resample(resample_offset).sum()
    )

    thresholds = pd.Series(
        {col: completeness_thresholds.get(col, completeness_thresholds["default"])
         for col in data_frame.columns}
    )

    is_under_complete = completeness.lt(thresholds, axis=1)
    is_over_complete = completeness.gt(1 + atol)
    completeness_mask = ~(is_under_complete | is_over_complete)
    return completeness_mask
