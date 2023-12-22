import logging
from datetime import datetime
from typing import Union, Optional

import pandas as pd

from pypromice.postprocess.csv2bufr import (
    find_positions,
    rolling_window,
    min_data_check,
)

logger = logging.getLogger(__name__)


def get_latest_data(
    df1: pd.DataFrame,
    stid: str,
    earliest_date: Union[datetime, str],
    lin_reg_time_limit: str,
    positions: Optional[dict],
) -> Optional[pd.Series]:
    # TODO: The data frames should be cropped with respect to a selected window
    # Check that the last valid index for all instantaneous values match
    # Note: we cannot always use the single most-recent timestamp in the dataframe
    # e.g. for 6-hr transmissions, *_u will have hourly data while *_i is nan
    # Need to check for last valid (non-nan) index instead
    last_valid_index = df1[earliest_date:][
        ["t_i", "p_i", "rh_i", "wspd_i", "wdir_i"]
    ].last_valid_index()

    if last_valid_index is None:
        logger.info(f"No recent instantaneous timestamps for {stid}!")
        return None
    else:
        # One or more of the values have valid indices (timestamps) within the allowed window.
        # The indices might not be equal.
        # We will throw this obset down the line, and there is a final min_data_check
        # to make sure we have minimum data requirements before writing to BUFR
        current_timestamp = last_valid_index
    logger.info(f"TIMESTAMP: {current_timestamp}")

    # Find positions
    # we only need to add positions to the BUFR file
    df1_limited, _ = find_positions(
        df1,
        stid,
        lin_reg_time_limit,
        current_timestamp,
        positions,
    )

    # Apply smoothing to z_boom_u
    # require at least 2 hourly obs? Sometimes seeing once/day data for z_boom_u
    df1_limited = rolling_window(df1_limited, "z_boom_u", "72H", 2, 1)

    # limit to single most recent valid row (convert to series)
    s1_current = df1_limited.loc[current_timestamp]
    s1_current["stid"] = stid

    # Check that we have minimum required valid data
    min_data_wx_result, min_data_pos_result = min_data_check(s1_current, stid)
    if min_data_wx_result is False:
        logger.warning(f"Failed min data wx {stid}")
        return None
    elif min_data_pos_result is False:
        logger.warning(f"Failed min data pos {stid}")
        return None

    return s1_current
