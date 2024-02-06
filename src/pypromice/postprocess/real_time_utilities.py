import logging
from datetime import datetime
from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pypromice.postprocess.wmo_config import positions_update_timestamp_only

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
    last_valid_index = df1[
        ["t_i", "p_i", "rh_i", "wspd_i", "wdir_i"]
    ].last_valid_index()

    if last_valid_index is None or last_valid_index < earliest_date:
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
    s1_current = (
        df1_limited
        # Forward fill nan values to handle variables with nan values at current_timestamp.
        # Only consider values from the earliest date
        .loc[earliest_date:].ffill()
        .loc[current_timestamp]
    )
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


def rolling_window(df, column, window, min_periods, decimals) -> pd.DataFrame:
    '''Apply a rolling window (smoothing) to the input column

    Parameters
    ----------
    df : pandas.Dataframe
        datetime-indexed df
    column : str
        The target column for applying rolling window
    window : str
        Window size (e.g. '24H' or 30D')
    min_periods : int
        Minimum number of observations in window required to have a value;
        otherwise, result is np.nan.
    decimals : int
        How many decimal places to round the output smoothed values

    Returns
    -------
    df : pandas.Dataframe
        The original input df, with added column for the smoothed values
    '''
    df['{}_smooth'.format(column)] = df[column].rolling(
        window,
        min_periods=min_periods,
        center=True,  # set the window labels as the center of the window
        closed='both'  # no points in the window are excluded (first or last)
    ).median().round(decimals=decimals)  # could also round to whole meters (decimals=0)
    return df


def find_positions(df, stid, time_limit, current_timestamp=None, positions=None):
    ''' Driver function to run linear_fit() and set valid lat, lon, and alt
    to df_limited, which is then used to set position data in BUFR.
    If 'positions' is not None (must pass --positions arg), we also write to
    the positions dict which will be written to AWS_latest_locations.csv for
    all stations (whether processed or skipped)

    Parameters
    ----------
    df : pandas dataframe
        The full tx dataframe
    stid : str
        The station ID, such as NUK_L
    time_limit : str
        Previous time to limit dataframe before applying linear regression.
        (e.g. '3M')
    current_timestamp : datetime64 time
        The timestamp for the most recent valid instantaneous data
    positions : dict, or None
        Dict storing current station positions. If present, we are writing
        positions to file.

    Returns
    -------
    df_limited : pandas dataframe
        Dataframe limited to time_limit, and including position data
    positions : dict
        Modified dict storing most-recent station positions.
    '''
    if stid in positions_update_timestamp_only:
        # we don't have a position-associated timestamp, just use the most recent transmission.
        # e.g. KAN_B (does not transmit position, and currently skipped because does not transmit
        # instantaneous obs). If KAN_B ever submits inst data (but not position) we will need to use
        # the config-seeded position coordinates to set positions here in df_limited.
        positions[stid]['timestamp'] = df.index.max()
        df_limited = df  # just to return something
    else:
        logger.info(f'finding positions for {stid}')
        df_limited = df.last(time_limit).copy()
        logger.info(f'last transmission: {df_limited.index.max()}')

        # Extrapolate recommended for altitude, optional for lat and lon.
        df_limited, lat_valid = linear_fit(df_limited, 'gps_lat', 6, stid)
        df_limited, lon_valid = linear_fit(df_limited, 'gps_lon', 6, stid)
        df_limited, alt_valid = linear_fit(df_limited, 'gps_alt', 1, stid)

        # If we have no valid lat, lon or alt data in the df_limited window, then interpolate
        # using full tx dataset.
        check_valid = {'gps_lat': lat_valid, 'gps_lon': lon_valid, 'gps_alt': alt_valid}
        check_valid_again = {}
        for k, v in check_valid.items():
            if v is False:
                logger.info(f'----> Using full history for linear extrapolation: {k}')
                logger.info(f'first transmission: {df.index.min()}')
                if k == 'gps_alt':
                    df, valid = linear_fit(df, k, 1, stid)
                else:
                    df, valid = linear_fit(df, k, 6, stid)
                check_valid_again[k] = valid
                if check_valid_again[k] is True:
                    df_limited[f'{k}_fit'] = df.last(time_limit)[f'{k}_fit']
                else:
                    logger.info(f'----> No data exists for {k}. Stubbing out with NaN.')
                    df_limited[f'{k}_fit'] = pd.Series(np.nan, index=df.last(time_limit).index)

        # SET POSITIONS FOR CSV FILE
        if positions is not None:
            if stid not in positions:
                positions[stid] = dict()
            if current_timestamp is None:
                # This is old data (> 2 days), not submitting to DMI, but writing to positions csv
                # Find the most recent row that has valid lat, lon and alt
                last_valid_timestamp = df_limited[
                    ['gps_lon_fit', 'gps_lat_fit', 'gps_alt_fit']].dropna().last_valid_index()
                if last_valid_timestamp is None:
                    # we are likely missing gps_alt_fit
                    last_valid_timestamp = df_limited[['gps_lon_fit', 'gps_lat_fit']].dropna().last_valid_index()
                    if last_valid_timestamp is None:
                        # last ditch effort
                        last_valid_timestamp = df_limited.index.max()
                s = df_limited.loc[last_valid_timestamp]
            else:
                s = df_limited.loc[current_timestamp]
            logger.info(f'writing positions for {stid}')
            pos_strings = ['lat', 'lon', 'alt']
            for p in pos_strings:
                if (f'gps_{p}_fit' in s) and (pd.isna(s[f'gps_{p}_fit']) is False):
                    positions[stid][p] = s[f'gps_{p}_fit']
            # Add timestamp
            positions[stid]['timestamp'] = s.name

    return df_limited, positions


def min_data_check(s, stid):
    '''Check that we have minimum required fields to proceed with writing to BUFR
    For wx vars, we currently require both air temp and pressure to be non-NaN.
    If you know a specific var is reporting bad data, you can ignore just that var
    using the vars_to_skip dict in wmo_config.

    Parameters
    ----------
    s : pandas series
        The current obset we are working with (for BUFR submission)
    stid : str
        The station ID, such as NUK_L

    Returns
    -------
    min_data_wx_result : bool
        True (default), the test for min wx data passed. False, the test failed.
    min_data_pos_result : bool
        True (default), the test for min position data passed. False, the test failed.
    '''
    min_data_wx_result = True
    min_data_pos_result = True

    # Can use pd.isna() or math.isnan() below...

    # Always require valid air temp and valid pressure (both must be non-nan)
    # if (pd.isna(s['t_i']) is False) and (pd.isna(s['p_i']) is False):
    #     pass
    # else:
    #     print('----> Failed min_data_check for air temp and pressure!')
    #     min_data_wx_result = False

    # If both air temp and pressure are nan, do not submit.
    # This will allow the case of having only one or the other.
    if (pd.isna(s['t_i']) is True) and (pd.isna(s['p_i']) is True):
        logger.warning('----> Failed min_data_check for air temp and pressure!')
        min_data_wx_result = False

    # Missing just elevation OK
    # if (pd.isna(s['gps_lat_fit']) is False) and (pd.isna(s['gps_lon_fit']) is False):
    #     pass
    # Require all three: lat, lon, elev
    if ((pd.isna(s['gps_lat_fit']) is False) and
            (pd.isna(s['gps_lon_fit']) is False) and
            (pd.isna(s['gps_alt_fit']) is False)):
        pass
    else:
        logger.warning('----> Failed min_data_check for position!')
        min_data_pos_result = False

    return min_data_wx_result, min_data_pos_result


def linear_fit(df, column, decimals, stid):
    '''Apply a linear regression to the input column

    Linear regression is following:
    https://realpython.com/linear-regression-in-python/#simple-linear-regression-with-scikit-learn

    Parameters
    ----------
    df : pandas.Dataframe
        datetime-indexed df, limited to desired time length for linear fit
    column : str
        The target column for applying linear fit
    decimals : int
        How many decimals to round the output fit values
    stid : str
        The station ID to be processed. e.g. 'KPC_U'
    extrapolate : boolean
        If False (default), only apply linear fit to timestamps with valid data
        If True, then extrapolate positions based on linear fit model

    Returns
    -------
    df : pandas.Dataframe
        The original input df, with added column for the linear regression values
    pos_valid : boolean
        If True (default), sufficient valid data found in recent (limited) data.
        If False, we need to return this status to find_positions and use full station history instead.
    '''
    # print('=========== linear_fit ===========')
    pos_valid = True
    if column in df:
        df_dropna = df[df[column].notna()]  # limit to only non-nan for the target column
        # if len(df_dropna[column].index.normalize().unique()) >= 10: # must have at least 10 unique days
        if len(df_dropna[column]) >= 15:  # must have at least 15 data points (could be hourly or daily)
            # Get datetime x values into epoch sec integers
            x_epoch = df_dropna.index.values.astype(np.int64) // 10 ** 9
            x = x_epoch.reshape(-1, 1)
            y = df_dropna[column].values  # can also reshape this, but not necessary
            model = LinearRegression().fit(x, y)

            # Adding prediction back to original df
            x_all = df.index.values.astype(np.int64) // 10 ** 9
            df['{}_fit'.format(column)] = model.predict(x_all.reshape(-1, 1)).round(decimals=decimals)

            # Plot data if desired
            # if stid == 'LYN_T':
            #     if (column == 'gps_lat') or (column == 'gps_lon') or (column == 'gps_alt'):
            #         import matplotlib.pyplot as plt
            #         plt.figure()
            #         df_dropna[column].plot(marker='o',ls='None')
            #         df['{}_fit'.format(column)].plot(marker='o', ls='None', color='red')
            #         plt.title('{} {}'.format(stid, column))
            #         plt.xlim(df.index.min(),df.index.max())
            #         plt.show()
        else:
            # Do not have 10 days of valid data, or all data is NaN.
            logger.warning('----> Insufficient {} data for {}!'.format(column, stid))
            pos_valid = False
    else:
        logger.warning('----> {} not found in dataframe!'.format(column))
        pass
    return df, pos_valid
