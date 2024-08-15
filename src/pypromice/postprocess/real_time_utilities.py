"""
Utility functions for processing real time / instantaneous AWS data.

This includes:
* Select latest data
* Noise filtering data

"""
import logging
from typing import Optional, Collection

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

__all__ = ["get_latest_data"]


logger = logging.getLogger(__name__)


def get_latest_data(
    df: pd.DataFrame,
    lin_reg_time_limit: str,
    vars_to_skip: Optional[Collection[str]] = None,
) -> Optional[pd.Series]:
    """
    Determine instantaneous values for the latest valid timestamp in the input dataframe

    * A valid timestamp is a timestamp with relevant instantaneous variables. See source code.
    * Location smoothing: Fit a linear regression model on gps coordinate over the period lin_reg_time_limit to determine latest values.
    * z_boom: Apply rolling window median filter smooth data

    The output series contains the same variables as the input dataframe plus smoothed variables:

    * gps_lat_fit
    * gps_lon_fit
    * gps_alt_fit
    * z_boom_u_smooth

    Parameters
    ----------
    df
        Input AWS l3 dataframe
    lin_reg_time_limit
        Previous time to limit dataframe before applying linear regression.

    Returns
    -------
    pd.Series with the latest data.

    """
    # TODO: The data frames should be cropped with respect to a selected window
    # Check that the last valid index for all instantaneous values match
    # Note: we cannot always use the single most-recent timestamp in the dataframe
    # e.g. for 6-hr transmissions, *_u will have hourly data while *_i is nan
    # Need to check for last valid (non-nan) index instead
    last_valid_index = df[["t_i", "p_i", "rh_i", "wspd_i", "wdir_i"]].last_valid_index()
    if last_valid_index is None:
        return None
    logger.info(f"TIMESTAMP: {last_valid_index}")

    # Find positions
    # we only need to add positions to the BUFR file
    df_limited = find_positions(
        df,
        lin_reg_time_limit,
    )

    if last_valid_index not in df_limited.index:
        logger.info("No valid data limited period")
        return None

    # Apply smoothing to z_boom_u
    # require at least 2 hourly obs? Sometimes seeing once/day data for z_boom_u
    df_limited = rolling_window(df_limited, "z_boom_u", "72h", 2, 3)
    
    # limit to single most recent valid row (convert to series)
    s_current = df_limited.loc[last_valid_index]

    if vars_to_skip is not None:
        s_current = filter_skipped_variables(s_current, vars_to_skip)

    return s_current


def filter_skipped_variables(
    row: pd.Series, vars_to_skip: Collection[str]
) -> pd.Series:
    """
    Mutate input series by setting var_to_skip to np.nan

    Parameters
    ----------
    row
    vars_to_skip
        List of variable names to be skipped

    Returns
    -------
    Input series

    """
    vars_to_skip = set(row.keys()) & set(vars_to_skip)
    for var_key in vars_to_skip:
        row[var_key] = np.nan
        logger.info("----> Skipping var: {}".format(var_key))
    return row


def rolling_window(df, column, window, min_periods, decimals) -> pd.DataFrame:
    """Apply a rolling window (smoothing) to the input column

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
    """
    df["{}_smooth".format(column)] = (
        df[column]
        .rolling(
            window,
            min_periods=min_periods,
            center=True,  # set the window labels as the center of the window
            closed="both",  # no points in the window are excluded (first or last)
        )
        .median()
        .round(decimals=decimals)
    )  # could also round to whole meters (decimals=0)
    return df


def find_positions(df, time_limit):
    """Driver function to run linear_fit() and set valid lat, lon, and alt
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
        (e.g. '91d')

    Returns
    -------
    df_limited : pandas dataframe
        Dataframe limited to time_limit, and including position data
    positions : dict
        Modified dict storing most-recent station positions.
    """
    logger.info("finding positions")
    time_delta = pd.Timedelta(time_limit)
    last_index = df.index.max()
    last_mask = df.index > last_index - time_delta
    df_limited = df.loc[last_mask].copy()

    logger.info(f"last transmission: {df_limited.index.max()}")

    # Extrapolate recommended for altitude, optional for lat and lon.
    df_limited, lat_valid = linear_fit(df_limited, "gps_lat", 7)
    df_limited, lon_valid = linear_fit(df_limited, "gps_lon", 7)
    df_limited, alt_valid = linear_fit(df_limited, "gps_alt", 4)

    # If we have no valid lat, lon or alt data in the df_limited window, then interpolate
    # using full tx dataset.
    check_valid = {"gps_lat": lat_valid, "gps_lon": lon_valid, "gps_alt": alt_valid}
    check_valid_again = {}
    for k, v in check_valid.items():
        if v is False:
            logger.info(f"----> Using full history for linear extrapolation: {k}")
            logger.info(f"first transmission: {df.index.min()}")
            if k == "gps_alt":
                df, valid = linear_fit(df, k, 2)
            else:
                df, valid = linear_fit(df, k, 7)
            check_valid_again[k] = valid
            if check_valid_again[k] is True:
                df_limited[f"{k}_fit"] = df.loc[df_limited.index, f"{k}_fit"]
            else:
                logger.info(f"----> No data exists for {k}. Stubbing out with NaN.")
                df_limited[f"{k}_fit"] = pd.Series(np.nan, index=df_limited.index)

    return df_limited


def linear_fit(df, column, decimals):
    """Apply a linear regression to the input column

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
    """
    # print('=========== linear_fit ===========')
    pos_valid = True
    if column in df:
        df_dropna = df[
            df[column].notna()
        ]  # limit to only non-nan for the target column
        # if len(df_dropna[column].index.normalize().unique()) >= 10: # must have at least 10 unique days
        if (
            len(df_dropna[column]) >= 15
        ):  # must have at least 15 data points (could be hourly or daily)
            # Get datetime x values into epoch sec integers
            x_epoch = df_dropna.index.values.astype(np.int64) // 10**9
            x = x_epoch.reshape(-1, 1)
            y = df_dropna[column].values  # can also reshape this, but not necessary
            model = LinearRegression().fit(x, y)

            # Adding prediction back to original df
            x_all = df.index.values.astype(np.int64) // 10**9
            df["{}_fit".format(column)] = model.predict(x_all.reshape(-1, 1)).round(
                decimals=decimals
            )

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
            logger.warning("----> Insufficient {} data!".format(column))
            pos_valid = False
    else:
        logger.warning("----> {} not found in dataframe!".format(column))
        pass
    return df, pos_valid
