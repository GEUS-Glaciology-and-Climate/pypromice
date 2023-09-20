import numpy as np
import pandas as pd
import xarray as xr


def differenceQC(ds: xr.Dataset) -> xr.Dataset:
    '''

    Parameters
    ----------
    ds : xr.Dataset
         Level 1 datset

    Returns
    -------
    ds_out : xr.Dataset
            Level 1 dataset with difference outliers set to NaN
    '''

    # the differenceQC is not done on the Windspeed
    # Optionally examine flagged data by setting make_plots to True
    # This is best done by running aws.py directly and setting 'test_station'
    # Plots will be shown before and after flag removal for each var

    stid = ds.station_id
    df = ds.to_dataframe()  # Switch to pandas

    # Define threshold dict to hold limit values, and the difference values.
    # Limit values indicate how much a variable has to change to the previous value
    # diff_period is how many hours a value can stay the same without being set to NaN
    # * are used to calculate and define all limits, which are then applied to *_u, *_l and *_i

    var_threshold = {
        't': {'static_limit': 0.001, 'diff_period': 1},
        'p': {'static_limit': 0.0001, 'diff_period': 24},
        'rh': {'static_limit': 0.0001, 'diff_period': 24}
    }

    for k in var_threshold.keys():

        var_all = [k + '_u', k + '_l', k + '_i']  # apply to upper, lower boom, and instant
        static_limit = var_threshold[k]['static_limit']  # loading static limit
        diff_period = var_threshold[k]['diff_period']  # loading diff period

        for v in var_all:
            if v in df:
                outliers = find_stationary_values(df[v], diff_period, static_limit)
                df.loc[outliers.index, v] = np.nan  # setting outliers to NaN

    # Back to xarray, and re-assign the original attrs
    ds_out = df.to_xarray()
    ds_out = ds_out.assign_attrs(ds.attrs)  # Dataset attrs
    for x in ds_out.data_vars:  # variable-specific attrs
        ds_out[x].attrs = ds[x].attrs
    # equivalent to above:
    # vals = [xr.DataArray(data=df_out[c], dims=['time'], coords={'time':df_out.index}, attrs=ds[c].attrs) for c in df_out.columns]
    # ds_out = xr.Dataset(dict(zip(df_out.columns, vals)), attrs=ds.attrs)
    return ds_out


def find_stationary_values(
        data: pd.Series,
        diff_period: int,
        static_limit: float,
) -> pd.Series:
    diff = data.diff()
    diff.fillna(method='ffill', inplace=True)  # forward filling all NaNs!
    diff = np.array(diff)
    outliers_mask = np.zeros_like(diff, dtype=bool)
    for i, d in enumerate(diff):  # algorithm that ensures values can stay the same within the outliers_mask
        if i > (diff_period - 1):
            if sum(abs(diff[i - diff_period:i])) < static_limit:
                outliers_mask[i - diff_period:i] = True
    outliers = data[outliers_mask]  # finding outliers in dataframe
    return outliers
