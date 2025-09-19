#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:58:39 2024

@author: pho
"""
import logging
import numpy as np
import pandas as pd
import xarray as xr
from pypromice.core.variables.wind import calculate_directional_wind_speed
logger = logging.getLogger(__name__)

def resample_dataset(ds_h, t, completeness_threshold=0.8):
    '''Resample L2 AWS data, e.g. hourly to daily average. This uses pandas
    DataFrame resampling at the moment as a work-around to the xarray Dataset
    resampling. As stated, xarray resampling is a lengthy process that takes
    ~2-3 minutes per operation: ds_d = ds_h.resample({'time':"1D"}).mean()
    This has now been fixed, so needs implementing:
    https://github.com/pydata/xarray/issues/4498#event-6610799698

    Parameters
    ----------
    ds_h : xarray.Dataset
        L3 AWS dataset either at 10 min (for raw data) or hourly (for tx data)
    t : str
        Resample factor( "60min", "1D" or "MS"), same variable definition as in
        pandas.DataFrame.resample()
    completeness_threshold : float
        Lower limit of completness of an hourly/daily/monthly aggregate (nr of
        samples in aggregate / expected nr of samples). Aggregates below that
        limit are replaced by NaNs.

    Returns
    -------
    ds_d : xarray.Dataset
        L3 AWS dataset resampled to the frequency defined by t
    '''
    # Convert dataset to DataFrame
    df_h = ds_h.to_dataframe()

    # Identify non-numeric columns
    non_numeric_cols = df_h.select_dtypes(exclude=['number']).columns

    # Log a warning and drop non-numeric columns
    if len(non_numeric_cols) > 0:
        for col in non_numeric_cols:
            unique_values = df_h[col].unique()
            logger.warning(f"Dropping column '{col}' because it is of type '{df_h[col].dtype}' and contains unique values: {unique_values}")

        df_h = df_h.drop(columns=non_numeric_cols)
    # Resample the DataFrame
    df_resampled = df_h.resample(t).mean()

    # exception for precip_u and precip_l which are accumulated with some reset
    # we therefore take the positive increments in the higher resolution data (like 10 min)
    # and sum them into the aggregated data (hourly/daily/monthly).
    # This makes the assumption of 0 precipitation during data gaps.
    for var in ['precip_u', 'precip_l']:
        if var in df_h.columns:
            df_resampled[var] = df_h[var].diff().clip(lower=0).resample(t).sum()

    # First attempt of completness filter
    df_resampled = apply_completeness_filters(df_resampled,
                                              df_h,
                                              t,
                                              time_thresh=completeness_threshold,
                                              value_thresh=completeness_threshold,
                                              )

    # taking the 10 min data and using it as instantaneous values:
    is_10_minutes_timestamp = (ds_h.time.diff(dim='time') / np.timedelta64(1, 's') == 600)
    if (t == '60min') and is_10_minutes_timestamp.any():
        cols_to_update = ['p_i', 't_i', 'rh_i', 'rh_i_wrt_ice_or_water', 'wspd_i', 'wdir_i','wspd_x_i','wspd_y_i']
        cols_origin = ['p_u', 't_u', 'rh_u', 'rh_u_wrt_ice_or_water', 'wspd_u', 'wdir_u','wspd_x_u','wspd_y_u']
        timestamp_10min = ds_h.time.where(is_10_minutes_timestamp, drop=True).to_index()
        timestamp_round_hour = df_resampled.index
        timestamp_to_update = timestamp_round_hour.intersection(timestamp_10min)

        for col, col_org in zip(cols_to_update, cols_origin):
            if col not in df_resampled.columns:
                df_resampled[col] = np.nan
            else:
                # if there are already instantaneous values in the dataset
                # we want to keep them as they are
                # removing timestamps where there is already t_i filled from a TX file
                missing_instantaneous = ds_h.reindex(time=timestamp_to_update)[col].isnull()
                timestamp_to_update = timestamp_to_update[missing_instantaneous]
            df_resampled.loc[timestamp_to_update, col] = ds_h.reindex(
                time= timestamp_to_update
                )[col_org].values
            if col == 'p_i':
                df_resampled.loc[timestamp_to_update, col] = df_resampled.loc[timestamp_to_update, col].values-1000

    # recalculating wind direction from averaged directional wind speeds
    for var in ['wdir_u','wdir_l']:
        boom = var.split('_')[1]
        if var in df_resampled.columns:
            if ('wspd_x_'+boom in df_resampled.columns) & ('wspd_y_'+boom in df_resampled.columns):
                df_resampled[var] = _calcWindDir(df_resampled['wspd_x_'+boom], df_resampled['wspd_y_'+boom])
            else:
                logger.info(var+' in dataframe but not wspd_x_'+boom+' nor wspd_y_'+boom+', recalculating them')
                ds_h['wspd_x_'+boom], ds_h['wspd_y_'+boom] = calculate_directional_wind_speed(ds_h['wspd_'+boom], ds_h['wdir_'+boom])
                df_resampled[['wspd_x_'+boom, 'wspd_y_'+boom]] = ds_h[['wspd_x_'+boom, 'wspd_y_'+boom]].to_dataframe().resample(t).mean()
                df_resampled[var] = _calcWindDir(df_resampled['wspd_x_'+boom], df_resampled['wspd_y_'+boom])

    # recalculating relative humidity from average vapour pressure and average
    # saturation vapor pressure
    for var in ['rh_u','rh_l']:
        lvl = var.split('_')[1]
        if var in df_resampled.columns:
            if ('t_'+lvl in ds_h.keys()):
                es_wtr, es_cor = calculateSaturationVaporPressure(ds_h['t_'+lvl])
                p_vap = ds_h[var] / 100 * es_wtr

                df_resampled[var] = (p_vap.to_series().resample(t).mean() \
                           / es_wtr.to_series().resample(t).mean())*100
                if var+'_wrt_ice_or_water' in df_resampled.keys():
                    df_resampled[var+'_wrt_ice_or_water'] = (p_vap.to_series().resample(t).mean() \
                               / es_cor.to_series().resample(t).mean())*100

    # passing each variable attribute to the ressample dataset
    vals = []
    for c in df_resampled.columns:
        if c in ds_h.data_vars:
            vals.append(xr.DataArray(
                data=df_resampled[c], dims=['time'],
               coords={'time':df_resampled.index}, attrs=ds_h[c].attrs))
        else:
            vals.append(xr.DataArray(
                data=df_resampled[c], dims=['time'],
               coords={'time':df_resampled.index}, attrs=None))

    ds_resampled = xr.Dataset(dict(zip(df_resampled.columns,vals)), attrs=ds_h.attrs)

    return ds_resampled


def compute_dt(index: pd.DatetimeIndex) -> pd.Series:
    """
    Compute timestep durations (dt) in seconds between consecutive samples.
    Only {600, 3600, 86400} are allowed; others set to NaN and backfilled.

    Parameters
    ----------
    index : pd.DatetimeIndex

    Returns
    -------
    pd.Series
        dt in seconds aligned to `index` (first value backfilled).
    """
    dt = index.to_series().diff().dt.total_seconds().round()
    allowed = {600.0, 3600.0, 86400.0}
    dt = dt.where(dt.isin(allowed), np.nan)
    return dt.bfill()


def compute_time_weights(dt: pd.Series, t: str) -> pd.Series:
    """
    Compute per-sample time weights (contribution to one target bin) from dt.

    Parameters
    ----------
    dt : pd.DatetimeIndex
        Timestep durations in seconds
    t : str
        Target resampling frequency ("60min", "1D", "MS").

    Returns
    -------
    pd.Series
        Weights in [0, 1] aligned to `index`.
    """
    w = pd.Series(0.0, index=dt.index)
    if t == "60min":
        w[dt == 600] = 1/6
        w[dt == 3600] = 1
        w[dt == 86400] = 0
    elif t == "1D":
        w[dt == 600] = 1/(6*24)
        w[dt == 3600] = 1/24
        w[dt == 86400] = 1
    elif t == "MS":
        w[dt == 600] = 1/(6*24*30)
        w[dt == 3600] = 1/(24*30)
        w[dt == 86400] = 1/30
    else:
        w[:] = 1.0
    return w


def compute_time_completeness(w: pd.Series, t: str) -> pd.Series:
    """
    Compute time-based completeness per resampled bin from per-sample weights.

    Parameters
    ----------
    w : pd.DatetimeIndex
        Weights in [0, 1] aligned to `index`.
    t : str
        Target resampling frequency.

    Returns
    -------
    pd.Series
        Completeness (0..1) per bin indexed by the resampled DateTimeIndex.
    """
    return w.resample(t).sum().clip(upper=1.0).fillna(0.0)


def compute_value_completeness(df_h: pd.DataFrame, w: pd.Series, t: str) -> pd.DataFrame:
    """
    Compute per-variable value completeness as weighted non-NaN fraction per bin.

    Parameters
    ----------
    df_h : pd.DataFrame
        High-resolution data with DateTimeIndex.
    w : pd.DatetimeIndex
        Weights in [0, 1] aligned to `index`.
    t : str
        Target resampling frequency.

    Returns
    -------
    pd.DataFrame
        Fraction (0..1) of expected weighted samples observed per variable and bin.
    """
    expected = w.resample(t).sum().clip(upper=1.0).where(lambda s: s > 0, np.nan)
    observed_equiv = (~df_h.isna()).astype(float).mul(w, axis=0).resample(t).sum()
    return observed_equiv.div(expected, axis=0).fillna(0.0).clip(0.0, 1.0)


def apply_completeness_filters(
    df_resampled: pd.DataFrame,
    df_h: pd.DataFrame,
    t: str,
    time_thresh: float = 0.8,
    value_thresh: float = 0.8,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Apply time- and value-based completeness filters to resampled data.

    Parameters
    ----------
    df_resampled : pd.DataFrame
        Resampled aggregates to be filtered.
    df_h : pd.DataFrame
        High-resolution source data (for completeness computation).
    t : str
        Target resampling frequency.
    time_thresh : float, optional
        Minimum time completeness threshold, by default 0.8.
    value_thresh : float, optional
        Minimum value completeness threshold, by default 0.8.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        (filtered_df, time_completeness, value_completeness).
    """
    dt = compute_dt(df_h.index)
    w = compute_time_weights(dt, t)
    tc = compute_time_completeness(w, t)
    vc = compute_value_completeness(df_h, w, t)
    out = df_resampled.copy()
    out.loc[tc.reindex(out.index, fill_value=0) < time_thresh] = np.nan
    out[vc.reindex(index=out.index, columns=out.columns, fill_value=0) < value_thresh] = np.nan
    return out


def calculateSaturationVaporPressure(t, T_0=273.15, T_100=373.15, es_0=6.1071,
                                     es_100=1013.246, eps=0.622):
    '''Calculate specific humidity

    Parameters
    ----------
    T_0 : float
        Steam point temperature. Default is 273.15.
    T_100 : float
        Steam point temperature in Kelvin
    t : xarray.DataArray
        Air temperature
    es_0 : float
        Saturation vapour pressure at the melting point (hPa)
    es_100 : float
        Saturation vapour pressure at steam point temperature (hPa)
    Returns
    -------
    xarray.DataArray
        Saturation vapour pressure with regard to water above 0 C (hPa)
    xarray.DataArray
        Saturation vapour pressure where subfreezing timestamps are with regards to ice (hPa)
    '''
    # Saturation vapour pressure above 0 C (hPa)
    es_wtr = 10**(-7.90298 * (T_100 / (t + T_0) - 1) + 5.02808 * np.log10(T_100 / (t + T_0))
                  - 1.3816E-7 * (10**(11.344 * (1 - (t + T_0) / T_100)) - 1)
                  + 8.1328E-3 * (10**(-3.49149 * (T_100 / (t + T_0) -1)) - 1) + np.log10(es_100))

    # Saturation vapour pressure below 0 C (hPa)
    es_ice = 10**(-9.09718 * (T_0 / (t + T_0) - 1) - 3.56654
                  * np.log10(T_0 / (t + T_0)) + 0.876793
                  * (1 - (t + T_0) / T_0)
                  + np.log10(es_0))

    # Saturation vapour pressure (hPa)
    es_cor = xr.where(t < 0, es_ice, es_wtr)

    return es_wtr, es_cor

def _calcWindDir(wspd_x, wspd_y):
    '''Calculate wind direction in degrees

    Parameters
    ----------
    wspd_x : xarray.DataArray
        Wind speed in X direction
    wspd_y : xarray.DataArray
        Wind speed in Y direction

    Returns
    -------
    wdir : xarray.DataArray
        Wind direction'''
    deg2rad = np.pi / 180
    rad2deg = 1 / deg2rad
    wdir = np.arctan2(wspd_x, wspd_y) * rad2deg
    wdir = (wdir + 360) % 360
    return wdir
