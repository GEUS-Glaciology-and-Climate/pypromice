#!/usr/bin/env python
"""
AWS Level 0 (L0) to Level 1 (L1) data processing
"""
__all__ = ["toL1"]

import numpy as np
import pandas as pd
import xarray as xr
import re, logging
logger = logging.getLogger(__name__)

from pypromice.core.variables.pressure_transducer_depth import correct_and_calculate_depth
from pypromice.core.qc.value_clipping import clip_values
from pypromice.core.variables import (wind, 
                                      air_temperature, 
                                      gps, 
                                      radiation,
                                      station_boom_height,
                                      station_pose,
                                      pressure_transducer_depth)


def toL1(L0: xr.DataArray,
         vars_df: pd.DataFrame
) -> xr.DataArray:
    """Process one Level 0 (L0) dataset to a
    Level 1 (L1) dataset

    Parameters
    ----------
    L0 : xarray.Dataset
        Level 0 dataset
    vars_df : pd.DataFrame
        Metadata dataframe

    Returns
    -------
    ds : xarray.Dataset
        Level 1 dataset
    """
    assert(type(L0) == xr.Dataset)
    ds = L0

    for l in list(ds.keys()):
        if l not in ['time', 'msg_i', 'gps_lat', 'gps_lon', 'gps_alt', 'gps_time']:
            ds[l] = _reformat_array(ds[l])

    # The following drops duplicate datetime indices. Needs to run before _addTimeShift!
    # We can optionally also drop duplicates within _addTimeShift using pandas duplicated,
    # but retaining the following code instead to preserve previous methods. PJW
    _, index = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index)

    # If we do not want to shift hourly average values back -1 hr, then comment the following line.
    ds = add_time_shift(ds, vars_df)

    # Convert radiation from engineering to physical units
    # TODO add metadata to indicate whether radiometer values are corrected with calibration values or not
    if hasattr(ds, 'dsr_eng_coef'):
        ds['dsr'] = radiation.convert_sr(ds['dsr'],
                                         ds.attrs['dsr_eng_coef'])
    if hasattr(ds, 'usr_eng_coef'):
        ds['usr'] = radiation.convert_sr(ds['usr'],
                                         ds.attrs['usr_eng_coef'])
    if hasattr(ds, 'dlr_eng_coef'):
        ds['dlr'] = radiation.convert_lr(ds['dlr'],
                                         ds['t_rad'],
                                         ds.attrs['dlr_eng_coef'])
    if hasattr(ds, 'ulr_eng_coef'):
        ds['ulr'] = radiation.convert_lr(ds['ulr'],
                                         ds['t_rad'],
                                         ds.attrs['ulr_eng_coef'])

    # Reformat boom height
    ds['z_boom_u'] = _reformat_array(ds['z_boom_u'])

    # Adjust sonic ranger readings for sensitivity to air temperature (interpolated)
    tu_lo = vars_df.loc["t_u","lo"]
    tu_hi = vars_df.loc["t_u","hi"]
    ds["t_u_interp"] = air_temperature.clip_and_interpolate(ds["t_u"], tu_lo, tu_hi)
    ds["z_boom_cor_u"] = station_boom_height.adjust(ds["z_boom_u"], ds["t_u_interp"])

    # Decode and convert GPS positions
    ds["gps_lat"], ds["gps_lon"], ds["gps_time"] = gps.decode_and_convert(ds["gps_lat"],
                                                                          ds["gps_lon"],
                                                                          ds["gps_time"],
                                                                          ds.attrs["latitude"],
                                                                          ds.attrs["longitude"])

    # Reformat GPS information
    for l in ['gps_lat', 'gps_lon', 'gps_alt','gps_time']:
        ds[l] = _reformat_array(ds[l])

    # Convert tilt voltage to degrees
    if hasattr(ds, "logger_type"):
        if ds.attrs["logger_type"].upper() == "CR1000":
            ds["tilt_x"] = station_pose.convert_and_filter_tilt(ds["tilt_x"])
            ds["tilt_y"] = station_pose.convert_and_filter_tilt(ds["tilt_y"])

    # Apply tilt factor (e.g. -1 will invert tilt angle)
    if hasattr(ds, "tilt_y_factor"):
        ds["tilt_y"] = station_pose.apply_tilt_factor(ds["tilt_y"],
                                                      ds.attrs["tilt_y_factor"])

    # Smooth station tilt
    # Note that this should be OK for CR1000 tx (data only every 6 hrs),
    # since we interpolate above in station_pose.convert_and_filter_tilt(). PJW
    # TODO smoothing should be changed to a fixed time window rather than based on sample steps. PHO
    # TODO a smoothing is performed here and at L1toL2 also. Is this needed? PHO
    ds["tilt_x"] = station_pose.smooth_tilt_with_moving_window(ds["tilt_x"])
    ds["tilt_y"] = station_pose.smooth_tilt_with_moving_window(ds["tilt_y"])

    # Apply wind factor if provided
    # This is in the case of an anemometer rotations improperly translated to wind speed by the logger program
    if hasattr(ds, 'wind_u_coef'):
        logger.info(f'Wind speed correction applied to wspd_u based on factor of {ds.attrs["wind_u_coef"]}')
        ds['wspd_u'] = wind.correct_wind_speed(ds['wspd_u'],
                                               ds.attrs['wind_u_coef'])
    if hasattr(ds, 'wind_l_coef'):
        logger.info(f'Wind speed correction applied to wspd_l based on factor of {ds.attrs["wind_l_coef"]}')
        ds['wspd_l'] = wind.correct_wind_speed(ds['wspd_l'],
                                               ds.attrs['wind_l_coef'])
    if hasattr(ds, 'wind_i_coef'):
        logger.info(f'Wind speed correction applied to wspd_i based on factor of {ds.attrs["wind_i_coef"]}')
        ds['wspd_i'] = wind.correct_wind_speed(ds['wspd_i'],
                                               ds.attrs['wind_i_coef'])
                                               
    # FRE has a special encoding/calibration for pressure, it is the only station that has attribute FRE_pressure_decoding = 1
    if hasattr(ds, 'FRE_pressure_decoding'):
        logger.info('Special decoding of air pressure')
        ds['p_u'] = (ds['p_u']+1000)*0.2 + 600


    # Handle cases where the bedrock attribute is incorrectly set
    if not 'bedrock' in ds.attrs:
        logger.warning('bedrock attribute is not set')
        ds.attrs['bedrock'] = False
    elif not isinstance(ds.attrs['bedrock'], bool):
        logger.warning(f'bedrock attribute is not boolean: {ds.attrs["bedrock"]}')
        ds.attrs['bedrock'] =  str(ds.attrs['bedrock']).lower() == 'true'
    is_bedrock = ds.attrs['bedrock']

    # Some bedrock stations (e.g. KAN_B) do not have tilt in L0 files
    # so we need to create them manually
    if is_bedrock:
        for var in ['tilt_x','tilt_y']:
            if var not in ds.data_vars:
                ds[var] = (('time'), np.full(ds['time'].size, np.nan))

        # WEG_B has a non-null z_pt even though it is a bedrock station
        if ~ds['z_pt'].isnull().all():
            ds['z_pt'] = (('time'), np.full(ds['time'].size, np.nan))
            logger.info('Warning: Non-null data for z_pt at a bedrock site')

    # Perform one-boom variable processing
    if ds.attrs["number_of_booms"]==1:
        if ~ds["z_pt"].isnull().all():

            # Adjust PTA fluid density and calculate depth
            if hasattr(ds, "pt_z_offset"):
                ds["z_pt"] = pressure_transducer_depth.apply_offset(ds["z_pt"],
                                                                    int(ds.attrs["pt_z_offset"]))

            ds["z_pt_cor"],ds["z_pt"]=pressure_transducer_depth.correct_and_calculate_depth(ds["z_pt"],
                                                                                            ds["p_u"],
                                                                                            ds.attrs["pt_antifreeze"],
                                                                                            ds.attrs["pt_z_factor"],
                                                                                            ds.attrs["pt_z_coef"],
                                                                                            ds.attrs["pt_z_p_coef"])

        # Adjust sonic ranger readings on stake for sensitivity to air temperature
        ds['z_stake'] = _reformat_array(ds['z_stake'])
        ds["z_stake_cor"] = station_boom_height.adjust(ds["z_stake"], ds["t_u_interp"])

    # Perform two-boom variable processing
    elif ds.attrs['number_of_booms']==2:

        # Reformat boom height
        ds['z_boom_l'] = _reformat_array(ds['z_boom_l'])

        # Adjust sonic ranger readings for sensitivity to air temperature (interpolated)
        tl_lo = vars_df.loc["t_l","lo"]
        tl_hi = vars_df.loc["t_l","hi"]
        ds["t_l_interp"] = air_temperature.clip_and_interpolate(ds["t_l"], tl_lo, tl_hi)
        ds["z_boom_cor_l"] = station_boom_height.adjust(ds["z_boom_l"], ds["t_l_interp"])

    # Clip values and remove redundant attribute information
    ds = clip_values(ds, vars_df)
    for key in ['hygroclip_t_offset', 'dsr_eng_coef', 'usr_eng_coef',
          'dlr_eng_coef', 'ulr_eng_coef', 'wind_u_coef','wind_l_coef',
          'wind_i_coef', 'pt_z_coef', 'pt_z_p_coef', 'pt_z_factor',
          'pt_antifreeze', 'boom_azimuth', 'nodata', 'conf', 'file']:
        ds.attrs.pop(key, None)

    # Return Level 1 dataset
    ds.attrs['level'] = 'L1'
    return ds

def add_time_shift(ds, vars_df):
    '''Shift times based on file format and logger type (shifting only hourly averaged values,
    and not instantaneous variables). For raw (10 min), all values are sampled instantaneously
    so do not shift. For STM (1 hour), values are averaged and assigned to end-of-hour by the
    logger, so shift by -1 hr. For TX (time frequency depends on v2 or v3) then time is shifted
    depending on logger type. We use the 'instantaneous_hourly' boolean from variables.csv to
    determine if a variable is considered instantaneous at hourly samples.

    This approach creates two separate sub-dataframes, one for hourly-averaged variables
    and another for instantaneous variables. The instantaneous dataframe should never be
    shifted. We apply shifting only to the hourly average dataframe, then concat the two
    dataframes back together.

    It is possible to use pandas merge or join instead of concat, there are equivalent methods
    in each. In this case, we use concat throughout.

    Fausto et al. 2021 specifies the convention of assigning hourly averages to start-of-hour,
    so we need to retain this unless clearly communicated to users.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to apply time shift to
    vars_df : pd.DataFrame
        Metadata dataframe

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset with shifted times
    '''
    df = ds.to_dataframe()
    # No need to drop duplicates here if performed prior to calling this function.
    # df = df[~df.index.duplicated(keep='first')] # drop duplicates, keep=first is arbitrary
    df['doy'] = df.index.dayofyear
    i_cols = [x for x in df.columns if x in vars_df.index and vars_df['instantaneous_hourly'][x] is True] # instantaneous only, list of columns
    df_i = df.filter(items=i_cols, axis=1) # instantaneous only dataframe
    df_a = df.drop(df_i.columns, axis=1) # hourly ave dataframe

    if ds.attrs['format'] == 'raw':
        # 10-minute data, no shifting
        df_out = df
    elif ds.attrs['format'] == 'STM':
        # hourly-averaged, non-transmitted
        # shift everything except instantaneous, any logger type
        df_a = df_a.shift(periods=-1, freq="h")
        df_out = pd.concat([df_a, df_i], axis=1) # different columns, same datetime indices
        df_out = df_out.sort_index()
    elif ds.attrs['format'] == 'TX':
        if ds.attrs['logger_type'] == 'CR1000X':
            # v3, data is hourly all year long
            # shift everything except instantaneous
            df_a = df_a.shift(periods=-1, freq="h")
            df_out = pd.concat([df_a, df_i], axis=1) # different columns, same datetime indices
            df_out = df_out.sort_index()
        elif ds.attrs['logger_type'] == 'CR1000':
            # v2, data is hourly (6-hr for instantaneous) for DOY 100-300, otherwise daily at 00 UTC
            # shift non-instantaneous hourly for DOY 100-300, else do not shift daily
            df_a_hourly = df_a.loc[(df_a['doy'] >= 100) & (df_a['doy'] <= 300)]
            # df_a_hourly = df_a.loc[df_a['doy'].between(100, 300, inclusive='both')] # equivalent to above
            df_a_daily_1 = df_a.loc[(df_a['doy'] < 100)]
            df_a_daily_2 = df_a.loc[(df_a['doy'] > 300)]

            # shift the hourly ave data
            df_a_hourly = df_a_hourly.shift(periods=-1, freq="h")

            # stitch everything back together
            df_concat_u = pd.concat([df_a_daily_1, df_a_daily_2, df_a_hourly], axis=0) # same columns, different datetime indices
            # It's now possible for df_concat_u to have duplicate datetime indices
            df_concat_u = df_concat_u[~df_concat_u.index.duplicated(keep='first')] # drop duplicates, keep=first is arbitrary

            df_out = pd.concat([df_concat_u, df_i], axis=1) # different columns, same datetime indices
            df_out = df_out.sort_index()

    # Back to xarray, and re-assign the original attrs
    df_out = df_out.drop('doy', axis=1)
    ds_out = df_out.to_xarray()
    ds_out = ds_out.assign_attrs(ds.attrs) # Dataset attrs
    for x in ds_out.data_vars: # variable-specific attrs
        ds_out[x].attrs = ds[x].attrs

    # equivalent to above:
    # vals = [xr.DataArray(data=df_out[c], dims=['time'], coords={'time':df_out.index}, attrs=ds[c].attrs) for c in df_out.columns]
    # ds_out = xr.Dataset(dict(zip(df_out.columns, vals)), attrs=ds.attrs)
    return ds_out


def _reformat_array(ds_arr):
    """Reformat DataArray values and attributes"""
    a = ds_arr.attrs
    ds_arr.values = pd.to_numeric(ds_arr, errors='coerce')
    ds_arr.attrs = a
    return ds_arr

#------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
