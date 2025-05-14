#!/usr/bin/env python
"""
AWS Level 0 (L0) to Level 1 (L1) data processing
"""
import numpy as np
import pandas as pd
import xarray as xr
import re, logging
from pypromice.process.value_clipping import clip_values
from pypromice.process import radiation, pressure_depth, boom_height, tilt
logger = logging.getLogger(__name__)


def toL1(L0, vars_df, T_0=273.15, tilt_threshold=-100):
    '''Process one Level 0 (L0) product to Level 1

    Parameters
    ----------
    L0 : xarray.Dataset
        Level 0 dataset
    vars_df : pd.DataFrame
        Metadata dataframe
    T_0 : int
        Air temperature for sonic ranger adjustment
    tilt_threshold : int
        Tilt-o-meter threshold for valid measurements
        
    Returns
    -------
    ds : xarray.Dataset
        Level 1 dataset
    '''
    # Assert input type
    assert(type(L0) == xr.Dataset)

    # Reassign input
    ds = L0

    # Add Level 1 attribute
    ds.attrs['level'] = 'L1'

    # Reformat most variables
    for l in list(ds.keys()):
        if l not in ['time', 'msg_i', 'gps_lat', 'gps_lon', 'gps_alt', 'gps_time']:
            ds[l] = _reformatArray(ds[l])                                               # TODO reformatting to occur as it is loaded as L0? PHO.

    # ds['time_orig'] = ds['time'] # Not used

    # The following drops duplicate datetime indices. Needs to run before _addTimeShift!
    # We can optionally also drop duplicates within _addTimeShift using pandas duplicated,
    # but retaining the following code instead to preserve previous methods. PJW
    _, index = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index)

    # If we do not want to shift hourly average values back -1 hr, then comment the following line.
    ds = add_time_shift(ds, vars_df)

    # Correct radiation measurements with coefficients
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
    ds['z_boom_u'] = _reformat_array(ds['z_boom_u'])                            # TODO Is this needed as it should be done on line 45? PHO

    # Correct boom height for air temperature sensitivity
    ds['z_boom_u'] = boom_height.correct_with_temp_interp(ds['z_boom_u'],
                                                          ds['t_u'],
                                                          vars_df,
                                                          T_0)

    # Decode GPS information
    if ds['gps_lat'].dtype.kind == 'O':                                        # Decode and reformat GPS information
        if 'NH' in ds['gps_lat'].dropna(dim='time').values[1]:
            logger.info('Found NH in GPS string')
            ds['gps_lat'] = gps.decode(ds['gps_lat'])
            ds['gps_lon'] = gps.decode(ds['gps_lon'])
            ds['gps_alt'] = gps.decode(ds['gps_alt'])

        elif 'L' in ds['gps_lat'].dropna(dim='time').values[1]:
            logger.info('Found L in GPS string')
            ds['gps_lat'] = gps.decode(ds['gps_lat'])
            ds['gps_lon'] = gps.decode(ds['gps_lon'])
            ds['gps_alt'] = gps.decode(ds['gps_alt'])
            for l in ['gps_lat', 'gps_lon']:
                ds[l] = ds[l]/100000
        else:
            try:
                # TODO this is a work around specifically for L0 RAW processing for THU_U.
                #  There should be a way to make this slicker. PHO
                ds['gps_lat'] = gps.decode(ds['gps_lat'])
                ds['gps_lon'] = gps.decode(ds['gps_lon'])
                ds['gps_alt'] = gps.decode(ds['gps_alt'])
                logger.info('No GPS string type detected, but decoding was successfully completed')
            except:
                logger.info('Invalid GPS type {ds["gps_lat"].dtype} for decoding')

    # Reformat gps measurements
    for l in ['gps_lat', 'gps_lon', 'gps_alt','gps_time']:
        ds[l] = _reformat_array(ds[l])

    # Reformat gps measurements if originating latitude and longitude positions are provided
    if hasattr(ds, 'latitude') and hasattr(ds, 'longitude'):
        logger.info('GPS reformatted based on latitude {ds.attrs["latitude"]}, longitude {ds.attrs["longitude"]}')
        ds['gps_lat'] = gps.reformat(ds['gps_lat'],
                                     ds.attrs['latitude'])
        ds['gps_lon'] = gps.reformat(ds['gps_lon'],
                                     ds.attrs['longitude'])

    # Convert and filter tilt to degrees, based on logger type
    if hasattr(ds, 'logger_type'):
        if ds.attrs['logger_type'].upper() == 'CR1000':
            logger.info('{ds.attrs["logger_type"]} tilt filtered')
            ds['tilt_x'] = tilt.filter(ds['tilt_x'],
                                       tilt_threshold)
            ds['tilt_y'] = tilt.filter(ds['tilt_y'],
                                       tilt_threshold)

    # Apply tilt factor if provided (e.g. -1 will invert tilt angle)
    if hasattr(ds, 'tilt_y_factor'):
        logger.info('Tilt correction applied based on factor of {ds.attrs["tilt_y_factor"]}')
        ds['tilt_y'] = tilt.apply_correction(ds['tilt_y'],
                                             ds.attrs['tilt_y_factor'])

    # Smooth tilt values
    # Note that this should be OK for CR1000 tx (data only every 6 hrs),
    # since we interpolate above in _getTiltDegrees. PJW
    ds['tilt_x'] = tilt.smooth(ds['tilt_x'],
                               tilt_threshold)
    ds['tilt_y'] = tilt.smooth(ds['tilt_y'],
                               tilt_threshold)

    # Fix tilt to zero if station is on bedrock
    if hasattr(ds, 'bedrock'):
        if ds.attrs['bedrock']==True or ds.attrs['bedrock'].lower() in 'true':

            # Ensures all passed datasets have a 'bedrock' attribute
            ds.attrs['bedrock'] = True

            # Assign all tilt values to zero
            ds['tilt_x'] = (('time'), np.arange(ds['time'].size)*0)
            ds['tilt_y'] = (('time'), np.arange(ds['time'].size)*0)
        else:

            # Ensure all passed datasets have a 'bedrock' attribute
            ds.attrs['bedrock'] = False

    # Ensure all passed datasets have a 'bedrock' attribute
    else:
        ds.attrs['bedrock'] = False

    # One-boom processing steps, if boom attribute equals 1
    if ds.attrs['number_of_booms']==1:

        # Apply z_pt (SR50 stake) offset if provided
        if ~ds['z_pt'].isnull().all():
            if hasattr(ds, 'pt_z_offset'):
                ds['z_pt'] = ds['z_pt'] + int(ds.attrs['pt_z_offset'])
            else:
                logger.info('pt_z_offset value not provided, therefore no offset applied')

            # Calculate pressure transducer fluid density
            ds['z_pt_cor'],ds['z_pt']=pressure_depth.correct(ds['z_pt'],
                                                             ds['p_u'],
                                                             ds.attrs['pt_antifreeze'],
                                                             ds.attrs['pt_z_factor'],
                                                             ds.attrs['pt_z_coef'],
                                                             ds.attrs['pt_z_p_coef'])

        # Reformat stake boom height
        ds['z_stake'] = _reformat_array(ds['z_stake'])

        # Correct stake sonic ranger height for air temperature sensitivity
        ds['z_stake'] = boom_height.correct(ds['z_stake'], ds['t_u'], T_0)

    # Two-boom processing steps, if boom attribute equals 2
    elif ds.attrs['number_of_booms']==2:

        # Reformat boom height values
        ds['z_boom_l'] = _reformat_array(ds['z_boom_l'])

        # Correct boom height for air temperature sensitivity
        ds['z_boom_l'] = boom_height.correct_with_temp_interp(ds['z_boom_l'],
                                                              ds['t_l'],
                                                              vars_df,
                                                              T_0)

    ds = clip_values(ds, vars_df)
    for key in ['hygroclip_t_offset', 'dsr_eng_coef', 'usr_eng_coef',
          'dlr_eng_coef', 'ulr_eng_coef', 'pt_z_coef', 'pt_z_p_coef',
          'pt_z_factor', 'pt_antifreeze', 'boom_azimuth', 'nodata',
          'conf', 'file']:
        ds.attrs.pop(key, None)

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
    '''Reformat DataArray values and attributes
    
    Parameters
    ----------
    ds_arr : xr.Dataarray
        Data array
    
    Returns
    -------
    ds_arr : xr.Dataarray
        Formatted data array
    '''
    a = ds_arr.attrs                                                           # Store
    ds_arr.values = pd.to_numeric(ds_arr, errors='coerce')
    ds_arr.attrs = a                                                           # Reformat
    return ds_arr

#------------------------------------------------------------------------------

if __name__ == "__main__": 
    # unittest.main() 
    pass    
