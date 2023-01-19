#!/usr/bin/env python
"""
pypromice L0 to L1 processing
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import re
import urllib.request

def toL1(L0, vars_df, flag_file=None, T_0=273.15, tilt_threshold=-100):
    '''Process one Level 0 (L0) product to Level 1

    Parameters
    ----------
    L0 : xarray.Dataset
        Level 0 dataset
    vars_df : pd.DataFrame
        Metadata dataframe
    flag_file : str
        Flag .csv file path for bad data
    T_0 : int
        Air temperature for sonic ranger adjustment
    tilt_threshold : int
        Tilt-o-meter threshold for valid measurements
        
    Returns
    -------
    ds : xarray.Dataset
        Level 1 dataset
    '''
    assert(type(L0) == xr.Dataset)
    ds = L0

    ds = _flagNAN(ds)                                                          # Flag NaNs
    ds = _adjustData(ds)                                                       # Flag NaNs

    for l in list(ds.keys()):
        if l not in ['time', 'msg_i', 'gps_lat', 'gps_lon', 'gps_alt', 'gps_time']:
            ds[l] = _reformatArray(ds[l])

    # ds['time_orig'] = ds['time'] # Not used

    # The following drops duplicate datetime indices. Needs to run before _addTimeShift!
    # We can optionally also drop duplicates within _addTimeShift using pandas duplicated,
    # but retaining the following code instead to preserve previous methods. PJW
    _, index = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index)

    # If we do not want to shift hourly average values back -1 hr, then comment the following line.
    ds = _addTimeShift(ds, vars_df)

    if hasattr(ds, 'dsr_eng_coef'): 
        ds['dsr'] = (ds['dsr'] * 10) / ds.attrs['dsr_eng_coef']                # Convert radiation from engineering to physical units
    if hasattr(ds, 'usr_eng_coef'):                                            # TODO add metadata to indicate whether radiometer values are corrected with calibration values or not
        ds['usr'] = (ds['usr'] * 10) / ds.attrs['usr_eng_coef']
    if hasattr(ds, 'dlr_eng_coef'):
        ds['dlr'] = ((ds['dlr'] * 10) / ds.attrs['dlr_eng_coef']) + 5.67E-8*(ds['t_rad'] + T_0)**4
    if hasattr(ds, 'ulr_eng_coef'):
        ds['ulr'] = ((ds['ulr'] * 10) / ds.attrs['ulr_eng_coef']) + 5.67E-8*(ds['t_rad'] + T_0)**4

    ds['z_boom_u'] = _reformatArray(ds['z_boom_u'])                            # Reformat boom height
    ds['z_boom_u'] = ds['z_boom_u'] * ((ds['t_u'] + T_0)/T_0)**0.5             # Adjust sonic ranger readings for sensitivity to air temperature       
    
    if ds['gps_lat'].dtype.kind == 'O':                                        # Decode and reformat GPS information
        if 'NH' in ds['gps_lat'].dropna(dim='time').values[1]:
            ds = _decodeGPS(ds, ['gps_lat','gps_lon','gps_time'])
        else:
            try:
                ds = _decodeGPS(ds, ['gps_lat','gps_lon','gps_time'])          # TODO this is a work around specifically for L0 RAW processing for THU_U. Find a way to make this slicker
            except:
                print('Invalid GPS type {ds["gps_lat"].dtype} for decoding')
            
    for l in ['gps_lat', 'gps_lon', 'gps_alt','gps_time']:
        ds[l] = _reformatArray(ds[l])  

    if hasattr(ds, 'latitude') and hasattr(ds, 'longitude'):
        ds['gps_lat'] = _reformatGPS(ds['gps_lat'], ds.attrs['latitude'])
        ds['gps_lon'] = _reformatGPS(ds['gps_lon'], ds.attrs['longitude'])

    if hasattr(ds, 'logger_type'):                                             # Convert tilt voltage to degrees
        if ds.attrs['logger_type'].upper() == 'CR1000':                    
            ds['tilt_x']  = _getTiltDegrees(ds['tilt_x'], tilt_threshold) 
            ds['tilt_y'] = _getTiltDegrees(ds['tilt_y'], tilt_threshold)  
            
    if hasattr(ds, 'tilt_y_factor'):                                           # Apply tilt factor (e.g. -1 will invert tilt angle)
        ds['tilt_y'] = ds['tilt_y']*ds.attrs['tilt_y_factor']

    # Smooth everything
    # Note that this should be OK for CR1000 tx (data only every 6 hrs),
    # since we interpolate above in _getTiltDegrees. PJW
    ds['tilt_x']  = _smoothTilt(ds['tilt_x'], 7)                               # Smooth tilt
    ds['tilt_y']  = _smoothTilt(ds['tilt_y'], 7)                               # TODO check tilt_y inversion +ive to -ive for Gc-Net stations
    
    if hasattr(ds, 'bedrock'):                                                 # Fix tilt to zero if station is on bedrock
        if ds.attrs['bedrock']==True or ds.attrs['bedrock'].lower() in 'true':
            ds['tilt_x'] = (('time'), np.arange(ds['time'].size)*0)
            ds['tilt_y'] = (('time'), np.arange(ds['time'].size)*0)
            
    ds['wdir_u'] = ds['wdir_u'].where(ds['wspd_u'] != 0)                       # Get directional wind speed                    
    ds['wspd_x_u'], ds['wspd_y_u'] = _calcWindDir(ds['wspd_u'], ds['wdir_u']) 
    
    if ds.attrs['number_of_booms']==1:                                         # 1-boom processing
        if ~ds['z_pt'].isnull().all():                                         # Calculate pressure transducer fluid density                                           
            if hasattr(ds, 'pt_z_offset'):                                     # Apply SR50 stake offset
                ds['z_pt'] = ds['z_pt'] + int(ds.attrs['pt_z_offset'])              
            ds['z_pt_cor'],ds['z_pt']=_getPressDepth(ds['z_pt'], ds['p_u'], 
                                                     ds.attrs['pt_antifreeze'], 
                                                     ds.attrs['pt_z_factor'], 
                                                     ds.attrs['pt_z_coef'], 
                                                     ds.attrs['pt_z_p_coef'])       
            
    elif ds.attrs['number_of_booms']==2:                                       # 2-boom processing
        ds['z_boom_l'] = _reformatArray(ds['z_boom_l'])                        # Reformat boom height    
        ds['z_boom_l'] = ds['z_boom_l'] * ((ds['t_l'] + T_0)/T_0)**0.5         # Adjust sonic ranger readings for sensitivity to air temperature
        ds['wdir_l'] = ds['wdir_l'].where(ds['wspd_l'] != 0)                   # Get directional wind speed    
        ds['wspd_x_l'], ds['wspd_y_l'] = _calcWindDir(ds['wspd_l'], ds['wdir_l'])
     
    if hasattr(ds, 'wdir_i'):    
        if ~ds['wdir_i'].isnull().all() and ~ds['wspd_i'].isnull().all():      # Instantaneous msg processing
            ds['wdir_i'] = ds['wdir_i'].where(ds['wspd_i'] != 0)               # Get directional wind speed                    
            ds['wspd_x_i'], ds['wspd_y_i'] = _calcWindDir(ds['wspd_i'], ds['wdir_i'])   
    return ds

def _flagNAN(ds_in, flag_file=None):
    '''Read flagged data from .csv file. For each variable, and downstream 
    dependents, flag as invalid (or other) if set in the flag .csv
    
    Parameters
    ----------
    ds_in : xr.Dataset
        Level 0 dataset
    flag_file : str
        File path to .csv flag file. The default is None.
    
    Returns
    -------
    ds : xr.Dataset
        Level 0 data with flagged data
    '''
    ds = ds_in.copy()
    try:
        os.mkdir('local')
        os.mkdir('local/flags')
        os.mkdir('local/adjustments')
    except:
        pass
    flag_url = "https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-data-issues/master/flags/" 
    try:
        urllib.request.urlretrieve(flag_url + ds.attrs["station_id"] + ".csv",
                               "local/flags/" + ds.attrs["station_id"] + ".csv")
    except:
        print('Could not find the flag file')
        print(flag_url + ds.attrs["station_id"] + ".csv")
        return ds
    
    flag_file = "local/flags/" + ds.attrs["station_id"] + ".csv"
    df = pd.read_csv(
                    flag_file,
                    comment="#", 
                    skipinitialspace=True,
                    ).dropna(how='all', axis='rows')
    
    df.t0 = pd.to_datetime(df.t0).dt.tz_localize(None)
    df.t1 = pd.to_datetime(df.t1).dt.tz_localize(None)
    # For now we only process the NAN flag
    df = df[df['flag'] == "NAN"]
    if df.shape[0] == 0: 
        return ds
    
    # Set flagged values
    for i in df.index:
        t0, t1, avar = df.loc[i,['t0','t1','variable']]
        
        # Set to all vars if var is "*"
        varlist = avar.split() if avar != '*' else list(ds.variables)
 
        if 'time' in varlist: varlist.remove("time")
        
        # Set to all times if times are "n/a"
        if pd.isnull(t0): 
            t0 = ds['time'].values[0]
        if pd.isnull(t1): 
            t1 = ds['time'].values[0]
        
        for v in varlist:
            ds[v] = ds[v].where((ds['time'] < t0) | (ds['time'] > t1))
        
        # TODO: Mark these values in the ds_flags dataset using perhaps 
        # flag_LUT.loc["NAN"]['value']
    return ds


def _adjustData(ds, var_list=[], skip_var=[]):
    ds_out = ds.copy()
    
    adj_url = "https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-data-issues/master/adjustments/" 
    try:
        urllib.request.urlretrieve(adj_url + ds.attrs["station_id"] + ".csv",
                               "local/adjustments/" + ds.attrs["station_id"] + ".csv")
    except:
        print('Could not find the adjustment file')
        print(adj_url + ds.attrs["station_id"] + "2.csv")
        return ds
    
    flag_file = "local/adjustments/" + ds.attrs["station_id"] + ".csv"
    adj_info = pd.read_csv(
        flag_file, 
        comment="#", 
        skipinitialspace=True,
    )

    adj_info.t0 = pd.to_datetime(adj_info.t0, utc=True)

    # if t1 is left empty, then adjustment is applied until the end of the file
    adj_info.loc[adj_info.t1.isnull(), "t1"] = pd.to_datetime(ds_out.time.values[-1]).isoformat()
    adj_info.t1 = pd.to_datetime(adj_info.t1, utc=True)

    # if "*" is given as variable then we append this adjustement for all variables
    # needs to be re-implemented
    # for ind in adj_info.loc[adj_info.variable == "*", :].time:
    #     line_template = adj_info.loc[ind, :].copy()
    #     for var in ds_out.columns:
    #         line_template.variable = var
    #         line_template.name = adj_info.time.max() + 1
    #         adj_info = adj_info.append(line_template)
    #     adj_info = adj_info.drop(labels=ind, axis=0)

    # first applies the time shift
    # at the moment time shift is applied in the time index, so applied to all variables
    # can be adapted to a variable-specific shift
    print(adj_info.adjust_function)
    if "time_shift" in adj_info.adjust_function.values:
        time_shifts = adj_info.loc[adj_info.adjust_function == "time_shift", :]
        adj_info = adj_info.loc[adj_info.adjust_function != "time_shift", :]
            
        for t0, t1, val in zip(
            time_shifts.t0,
            time_shifts.t1,
            time_shifts.adjust_value,
        ):
            t0 = pd.to_datetime(t0)
            t1 = pd.to_datetime(t1)
            ds_shifted = ds_out.sel(time=slice(t0,t1))
            ds_shifted['time'] = ds_shifted.time.values + pd.Timedelta(days = val)
            
            # here we concatenate what was before the shifted part, the shifted
            # part and what was after the shifted part
            # note that if any data was already present in the target period 
            # (where the data lands after the shift), it is overwritten
            
            ds_out = xr.concat(
                                    (
                                        ds_out.sel(time=slice(pd.to_datetime(ds_out.time.values[0], utc=True),
                                                              t0 + pd.Timedelta(days = val))),
                                        ds_shifted,
                                        ds_out.sel(time=slice(t1 + pd.Timedelta(days = val),
                                                              pd.to_datetime(ds_out.time.values[-1], utc=True)))
                                    ),
                                    dim = 'time',
                                   )
            if t0 > pd.Timestamp.now(tz='utc'):
                ds_out = ds_out.sel(time=slice(pd.to_datetime(ds_out.time.values[0], utc=True),
                                               t0))
    
    # now applying all the other adjustments
    adj_info = adj_info.sort_values(by=["variable", "t0"])
    adj_info.set_index(["variable", "t0"], drop=False, inplace=True)

    if len(var_list) == 0:
        var_list = np.unique(adj_info.variable)
    else:
        adj_info = adj_info.loc[np.isin(adj_info.variable, var_list), :]
        var_list = np.unique(adj_info.variable)

    if len(skip_var) > 0:
        adj_info = adj_info.loc[~np.isin(adj_info.variable, skip_var), :]
        var_list = np.unique(adj_info.variable)

    for var in var_list:
        for t0, t1, func, val in zip(
            adj_info.loc[var].t0,
            adj_info.loc[var].t1,
            adj_info.loc[var].adjust_function,
            adj_info.loc[var].adjust_value,
        ):
            if (t0 > pd.to_datetime(ds_out.time.values[-1], utc=True)) | (t1 < pd.to_datetime(ds_out.time.values[0], utc=True)):
                continue


            if func == "add":
                ds_out[var].loc[dict(time=slice(t0, t1))] = ds_out[var].loc[dict(time=slice(t0, t1))].values + val
                # flagging adjusted values
                # if var + "_adj_flag" not in ds_out.columns:
                #     ds_out[var + "_adj_flag"] = 0
                # msk = ds_out[var].loc[dict(time=slice(t0, t1))])].notnull()
                # ind = ds_out[var].loc[dict(time=slice(t0, t1))])].loc[msk].time
                # ds_out.loc[ind, var + "_adj_flag"] = 1

            if func == "multiply":
                ds_out[var].loc[dict(time=slice(t0, t1))] = ds_out[var].loc[dict(time=slice(t0, t1))].values * val
                if "DW" in var:
                    ds_out[var].loc[dict(time=slice(t0, t1))] = ds_out[var].loc[dict(time=slice(t0, t1))] % 360
                # flagging adjusted values
                # if var + "_adj_flag" not in ds_out.columns:
                #     ds_out[var + "_adj_flag"] = 0
                # msk = ds_out[var].loc[dict(time=slice(t0, t1))].notnull()
                # ind = ds_out[var].loc[dict(time=slice(t0, t1))].loc[msk].time
                # ds_out.loc[ind, var + "_adj_flag"] = 1

            if func == "min_filter":
                tmp = ds_out[var].loc[dict(time=slice(t0, t1))].values
                tmp[tmp < val] = np.nan
                
            if func == "max_filter":
                tmp = ds_out[var].loc[dict(time=slice(t0, t1))].values
                tmp[tmp > val] = np.nan
                ds_out[var].loc[dict(time=slice(t0, t1))] = tmp
                
            if func == "upper_perc_filter":
                tmp = ds_out[var].loc[dict(time=slice(t0, t1))].copy()
                df_w = ds_out[var].loc[dict(time=slice(t0, t1))].resample("14D").quantile(1 - val / 100)
                df_w = ds_out[var].loc[dict(time=slice(t0, t1))].resample("14D").var()
                for m_start, m_end in zip(df_w.time[:-2], df_w.time[1:]):
                    msk = (tmp.time >= m_start) & (tmp.time < m_end)
                    values_month = tmp.loc[msk].values
                    values_month[values_month < df_w.loc[m_start]] = np.nan
                    tmp.loc[msk] = values_month

                ds_out[var].loc[dict(time=slice(t0, t1))] = tmp.values

            if func == "biweekly_upper_range_filter":
                tmp = ds_out[var].loc[dict(time=slice(t0, t1))].copy()
                df_max = ds_out[var].loc[dict(time=slice(t0, t1))].resample("14D").max()
                for m_start, m_end in zip(df_max.time[:-2], df_max.time[1:]):
                    msk = (tmp.time >= m_start) & (tmp.time < m_end)
                    lim = df_max.loc[m_start] - val
                    values_month = tmp.loc[msk].values
                    values_month[values_month < lim] = np.nan
                    tmp.loc[msk] = values_month
                # remaining samples following outside of the last 2 weeks window
                msk = tmp.time >= m_end
                lim = df_max.loc[m_start] - val
                values_month = tmp.loc[msk].values
                values_month[values_month < lim] = np.nan
                tmp.loc[msk] = values_month
                # updating original pandas
                ds_out[var].loc[dict(time=slice(t0, t1))] = tmp.values

            if func == "hampel_filter":
                tmp = ds_out[var].loc[dict(time=slice(t0, t1))]
                tmp = _hampel(tmp, k=7 * 24, t0=val)
                ds_out[var].loc[dict(time=slice(t0, t1))] = tmp.values

            if func == "grad_filter":
                tmp = ds_out[var].loc[dict(time=slice(t0, t1))].copy()
                msk = ds_out[var].loc[dict(time=slice(t0, t1))].copy().diff()
                tmp[np.roll(msk.abs() > val, -1)] = np.nan
                ds_out[var].loc[dict(time=slice(t0, t1))] = tmp

            if "swap_with_" in func:
                var2 = func[10:]
                val_var = ds_out[var].loc[dict(time=slice(t0, t1))].values.copy()
                val_var2 = ds_out[var2].loc[dict(time=slice(t0, t1))].values.copy()
                ds_out[var2].loc[dict(time=slice(t0, t1))] = val_var
                ds_out[var].loc[dict(time=slice(t0, t1))] = val_var2

            if func == "rotate":
                ds_out[var].loc[dict(time=slice(t0, t1))] = (ds_out[var].loc[dict(time=slice(t0, t1))].values + val) % 360

    return ds_out


def _hampel(vals_orig, k=7*24, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    #Make copy so original not edited
    vals=vals_orig.copy()    
    #Hampel Filter
    L= 1.4826
    rolling_median=vals.rolling(k).median()
    difference=np.abs(rolling_median-vals)
    median_abs_deviation=difference.rolling(k).median()
    threshold= t0 *L * median_abs_deviation
    outlier_idx=difference>threshold
    outlier_idx[0:round(k/2)]=False
    vals.loc[outlier_idx]=np.nan
    return(vals)


def _popCols(ds, booms, data_type, vars_df, cols):
    '''Populate data array columns with given variable names from look-up table
    
    Parameters
    ----------
    ds : xr.Dataset
        Data set
    booms : int
        Number of booms (1 or 2)
    data_type : str
        Type of data ("tx", "raw")
    vars_df : pd.DataFrame
        Variables lookup table
    cols : list
        Names of columns to populate
    
    Returns
    -------
    ds : xr.Dataset
        Data with populated columns
    '''
    if booms==1:
        names = vars_df.loc[(vars_df[cols[0]]!='two-boom')]

    elif booms==2:
        names = vars_df.loc[(vars_df[cols[0]]!='one-boom')]
       
    for v in list(names.index):
        if v not in list(ds.variables):
            ds[v] = (('time'), np.arange(ds['time'].size)*np.nan)      
    return ds

# def _popCols(ds, booms, data_type, vars_df, cols):
#     if booms==1:
#         if data_type !='TX':
#             names = vars_df.loc[(vars_df[cols[0]]!='two-boom')]
#         else:
#             names = vars_df.loc[(vars_df[cols[0]] != 'two-boom') & vars_df[cols[1]] != 'tx']
    
#     elif booms==2:
#         if data_type !='TX':
#             names = vars_df.loc[(vars_df[cols[0]]!='two-boom')]
#         else:
#             names = vars_df.loc[(vars_df[cols[0]] != 'two-boom') & vars_df[cols[1]] != 'tx']
       
#     for v in list(names.index):
#         if v not in list(ds.variables):
#             ds[v] = (('time'), np.arange(ds['time'].size)*np.nan)      
#     return ds

def _addTimeShift(ds, vars_df):
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
        df_a = df_a.shift(periods=-1, freq="H")
        df_out = pd.concat([df_a, df_i], axis=1) # different columns, same datetime indices
    elif ds.attrs['format'] == 'TX':
        if ds.attrs['logger_type'] == 'CR1000X':
            # v3, data is hourly all year long
            # shift everything except instantaneous
            df_a = df_a.shift(periods=-1, freq="H")
            df_out = pd.concat([df_a, df_i], axis=1) # different columns, same datetime indices
        elif ds.attrs['logger_type'] == 'CR1000':
            # v2, data is hourly (6-hr for instantaneous) for DOY 100-300, otherwise daily at 00 UTC
            # shift non-instantaneous hourly for DOY 100-300, else do not shift daily
            df_a_hourly = df_a.loc[(df_a['doy'] >= 100) & (df_a['doy'] <= 300)]
            # df_a_hourly = df_a.loc[df_a['doy'].between(100, 300, inclusive='both')] # equivalent to above
            df_a_daily_1 = df_a.loc[(df_a['doy'] < 100)]
            df_a_daily_2 = df_a.loc[(df_a['doy'] > 300)]

            # shift the hourly ave data
            df_a_hourly = df_a_hourly.shift(periods=-1, freq="H")

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

def _removeVars(ds, v_names):
    '''Remove redundant variables if present in dataset
    
    Parameters
    ----------
    ds : xr.Dataset
        Data set
    v_names : list
        List of column names to drop
    
    Returns
    -------
    ds : xr.Dataset
        Data set with removed variables
    '''
    for v in v_names:
        if v in list(ds.variables): ds = ds.drop_vars(v)
    return ds

def _getPressDepth(z_pt, p, pt_antifreeze, pt_z_factor, pt_z_coef, pt_z_p_coef): 
    '''Adjust pressure depth and calculate pressure transducer depth based on 
    pressure transducer fluid density
    
    Parameters
    ----------
    z_pt : xr.Dataarray
        Pressure transducer height (corrected for offset)
    p : xr.Dataarray
        Air pressure
    pt_antifreeze : float
        Pressure transducer anti-freeze percentage for fluid density 
        correction
    pt_z_factor : float
        Pressure transducer factor
    pt_z_coef : float
        Pressure transducer coefficient
    pt_z_p_coef : float
        Pressure transducer coefficient
    
    Returns
    -------
    z_pt_cor : xr.Dataarray
        Pressure transducer height corrected
    z_pt : xr.Dataarray
        Pressure transducer depth
    '''
    # Calculate pressure transducer fluid density                                        
    if pt_antifreeze == 50:                                                    #TODO: Implement function w/ reference (analytical or from LUT)                                             
        rho_af = 1092                                                          #TODO: Track uncertainty
    elif pt_antifreeze == 100:
        rho_af = 1145
    else:
        rho_af = np.nan
        print('ERROR: Incorrect metadata: "pt_antifreeze" = ' +
              f'{pt_antifreeze}. Antifreeze mix only supported at 50% or 100%')
        # assert(False)
                
    # Correct pressure depth
    z_pt_cor = z_pt * pt_z_coef * pt_z_factor * 998.0 / rho_af + 100 * (pt_z_p_coef - p) / (rho_af * 9.81)

    # Calculate pressure transducer depth
    z_pt = z_pt * pt_z_coef * pt_z_factor * 998.0 / rho_af
    
    return z_pt_cor, z_pt

def _smoothTilt(tilt, win_size):
    '''Smooth tilt values using a rolling window. This is translated from the
    previous IDL/GDL smoothing algorithm:
    tiltX = smooth(tiltX,7,/EDGE_MIRROR,MISSING=-999) & tiltY = smooth(tiltY,7,/EDGE_MIRROR, MISSING=-999)
    endif
    In Python, this should be
    dstxy = dstxy.rolling(time=7, win_type='boxcar', center=True).mean()
    But the EDGE_MIRROR makes it a bit more complicated
    
    Parameters
    ----------
    tilt : xarray.DataArray
        Array (either 'tilt_x' or 'tilt_y'), tilt values (can be in degrees or voltage)
    win_size : int
        Window size to use in pandas 'rolling' method.
        e.g. a value of 7 spans 70 minutes using 10 minute data.

    Returns
    -------
    tdf_rolling : tuple, as: (str, numpy.ndarray)
        The numpy array is the tilt values, smoothed with a rolling mean
    '''
    s = int(win_size/2)
    tdf = tilt.to_dataframe()
    mirror_start = tdf.iloc[:s][::-1]
    mirror_end = tdf.iloc[-s:][::-1]
    mirrored_tdf = pd.concat([mirror_start, tdf, mirror_end])

    tdf_rolling = (
        ('time'),
        mirrored_tdf.rolling(
            win_size, win_type='boxcar', min_periods=1, center=True
            ).mean()[s:-s].values.flatten()
        )
    return tdf_rolling

def _getTiltDegrees(tilt, threshold):
    '''Filter tilt with given threshold, and convert from voltage to degrees. 
    Voltage-to-degrees converseion is based on the equation in 3.2.9 at 
    https://essd.copernicus.org/articles/13/3819/2021/#section3    

    Parameters
    ----------
    tilt : xarray.DataArray
        Array (either 'tilt_x' or 'tilt_y'), tilt values (voltage)
    threshold : int
        Values below this threshold (-100) will not be retained.
    
    Returns
    -------
    dst.interpolate_na() : xarray.DataArray
        Array (either 'tilt_x' or 'tilt_y'), tilt values (degrees)
    '''
    # notOKtiltX = where(tiltX lt -100, complement=OKtiltX) & notOKtiltY = where(tiltY lt -100, complement=OKtiltY)
    notOKtilt = (tilt < threshold)
    OKtilt = (tilt >= threshold)
    tilt = tilt / 10
    
    # IDL version:
    # tiltX = tiltX/10.
    # tiltnonzero = where(tiltX ne 0 and tiltX gt -40 and tiltX lt 40)
    # if n_elements(tiltnonzero) ne 1 then tiltX[tiltnonzero] = tiltX[tiltnonzero]/abs(tiltX[tiltnonzero])*(-0.49*(abs(tiltX[tiltnonzero]))^4 + 3.6*(abs(tiltX[tiltnonzero]))^3 - 10.4*(abs(tiltX[tiltnonzero]))^2 +21.1*(abs(tiltX[tiltnonzero])))
    # tiltY = tiltY/10.
    # tiltnonzero = where(tiltY ne 0 and tiltY gt -40 and tiltY lt 40)
    # if n_elements(tiltnonzero) ne 1 then tiltY[tiltnonzero] = tiltY[tiltnonzero]/abs(tiltY[tiltnonzero])*(-0.49*(abs(tiltY[tiltnonzero]))^4 + 3.6*(abs(tiltY[tiltnonzero]))^3 - 10.4*(abs(tiltY[tiltnonzero]))^2 +21.1*(abs(tiltY[tiltnonzero])))
    
    dst = tilt
    nz = (dst != 0) & (np.abs(dst) < 40)
    
    dst = dst.where(~nz, other = dst / np.abs(dst)
                      * (-0.49
                         * (np.abs(dst))**4 + 3.6
                         * (np.abs(dst))**3 - 10.4
                         * (np.abs(dst))**2 + 21.1
                         * (np.abs(dst))))
    
    # if n_elements(OKtiltX) gt 1 then tiltX[notOKtiltX] = interpol(tiltX[OKtiltX],OKtiltX,notOKtiltX) ; Interpolate over gaps for radiation correction; set to -999 again below.
    dst = dst.where(~notOKtilt)
    return dst.interpolate_na(dim='time', use_coordinate=False)                #TODO: Filling w/o considering time gaps to re-create IDL/GDL outputs. Should fill with coordinate not False. Also consider 'max_gap' option?

def _calcWindDir(wspd, wdir, deg2rad=np.pi/180):
    '''Calculate directional wind speed from wind speed and direction
    
    Parameters
    ----------
    wspd : xr.Dataarray
        Wind speed data array
    wdir : xr.Dataarray
        Wind direction data array
    deg2rad : float
        Degree to radians coefficient. The default is np.pi/180
    
    Returns
    -------
    wspd_x : xr.Dataarray
        Wind speed in X direction
    wspd_y : xr.Datarray
        Wind speed in Y direction
    '''        
    wspd_x = wspd * np.sin(wdir * deg2rad)
    wspd_y = wspd * np.cos(wdir * deg2rad) 
    return wspd_x, wspd_y
    
def _decodeGPS(ds, gps_names):
    '''Decode GPS information based on names of GPS attributes. This should be 
    applied if gps information does not consist of float values
    
    Parameters
    ----------
    ds : xr.Dataset
        Data set
    gps_names : list
        Variable names for GPS information, such as "gps_lat", "gps_lon" and
        "gps_alt"
    
    Returns
    -------
    ds : xr.Dataset
        Data set with decoded GPS information
    '''
    for v in gps_names:
        a = ds[v].attrs    
        str2nums = [re.findall(r"[-+]?\d*\.\d+|\d+", _) if isinstance(_, str) else [np.nan] for _ in ds[v].values]
        ds[v][:] = pd.DataFrame(str2nums).astype(float).T.values[0]
        ds[v] = ds[v].astype(float)
        ds[v].attrs = a 
    return ds

def _reformatArray(ds_arr):
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

def _reformatGPS(pos_arr, attrs):
    '''Correct position if only recorded minutes (and not degrees), and 
    reformat values and attributes
    
    Parameters
    ----------
    pos_arr : xr.Dataarray
        GPS position array
    attrs : dict
        Array attributes
    
    Returns
    -------
    pos_arr : xr.Dataarray
        Formatted GPS position array
    '''       
    if np.any((pos_arr <= 90) & (pos_arr > 0)):  
        pos_arr = pos_arr + 100*attrs
    a = pos_arr.attrs                                                     
    pos_arr = np.floor(pos_arr / 100) + (pos_arr / 100 - np.floor(pos_arr / 100)) * 100 / 60
    pos_arr.attrs = a 
    return pos_arr       
        
#------------------------------------------------------------------------------

if __name__ == "__main__": 
    # unittest.main() 
    pass    
