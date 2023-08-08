#!/usr/bin/env python
"""
AWS Level 1 (L1) to Level 2 (L2) data processing
"""
import numpy as np
import urllib.request
from urllib.error import HTTPError, URLError
import pandas as pd
import subprocess
import os
import xarray as xr
import sqlite3

# from IPython import embed

def toL2(L1, T_0=273.15, ews=1013.246, ei0=6.1071, eps_overcast=1., 
         eps_clear=9.36508e-6, emissivity=0.97):
    '''Process one Level 1 (L1) product to Level 2

    Parameters
    ----------
    L1 : xarray.Dataset
        Level 1 dataset
    T_0 : float, optional
        Ice point temperature in K. The default is 273.15.
    ews : float, optional
        Saturation pressure (normal atmosphere) at steam point temperature. 
        The default is 1013.246.
    ei0 : float, optional
        Saturation pressure (normal atmosphere) at ice-point temperature. The 
        default is 6.1071.
    eps_overcast : int, optional
        Cloud overcast. The default is 1..
    eps_clear : int, optional
        Cloud clear. The default is 9.36508e-6.
    emissivity : int, optional
        Emissivity. The default is 0.97.

    Returns
    -------
    ds : xarray.Dataset
        Level 2 dataset
    '''
    ds = L1.copy(deep=True)                                                    # Reassign dataset
    try:
        ds = adjustTime(ds)                                                    # Adjust time after a user-defined csv files
        ds = flagNAN(ds)                                                       # Flag NaNs after a user-defined csv files
        ds = adjustData(ds)                                                    # Adjust data after a user-defined csv files
    except Exception as e: 
        print('Flagging and fixing failed:')
        print(e)
        
        
    ds = differenceQC(ds)                                                      # Flag and Remove difference outliers     
    ds = percentileQC(ds)                                                      # Flag and remove percentile outliers

    T_100 = _getTempK(T_0)  
    ds['rh_u_cor'] = correctHumidity(ds['rh_u'], ds['t_u'],  
                                     T_0, T_100, ews, ei0)                       
        
    # Determiune cloud cover
    cc = calcCloudCoverage(ds['t_u'], T_0, eps_overcast, eps_clear,            # Calculate cloud coverage
                           ds['dlr'], ds.attrs['station_id'])  
    ds['cc'] = (('time'), cc.data)
    
    # Determine surface temperature
    ds['t_surf'] = calcSurfaceTemperature(T_0, ds['ulr'], ds['dlr'],           # Calculate surface temperature
                                          emissivity)
    
    # Determine station position relative to sun    
    doy = ds['time'].to_dataframe().index.dayofyear.values                     # Gather variables to calculate sun pos
    hour = ds['time'].to_dataframe().index.hour.values
    minute = ds['time'].to_dataframe().index.minute.values
    
    if hasattr(ds, 'latitude') and hasattr(ds, 'longitude'):
        lat = ds.attrs['latitude']                                             # TODO Why is mean GPS lat lon not preferred for calcs?
        lon = ds.attrs['longitude']
    else:
        lat = ds['gps_lat'].mean()
        lon = ds['gps_lon'].mean()
    
    deg2rad, rad2deg = _getRotation()                                          # Get degree-radian conversions  
    phi_sensor_rad, theta_sensor_rad = calcTilt(ds['tilt_x'], ds['tilt_y'],    # Calculate station tilt 
                                                deg2rad)  
    
    Declination_rad = calcDeclination(doy, hour, minute)                       # Calculate declination
    HourAngle_rad = calcHourAngle(hour, minute, lon)                           # Calculate hour angle       
    ZenithAngle_rad, ZenithAngle_deg = calcZenith(lat, Declination_rad,        # Calculate zenith
                                                  HourAngle_rad, deg2rad, 
                                                  rad2deg)
    
    # Correct Downwelling shortwave radiation
    DifFrac = 0.2 + 0.8 * cc 
    CorFac_all = calcCorrectionFactor(Declination_rad, phi_sensor_rad,         # Calculate correction
                                      theta_sensor_rad, HourAngle_rad, 
                                      ZenithAngle_rad, ZenithAngle_deg, 
                                      lat, DifFrac, deg2rad)        
    ds['dsr_cor'] = ds['dsr'].copy(deep=True) * CorFac_all                     # Apply correction

    AngleDif_deg = calcAngleDiff(ZenithAngle_rad, HourAngle_rad,               # Calculate angle between sun and sensor
                                 phi_sensor_rad, theta_sensor_rad)   
    
    ds['albedo'], OKalbedos = calcAlbedo(ds['usr'], ds['dsr_cor'],             # Determine albedo
                              AngleDif_deg, ZenithAngle_deg)

    # Correct upwelling and downwelling shortwave radiation
    sunonlowerdome =(AngleDif_deg >= 90) & (ZenithAngle_deg <= 90)             # Determine when sun is in FOV of lower sensor, assuming sensor measures only diffuse radiation
    ds['dsr_cor'] = ds['dsr_cor'].where(~sunonlowerdome, 
                                        other=ds['dsr'] / DifFrac)             # Apply to downwelling
    ds['usr_cor'] = ds['usr'].copy(deep=True)
    ds['usr_cor'] = ds['usr_cor'].where(~sunonlowerdome, 
                                        other=ds['albedo'] * ds['dsr'] / DifFrac) # Apply to upwelling
    bad = (ZenithAngle_deg > 95) | (ds['dsr_cor'] <= 0) | (ds['usr_cor'] <= 0) # Set to zero for solar zenith angles larger than 95 deg or either values are (less than) zero
    ds['dsr_cor'][bad] = 0
    ds['usr_cor'][bad] = 0
    ds['dsr_cor'] = ds['usr_cor'].copy(deep=True) / ds['albedo']               # Correct DWR using more reliable USWR when sun not in sight of upper sensor  
    ds['albedo'] = ds['albedo'].where(OKalbedos)                               #TODO remove?      
    
    # Remove data where TOA shortwave radiation invalid
    isr_toa = calcTOA(ZenithAngle_deg, ZenithAngle_rad)                        # Calculate TOA shortwave radiation                         
    TOA_crit_nopass = (ds['dsr_cor'] > (0.9 * isr_toa + 10))                   # Determine filter
    ds['dsr_cor'][TOA_crit_nopass] = np.nan                                    # Apply filter and interpolate
    ds['usr_cor'][TOA_crit_nopass] = np.nan
    ds['dsr_cor'] = ds['dsr_cor'].interpolate_na(dim='time', use_coordinate=False)
    ds['usr_cor'] = ds['usr_cor'].interpolate_na(dim='time', use_coordinate=False)

    # # Check sun position
    # sundown = ZenithAngle_deg >= 90
    # _checkSunPos(ds, OKalbedos, sundown, sunonlowerdome, TOA_crit_nopass)    

    if hasattr(ds, 'correct_precip'):                                          # Correct precipitation
        precip_flag=ds.attrs['correct_precip']
    else:
        precip_flag=True
    if ~ds['precip_u'].isnull().all() and precip_flag: 
        ds['precip_u_cor'], ds['precip_u_rate'] = correctPrecip(ds['precip_u'], 
                                                                ds['wspd_u'])
    if ds.attrs['number_of_booms']==2:      
        ds['rh_l_cor'] = correctHumidity(ds['rh_l'], ds['t_l'],           # Correct relative humidity
                                         T_0, T_100, ews, ei0)                          
        
        if ~ds['precip_l'].isnull().all() and precip_flag:                     # Correct precipitation
            ds['precip_l_cor'], ds['precip_l_rate']= correctPrecip(ds['precip_l'],  
                                                                   ds['wspd_l'])
    
    if hasattr(ds,'t_i'):       
        if ~ds['t_i'].isnull().all():                                          # Instantaneous msg processing
            ds['rh_i_cor'] = correctHumidity(ds['rh_i'], ds['t_i'],       # Correct relative humidity
                                             T_0, T_100, ews, ei0)                   
    return ds

def differenceQC(ds):
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
    df = ds.to_dataframe() # Switch to pandas
    
    # Define threshold dict to hold limit values, and the difference values.
    # Limit values indicate how much a variable has to change to the previous value
    # diff_period is how many hours a value can stay the same without being set to NaN 
    # * are used to calculate and define all limits, which are then applied to *_u, *_l and *_i
    
    var_threshold = {
        't': {'static_limit': 0.001, 'diff_period' : 1}, 
        'p': {'static_limit': 0.0001, 'diff_period' : 24},
        'rh': {'static_limit': 0.0001, 'diff_period' : 24}
        }

    
    for k in var_threshold.keys():
        
         var_all = [k + '_u',k + '_l',k + '_i'] # apply to upper, lower boom, and instant
         static_lim = var_threshold[k]['static_limit'] # loading static limit
         diff_h = var_threshold[k]['diff_period'] # loading diff period
         
         for v in var_all:
             if v in df:
                
                data = df[v]
                diff = data.diff()
                diff.fillna(method='ffill', inplace=True) # forward filling all NaNs! 
                diff = np.array(diff)

                diff_period = np.ones_like(diff) * False
                
                for i,d in enumerate(diff): # algorithm that ensures values can stay the same within the diff_period
                    if i > (diff_h-1): 
                        if sum(abs(diff[i-diff_h:i])) < static_lim:
                            diff_period[i-diff_h:i] = True
                
                diff_period = np.array(diff_period).astype('bool')
                
                outliers = df[v][diff_period] # finding outliers in dataframe
                
                df.loc[outliers.index,v] = np.nan # setting outliers to NaN
                
    # Back to xarray, and re-assign the original attrs
    ds_out = df.to_xarray()
    ds_out = ds_out.assign_attrs(ds.attrs) # Dataset attrs
    for x in ds_out.data_vars: # variable-specific attrs
        ds_out[x].attrs = ds[x].attrs
    # equivalent to above:
    # vals = [xr.DataArray(data=df_out[c], dims=['time'], coords={'time':df_out.index}, attrs=ds[c].attrs) for c in df_out.columns]
    # ds_out = xr.Dataset(dict(zip(df_out.columns, vals)), attrs=ds.attrs)
    return ds_out
    

def percentileQC(ds):
    '''
    Parameters
    ----------
    ds : xr.Dataset
        Level 1 dataset

    Returns
    -------
    ds_out : xr.Dataset
        Level 1 dataset with percentile outliers set to NaN
    '''
    # Check if on-disk sqlite db exists
    # If it not exists, then run compute_percentiles.py
    
    file_path = '../qc/percentiles.db'
    script_path = os.path.abspath('../qc/compute_percentiles.py')
    
    current_path = os.getcwd()
    print(f'This is my current Path {current_path}')    
    
    if not os.path.isfile(file_path):
        subprocess.run([script_path])
    
    # Optionally examine flagged data by setting make_plots to True
    # This is best done by running aws.py directly and setting 'test_station'
    # Plots will be shown before and after flag removal for each var
    make_plots = False

    stid = ds.station_id
    df = ds.to_dataframe() # Switch to pandas

    # Define threshold dict to hold limit values, and 'hi' and 'lo' percentile.
    # Limit values indicate how far we will go beyond the hi and lo percentiles to flag outliers.
    # *_u are used to calculate and define all limits, which are then applied to *_u, *_l and *_i
    var_threshold = {
        't_u': {'limit': 9}, # 'hi' and 'lo' will be held in 'seasons' dict
        'p_u': {'limit': 12},
        'rh_u': {'limit': 12},
        'wspd_u': {'limit': 12},
        }

    
    # Query from the on-disk sqlite db for specified percentiles
    con = sqlite3.connect('../qc/percentiles.db')
    cur = con.cursor()
    for k in var_threshold.keys():
        if k == 't_u':
            # Different pattern for t_u, which considers seasons
            # 1: winter (DecJanFeb), 2: spring (MarAprMay), 3: summer (JunJulAug), 4: fall (SepOctNov)
            seasons = {1: {}, 2: {}, 3: {}, 4: {}}
            sql = f"SELECT p0p5,p99p5,season FROM {k} WHERE season in (1,2,3,4) and stid = ?"
            cur.execute(sql, [stid])
            result = cur.fetchall()
            for row in result:
                # row[0] is p0p5, row[1] is p99p5, row[2] is the season integer
                seasons[row[2]]['lo'] = row[0] # 0.005
                seasons[row[2]]['hi'] = row[1] # 0.995
                var_threshold[k]['seasons'] = seasons
        else:
            sql = f"SELECT p0p5,p99p5 FROM {k} WHERE stid = ?"
            cur.execute(sql, [stid])
            result = cur.fetchone() # we only expect one row back per station
            var_threshold[k]['lo'] = result[0] # 0.005
            var_threshold[k]['hi'] = result[1] # 0.995

    con.close() # close the database connection (and cursor)

    # Set flagged data to NaN
    for k in var_threshold.keys():
        if k == 't_u':
            # use t_u thresholds to flag t_u, t_l, t_i
            base_var = k.split('_')[0]
            vars_all = [k, base_var+'_l', base_var+'_i']
            for t in vars_all:
                if t in df:
                    winter = df[t][df.index.month.isin([12,1,2])]
                    spring = df[t][df.index.month.isin([3,4,5])]
                    summer = df[t][df.index.month.isin([6,7,8])]
                    fall = df[t][df.index.month.isin([9,10,11])]
                    season_dfs = [winter,spring,summer,fall]

                    if make_plots:
                        _plot_percentiles_t(k,t,df,season_dfs,var_threshold,stid) # BEFORE OUTLIER REMOVAL
                    for x1,x2 in zip([1,2,3,4], season_dfs):
                        print(f'percentile flagging {t} {x1}')
                        lower_thresh = var_threshold[k]['seasons'][x1]['lo'] - var_threshold[k]['limit']
                        upper_thresh = var_threshold[k]['seasons'][x1]['hi'] + var_threshold[k]['limit']
                        outliers_upper = x2[x2.values > upper_thresh]
                        outliers_lower = x2[x2.values < lower_thresh]
                        outliers = pd.concat([outliers_upper,outliers_lower])
                        df.loc[outliers.index,t] = np.nan
                        df.loc[outliers.index,t] = np.nan

                    if make_plots:
                        _plot_percentiles_t(k,t,df,season_dfs,var_threshold,stid) # AFTER OUTLIER REMOVAL
        else:
            # use *_u thresholds to flag *_u, *_l, *_i
            base_var = k.split('_')[0]
            vars_all = [k, base_var+'_l', base_var+'_i']
            for t in vars_all:
                if t in df:
                    print(f'percentile flagging {t}')
                    upper_thresh = var_threshold[k]['hi'] + var_threshold[k]['limit']
                    lower_thresh = var_threshold[k]['lo'] - var_threshold[k]['limit']
                    if make_plots:
                        _plot_percentiles(k,t,df,var_threshold,upper_thresh,lower_thresh,stid) # BEFORE OUTLIER REMOVAL
                    if t == 'p_i':
                        # shift p_i so we can use the p_u thresholds
                        shift_p = df[t]+1000.
                        outliers_upper = shift_p[shift_p.values > upper_thresh]
                        outliers_lower = shift_p[shift_p.values < lower_thresh]
                    else:
                        outliers_upper = df[t][df[t].values > upper_thresh]
                        outliers_lower = df[t][df[t].values < lower_thresh]
                    outliers = pd.concat([outliers_upper,outliers_lower])
                    df.loc[outliers.index,t] = np.nan
                    df.loc[outliers.index,t] = np.nan

                    if make_plots:
                        _plot_percentiles(k,t,df,var_threshold,upper_thresh,lower_thresh,stid) # AFTER OUTLIER REMOVAL

    # Back to xarray, and re-assign the original attrs
    ds_out = df.to_xarray()
    ds_out = ds_out.assign_attrs(ds.attrs) # Dataset attrs
    for x in ds_out.data_vars: # variable-specific attrs
        ds_out[x].attrs = ds[x].attrs
    # equivalent to above:
    # vals = [xr.DataArray(data=df_out[c], dims=['time'], coords={'time':df_out.index}, attrs=ds[c].attrs) for c in df_out.columns]
    # ds_out = xr.Dataset(dict(zip(df_out.columns, vals)), attrs=ds.attrs)
    return ds_out

def flagNAN(ds_in, 
            flag_url='https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-data-issues/master/flags/',
            flag_dir='local/flags/'):
    '''Read flagged data from .csv file. For each variable, and downstream 
    dependents, flag as invalid (or other) if set in the flag .csv
    
    Parameters
    ----------
    ds_in : xr.Dataset
        Level 0 dataset
    flag_url : str
        URL to directory where .csv flag files can be found
    flag_dir : str
        File directory where .csv flag files can be found
    
    Returns
    -------
    ds : xr.Dataset
        Level 0 data with flagged data
    '''
    ds = ds_in.copy()
    df = None
        
    df = _getDF(flag_url + ds.attrs["station_id"] + ".csv",
                os.path.join(flag_dir, ds.attrs["station_id"] + ".csv"),
                # download = False,  # only for working on draft local flag'n'fix files
                )
    
    if isinstance(df, pd.DataFrame):
        df.t0 = pd.to_datetime(df.t0).dt.tz_localize(None)
        df.t1 = pd.to_datetime(df.t1).dt.tz_localize(None)
        
        if df.shape[0] > 0: 
            for i in df.index:
                t0, t1, avar = df.loc[i,['t0','t1','variable']]
                
                if avar == '*':
                    # Set to all vars if var is "*"
                    varlist = list(ds.keys())
                elif '*' in avar:
                    # Reads as regex if contains "*" and other characters (e.g. 't_i_.*($)')
                    varlist = pd.DataFrame(columns = list(ds.keys())).filter(regex=(avar)).columns
                else:
                    varlist = avar.split() 
         
                if 'time' in varlist: varlist.remove("time")
                
                # Set to all times if times are "n/a"
                if pd.isnull(t0): 
                    t0 = ds['time'].values[0]
                if pd.isnull(t1): 
                    t1 = ds['time'].values[-1]
                
                for v in varlist:
                    if v in list(ds.keys()):
                        print('---> flagging',t0, t1, v)
                        ds[v] = ds[v].where((ds['time'] < t0) | (ds['time'] > t1))
                    else:
                        print('---> could not flag', v,', not in dataset')            
    return ds


def adjustTime(ds, 
               adj_url="https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-data-issues/master/adjustments/", 
               adj_dir='local/adjustments/', 
               var_list=[], skip_var=[]):
    '''Read adjustment data from .csv file. Only applies the "time_shift" adjustment
    
    Parameters
    ----------
    ds : xr.Dataset
        Level 0 dataset
    adj_url : str
        URL to directory where .csv adjustment files can be found
    adj_dir : str
        File directory where .csv adjustment files can be found
    
    Returns
    -------
    ds : xr.Dataset
        Level 0 data with flagged data
    '''
    ds_out = ds.copy()
    adj_info=None
        
    adj_info = _getDF(adj_url + ds.attrs["station_id"] + ".csv",
                      os.path.join(adj_dir, ds.attrs["station_id"] + ".csv"),
                      # download = False,
                      verbose = False)
    
    if isinstance(adj_info, pd.DataFrame):

        
        if "time_shift" in adj_info.adjust_function.values:
            time_shifts = adj_info.loc[adj_info.adjust_function == "time_shift", :]
            # if t1 is left empty, then adjustment is applied until the end of the file
            time_shifts.loc[time_shifts.t1.isnull(), "t1"] = pd.to_datetime(ds_out.time.values[-1]).isoformat()
            time_shifts.t0 = pd.to_datetime(time_shifts.t0).dt.tz_localize(None)
            time_shifts.t1 = pd.to_datetime(time_shifts.t1).dt.tz_localize(None)
            
            for t0, t1, val in zip(
                time_shifts.t0,
                time_shifts.t1,
                time_shifts.adjust_value,
            ):
                ds_shifted = ds_out.sel(time=slice(t0,t1))
                ds_shifted['time'] = ds_shifted.time.values + pd.Timedelta(days = val)
                
                # here we concatenate what was before the shifted part, the shifted
                # part and what was after the shifted part
                # note that if any data was already present in the target period 
                # (where the data lands after the shift), it is overwritten
                
                ds_out = xr.concat(
                                        (
                                            ds_out.sel(time=slice(pd.to_datetime(ds_out.time.values[0]),
                                                                  t0 + pd.Timedelta(days = val))),
                                            ds_shifted,
                                            ds_out.sel(time=slice(t1 + pd.Timedelta(days = val),
                                                                  pd.to_datetime(ds_out.time.values[-1])))
                                        ),
                                        dim = 'time',
                                       )
                if t0 > pd.Timestamp.now():
                    ds_out = ds_out.sel(time=slice(pd.to_datetime(ds_out.time.values[0]),
                                                   t0))
    return ds_out


def adjustData(ds, 
               adj_url="https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-data-issues/master/adjustments/", 
               adj_dir='local/adjustments/', 
               var_list=[], skip_var=[]):
    '''Read adjustment data from .csv file. For each variable, and downstream 
    dependents, adjust data accordingly if set in the adjustment .csv
    
    Parameters
    ----------
    ds : xr.Dataset
        Level 0 dataset
    adj_url : str
        URL to directory where .csv adjustment files can be found
    adj_dir : str
        File directory where .csv adjustment files can be found
    
    Returns
    -------
    ds : xr.Dataset
        Level 0 data with flagged data
    '''
    ds_out = ds.copy()
    adj_info=None 
    adj_info = _getDF(adj_url + ds.attrs["station_id"] + ".csv",
                      os.path.join(adj_dir, ds.attrs["station_id"] + ".csv"),
                      # download = False,  # only for working on draft local flag'n'fix files
                      )
    
    if isinstance(adj_info, pd.DataFrame):
        # removing potential time shifts from the adjustment list
        adj_info = adj_info.loc[adj_info.adjust_function != "time_shift", :]
   
        # if t1 is left empty, then adjustment is applied until the end of the file
        adj_info.loc[adj_info.t0.isnull(), "t0"] = ds_out.time.values[0]
        adj_info.loc[adj_info.t1.isnull(), "t1"] = ds_out.time.values[-1]
        # making all timestamps timezone naive (compatibility with xarray)
        adj_info.t0 = pd.to_datetime(adj_info.t0).dt.tz_localize(None)
        adj_info.t1 = pd.to_datetime(adj_info.t1).dt.tz_localize(None)
            
        # if "*" is in the variable name then we interpret it as regex
        selec =  adj_info['variable'].str.contains('\*') & (adj_info['variable'] != "*")
        for ind in adj_info.loc[selec, :].index:
            line_template = adj_info.loc[ind, :].copy()
            regex = adj_info.loc[ind, 'variable']
            for var in pd.DataFrame(columns = list(ds.keys())).filter(regex=regex).columns:
                line_template.variable = var
                line_template.name = adj_info.index.max() + 1
                adj_info = pd.concat((adj_info, line_template.to_frame().transpose()),axis=0)
            adj_info = adj_info.drop(labels=ind, axis=0)
            
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
            if var not in list(ds_out.keys()):
                print('could not adjust',var,', not in dataset')
                continue
            for t0, t1, func, val in zip(
                adj_info.loc[var].t0,
                adj_info.loc[var].t1,
                adj_info.loc[var].adjust_function,
                adj_info.loc[var].adjust_value,
            ):
                if (t0 > pd.to_datetime(ds_out.time.values[-1])) | (t1 < pd.to_datetime(ds_out.time.values[0])):
                    continue
                print('--->',t0, t1, var, func, val)
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

def calcCloudCoverage(T, T_0, eps_overcast, eps_clear, dlr, station_id):
    '''Calculate cloud cover from T and T_0
    
    Parameters
    ----------
    T : xarray.DataArray
        Air temperature 1
    T_0 : xarray.DataArray
        Air temperature 0
    eps_overcast : int
        Cloud overcast assumption, from Swinbank (1963)
    eps_clear : int
        Cloud clear assumption, from Swinbank (1963)
    dlr : xarray.DataArray
        Downwelling longwave radiation, with array of same length as T and T_0
    station_id : str
        Station ID string, for special cases at selected stations where cloud
        overcast and cloud clear assumptions are pre-defined. Currently
        KAN_M and KAN_U are special cases, but this will need to be done for
        all stations eventually
    
    Returns
    -------
    cc : xarray.DataArray
        Cloud cover data array
    '''
    if station_id == 'KAN_M':
       LR_overcast = 315 + 4*T
       LR_clear = 30 + 4.6e-13 * (T + T_0)**6
    elif station_id == 'KAN_U':
       LR_overcast = 305 + 4*T
       LR_clear = 220 + 3.5*T
    else:
       LR_overcast = eps_overcast * 5.67e-8 *(T + T_0)**4                      
       LR_clear = eps_clear * 5.67e-8 * (T + T_0)**6   
    cc = (dlr - LR_clear) / (LR_overcast - LR_clear)
    cc[cc > 1] = 1
    cc[cc < 0] = 0
    return cc

def calcSurfaceTemperature(T_0, ulr, dlr, emissivity):
    '''Calculate surface temperature from air temperature, upwelling and 
    downwelling radiation and emissivity
    
    Parameters
    ----------
    T_0 : xarray.DataArray
        Air temperature
    ulr : xarray.DataArray
        Upwelling longwave radiation
    dlr : xarray.DataArray
        Downwelling longwave radiation
    emissivity : int
        Assumed emissivity
    
    Returns
    -------
    xarray.DataArray
        Calculated surface temperature
    '''
    t_surf = ((ulr - (1 - emissivity) * dlr) / emissivity / 5.67e-8)**0.25 - T_0
    return t_surf.where(t_surf <= 0, other = 0)
    
def calcTilt(tilt_x, tilt_y, deg2rad):
    '''Calculate station tilt
    
    Parameters
    ----------
    tilt_x : xarray.DataArray
        X tilt inclinometer measurements
    tilt_y : xarray.DataArray
        Y tilt inclinometer measurements
    deg2rad : float
        Degrees to radians conversion
    
    Returns
    -------
    phi_sensor_rad : xarray.DataArray
        Spherical tilt coordinates
    theta_sensor_rad : xarray.DataArray
        Total tilt of sensor, where 0 is horizontal
    '''
    # Tilt as radians
    tx = tilt_x * deg2rad
    ty = tilt_y * deg2rad
    
    # Calculate cartesian coordinates
    X = np.sin(tx) * np.cos(tx) * np.sin(ty)**2 + np.sin(tx) * np.cos(ty)**2
    Y = np.sin(ty) * np.cos(ty) * np.sin(tx)**2 + np.sin(ty) * np.cos(tx)**2
    Z = np.cos(tx) * np.cos(ty) + np.sin(tx)**2 * np.sin(ty)**2
 
    # Calculate spherical coordinates
    phi_sensor_rad = -np.pi /2 - np.arctan(Y/X)
    phi_sensor_rad[X > 0] += np.pi
    phi_sensor_rad[(X == 0) & (Y < 0)] = np.pi
    phi_sensor_rad[(X == 0) & (Y == 0)] = 0
    phi_sensor_rad[phi_sensor_rad < 0] += 2*np.pi

    # Total tilt of the sensor, i.e. 0 when horizontal
    theta_sensor_rad = np.arccos(Z / (X**2 + Y**2 + Z**2)**0.5) 
    # phi_sensor_deg = phi_sensor_rad * rad2deg                                #TODO take these out if not needed
    # theta_sensor_deg = theta_sensor_rad * rad2deg
    return phi_sensor_rad, theta_sensor_rad 

def correctHumidity(rh, T, T_0, T_100, ews, ei0):                        #TODO figure out if T replicate is needed
    '''Correct relative humidity using Groff & Gratch method, where values are
    set when freezing and remain the original values when not freezing

    Parameters
    ----------
    rh : xarray.DataArray
        Relative humidity
    T : xarray.DataArray
        Air temperature
    T_0 : float
        Ice point temperature in K
    T_100 : float
        Steam point temperature in K
    ews : float
        Saturation pressure (normal atmosphere) at steam point temperature
    ei0 : float
        Saturation pressure (normal atmosphere) at ice-point temperature
        
    Returns
    -------
    rh_cor : xarray.DataArray
        Corrected relative humidity
    '''                                            
    # Convert to hPa (Groff & Gratch)   
    e_s_wtr = 10**(-7.90298 * (T_100 / (T + T_0) - 1)
                   + 5.02808 * np.log10(T_100 / (T + T_0)) 
                   - 1.3816E-7 * (10**(11.344 * (1 - (T + T_0) / T_100)) - 1)
                   + 8.1328E-3 * (10**(-3.49149 * (T_100/(T + T_0) - 1)) -1)
                   + np.log10(ews))
    e_s_ice = 10**(-9.09718 * (T_0 / (T + T_0) - 1)
                   - 3.56654 * np.log10(T_0 / (T + T_0))
                   + 0.876793 * (1 - (T + T_0) / T_0)
                   + np.log10(ei0))
    
    # Define freezing point. Why > -100?  
    freezing = (T < 0) & (T > -100).values 

    # Set to Groff & Gratch values when freezing, otherwise just rh                         
    rh_cor = rh.where(~freezing, other = rh*(e_s_wtr / e_s_ice))  
    return rh_cor                               

def correctPrecip(precip, wspd):
    '''Correct precipitation with the undercatch correction method used in 
    Yang et al. (1999) and Box et al. (2022), based on Goodison et al. (1998)
    
    Yang, D., Ishida, S., Goodison, B. E., and Gunther, T.: Bias correction of 
    daily precipitation measurements for Greenland, 
    https://doi.org/10.1029/1998jd200110, 1999.
    
    Box, J., Wehrle, A., van As, D., Fausto, R., Kjeldsen, K., Dachauer, A.,
    Ahlstrom, A. P., and Picard, G.: Greenland Ice Sheet rainfall, heat and 
    albedo feedback imapacts from the Mid-August 2021 atmospheric river, 
    Geophys. Res. Lett. 49 (11), e2021GL097356, 
    https://doi.org/10.1029/2021GL097356, 2022.
    
    Goodison, B. E., Louie, P. Y. T., and Yang, D.: Solid Precipitation 
    Measurement Intercomparison, WMO, 1998
    
    Parameters
    ----------
    precip : xarray.DataArray
        Cumulative precipitation measurements
    wspd : xarray.DataArray
        Wind speed measurements
    
    Returns
    -------
    precip_cor : xarray.DataArray
        Cumulative precipitation corrected
    precip_rate : xarray.DataArray
        Precipitation rate corrected
    '''
    # Calculate undercatch correction factor
    corr=100/(100.00-4.37*wspd+0.35*wspd*wspd)

    # Fix all values below 1.02 to 1.02
    corr = corr.where(corr>1.02, other=1.02)

    # Fill nan values in precip with preceding value                                           
    precip = precip.ffill(dim='time')   
    
    # Calculate precipitation rate
    precip_rate = precip.diff(dim='time', n=1)
    
    # Apply correction to rate
    precip_rate = precip_rate*corr
    
    # Flag rain bucket reset
    precip_rate = precip_rate.where(precip_rate>-0.01, other=np.nan) 
    b = precip_rate.to_dataframe('precip_flag').notna().to_xarray()
    
    # Get corrected cumulative precipitation, reset if rain bucket flag
    precip_cor = precip_rate.cumsum()-precip_rate.cumsum().where(~b['precip_flag']).ffill(dim='time').fillna(0).astype(float)
       
    return precip_cor, precip_rate
                                  
def calcDeclination(doy, hour, minute):
    '''Calculate sun declination based on time
    
    Parameters
    ----------
    doy : int
        Day of year
    hour : int
        Hour of day
    minute : int
        Minute of hour
    
    Returns
    -------
    float
        Sun declination
    '''
    d0_rad = 2 * np.pi * (doy + (hour + minute / 60) / 24 -1) / 365
    return np.arcsin(0.006918 - 0.399912
                     * np.cos(d0_rad) + 0.070257
                     * np.sin(d0_rad) - 0.006758
                     * np.cos(2 * d0_rad) + 0.000907
                     * np.sin(2 * d0_rad) - 0.002697
                     * np.cos(3 * d0_rad) + 0.00148
                     * np.sin(3 * d0_rad))

def calcHourAngle(hour, minute, lon):
    '''Calculate hour angle of sun based on time and longitude. Make sure that
    time is set to UTC and longitude is positive when west. Hour angle should
    be 0 at noon
    
    Parameters
    ----------
    hour : int
        Hour of day
    minute : int
        Minute of hour
    lon : float
        Longitude
    
    Returns
    -------
    float
        Hour angle of sun
    '''
    return 2 * np.pi * (((hour + minute / 60) / 24 - 0.5) - lon/360)
     # ; - 15.*timezone/360.)

def calcDirectionDeg(HourAngle_rad):                                          #TODO remove if not plan to use this
    '''Calculate sun direction as degrees. This is an alternative to 
    _calcHourAngle that is currently not implemented into the offical L0>>L3
    workflow. Here, 180 degrees is at noon (NH), as opposed to HourAngle
    
    Parameters
    ----------
    HourAngle_rad : float
        Sun hour angle in radians
    
    Returns
    -------
    DirectionSun_deg
        Sun direction in degrees
    '''
    DirectionSun_deg = HourAngle_rad * 180/np.pi - 180
    DirectionSun_deg[DirectionSun_deg < 0] += 360
    DirectionSun_deg[DirectionSun_deg < 0] += 360
    return DirectionSun_deg

def calcZenith(lat, Declination_rad, HourAngle_rad, deg2rad, rad2deg):
    '''Calculate sun zenith in radians and degrees
    
    Parameters
    ----------
    lat : float
        Latitude
    Declination_Rad : float
        Sun declination in radians
    HourAngle_rad : float
        Sun hour angle in radians
    deg2rad : float
        Degrees to radians conversion
    rad2deg : float
        Radians to degrees conversion
    
    Returns
    -------
    ZenithAngle_rad : float
        Zenith angle in radians
    ZenithAngle_deg : float
        Zenith angle in degrees
    '''
    ZenithAngle_rad = np.arccos(np.cos(lat * deg2rad)
                                * np.cos(Declination_rad)
                                * np.cos(HourAngle_rad)
                                + np.sin(lat * deg2rad)
                                * np.sin(Declination_rad))
    
    ZenithAngle_deg = ZenithAngle_rad * rad2deg   
    return ZenithAngle_rad, ZenithAngle_deg

def calcAngleDiff(ZenithAngle_rad, HourAngle_rad, phi_sensor_rad, 
                  theta_sensor_rad):
    '''Calculate angle between sun and upper sensor (to determine when sun is
    in sight of upper sensor
    
    Parameters
    ----------
    ZenithAngle_rad : float
        Zenith angle in radians
    HourAngle_rad : float 
        Sun hour angle in radians
    phi_sensor_rad : xarray.DataArray
        Spherical tilt coordinates
    theta_sensor_rad : xarray.DataArray
        Total tilt of sensor, where 0 is horizontal
    
    Returns
    -------
    float
        Angle between sun and sensor
    '''
    return 180 / np.pi * np.arccos(np.sin(ZenithAngle_rad) 
                                   * np.cos(HourAngle_rad + np.pi)
                                   * np.sin(theta_sensor_rad)
                                   * np.cos(phi_sensor_rad)
                                   + np.sin(ZenithAngle_rad)
                                   * np.sin(HourAngle_rad + np.pi)
                                   * np.sin(theta_sensor_rad)
                                   * np.sin(phi_sensor_rad)
                                   + np.cos(ZenithAngle_rad)
                                   * np.cos(theta_sensor_rad))  

def calcAlbedo(usr, dsr_cor, AngleDif_deg, ZenithAngle_deg):
    '''Calculate surface albedo based on upwelling and downwelling shorwave 
    flux, the angle between the sun and sensor, and the sun zenith
    
    Parameters
    ----------
    usr : xarray.DataArray
        Upwelling shortwave radiation
    dsr_cor : xarray.DataArray
        Downwelling shortwave radiation corrected
    AngleDif_def : float
        Angle between sun and sensor in degrees
    ZenithAngle_deg: float
        Zenith angle in degrees
    
    Returns
    -------
    albedo : xarray.DataArray
        Derived albedo
    OKalbedos : xarray.DataArray
        Valid albedo measurements
    '''
    albedo = usr / dsr_cor    
    
    # NaN bad data
    OKalbedos = (AngleDif_deg < 70) & (ZenithAngle_deg < 70) & (albedo < 1) & (albedo > 0)    
    albedo[~OKalbedos] = np.nan             
    
    # Interpolate all. Note "use_coordinate=False" is used here to force 
    # comparison against the GDL code when that is run with *only* a TX file. 
    # Should eventually set to default (True) and interpolate based on time, 
    # not index.                                           
    albedo = albedo.interpolate_na(dim='time', use_coordinate=False)           
    albedo = albedo.ffill(dim='time').bfill(dim='time')                        #TODO remove this line and one above?
    return albedo, OKalbedos
    
def calcTOA(ZenithAngle_deg, ZenithAngle_rad):
    '''Calculate incoming shortwave radiation at the top of the atmosphere,
    accounting for sunset periods
    
    Parameters
    ----------
    ZenithAngle_deg : float
        Zenith angle in degrees
    ZenithAngle_rad : float
        Zenith angle in radians
    
    Returns
    -------
    isr_toa : float
        Incoming shortwave radiation at the top of the atmosphere
    '''
    sundown = ZenithAngle_deg >= 90
    
    # Incoming shortware radiation at the top of the atmosphere
    isr_toa = 1372 * np.cos(ZenithAngle_rad) 
    isr_toa[sundown] = 0
    return isr_toa 
    
def calcCorrectionFactor(Declination_rad, phi_sensor_rad, theta_sensor_rad, 
                          HourAngle_rad, ZenithAngle_rad, ZenithAngle_deg, 
                          lat, DifFrac, deg2rad):    
    '''Calculate correction factor for direct beam radiation, as described 
    here: http://solardat.uoregon.edu/SolarRadiationBasics.html
    
    Offset correction (where solar zenith angles larger than 110 degrees) not
    implemented as it should not improve the accuracy of well-calibrated
    instruments. It would go something like this:
    ds['dsr'] = ds['dsr'] - ds['dwr_offset']
    SRout = SRout - SRout_offset
    
    Parameters
    ----------
    Declination_rad : float
        Declination in radians
    phi_sensor_rad : xarray.DataArray
        Spherical tilt coordinates
    theta_sensor_rad : xarray.DataArray
        Total tilt of sensor, where 0 is horizontal
    HourAngle_rad : float 
        Sun hour angle in radians
    ZenithAngle_rad : float 
        Zenith angle in radians
    ZenithAngle_deg : float 
        Zenith Angle in degrees
    lat :  float
        Latitude
    DifFrac : float
        Fractional cloud cover
    deg2rad : float
        Degrees to radians conversion
        
    Returns
    -------
    CorFac_all : xarray.DataArray
        Correction factor
    '''
    CorFac = np.sin(Declination_rad) * np.sin(lat * deg2rad) \
        * np.cos(theta_sensor_rad) \
        - np.sin(Declination_rad) \
        * np.cos(lat * deg2rad) \
        * np.sin(theta_sensor_rad) \
        * np.cos(phi_sensor_rad + np.pi) \
        + np.cos(Declination_rad) \
        * np.cos(lat * deg2rad) \
        * np.cos(theta_sensor_rad) \
        * np.cos(HourAngle_rad) \
        + np.cos(Declination_rad) \
        * np.sin(lat * deg2rad) \
        * np.sin(theta_sensor_rad) \
        * np.cos(phi_sensor_rad + np.pi) \
        * np.cos(HourAngle_rad) \
        + np.cos(Declination_rad) \
        * np.sin(theta_sensor_rad) \
        * np.sin(phi_sensor_rad + np.pi) \
        * np.sin(HourAngle_rad) \
    
    CorFac = np.cos(ZenithAngle_rad) / CorFac
    # sun out of field of view upper sensor
    CorFac[(CorFac < 0) | (ZenithAngle_deg > 90)] = 1
    
    # Calculating ds['dsr'] over a horizontal surface corrected for station/sensor tilt
    CorFac_all = CorFac / (1 - DifFrac + CorFac * DifFrac)
    
    return CorFac_all

def _getDF(flag_url, flag_file, download=True, verbose=True):
    '''Get dataframe from flag or adjust file. First attempt to retrieve from 
    URL. If this fails then attempt to retrieve from local file 
    
    Parameters
    ----------
    flag_url : str
        URL address to file
    flag_file : str
        Local path to file
    download : bool
        Flag to download file from URL
    
    Returns
    -------
    df : pd.DataFrame
        Flag or adjustment dataframe
    '''
     
    # Download local copy as csv
    if download==True:
        os.makedirs(os.path.dirname(flag_file), exist_ok = True)  
        
        try:
            urllib.request.urlretrieve(flag_url, flag_file)
            if verbose: print('Downloaded a',
                              flag_file.split('/')[-2][:-1],
                              f'file to {flag_file}')
        except (HTTPError, URLError) as e:
            if verbose: print('Unable to download',
                              flag_file.split('/')[-2][:-1],
                              f'file, using local file: {flag_file}')
    else:
        if verbose: print('Using local',
                          flag_file.split('/')[-2][:-1],
                          f'file: {flag_file}')

    if os.path.isfile(flag_file):
        df = pd.read_csv(
                        flag_file, 
                        comment="#", 
                        skipinitialspace=True,
                        ).dropna(how='all', axis='rows') 
    else:
        df=None
        if verbose: print('No', flag_file.split('/')[-2][:-1], 'file to read.')
    return df


def _hampel(vals_orig, k=7*24, t0=3):
    '''Hampel filter
    
    Parameters
    ----------
    vals : pd.DataSeries
        Series of values from which to remove outliers
    k : int
        Size of window, including the sample. For example, 7 is equal to 3 on 
        either side of value. The default is 7*24.
    t0 : int
        Threshold value. The default is 3.
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

def _checkSunPos(ds, OKalbedos, sundown, sunonlowerdome, TOA_crit_nopass):
    '''Check sun position
    
    Parameters
    ----------
    ds : xarray.Dataset
        Data set
    OKalbedos : xarray.DataArray
        Valid measurements flag
    sundown : xarray.DataArray
        Sun below horizon flag
    sunonlowerdome : xarray.DataArray
        Sun in view of lower sensor flag
    TOA_crit_nopass : xarray.DataArray
        Top-of-Atmosphere flag
    '''
    valid = (~(ds['dsr_cor'].isnull())).sum()
    print('Sun in view of upper sensor / workable albedos:', OKalbedos.sum().values,
          (100*OKalbedos.sum()/valid).round().values, "%")
    print('Sun below horizon:', sundown.sum(),
          (100*sundown.sum()/valid).round().values, "%")
    print('Sun in view of lower sensor:', sunonlowerdome.sum().values,
          (100*sunonlowerdome.sum()/valid).round().values, "%")
    print('Spikes removed using TOA criteria:', TOA_crit_nopass.sum().values,
          (100*TOA_crit_nopass.sum()/valid).round().values, "%")
    print('Mean net SR change by corrections:',
          (ds['dsr_cor']-ds['usr_cor']-ds['dsr']+ds['usr']).sum().values/valid.values,
          "W/m2")
    
def _getTempK(T_0):                                                            #TODO same as L2toL3._getTempK()
    '''Return steam point temperature in Kelvins
    
    Parameters
    ----------
    T_0 : float
        Ice point temperature in K
    
    Returns
    -------
    float
        Steam point temperature in K'''
    return T_0+100

def _getRotation():                                                            #TODO same as L2toL3._getRotation()
    '''Return degrees-to-radians and radians-to-degrees''' 
    deg2rad = np.pi / 180
    rad2deg = 1 / deg2rad
    return deg2rad, rad2deg

def _plot_percentiles_t(k, t, df, season_dfs, var_threshold, stid):
    '''Plot data and percentile thresholds for air temp (seasonal)'''
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,12))
    inst_var = t.split('_')[0] + '_i'
    if inst_var in df:
        i_plot = df[inst_var]
        plt.scatter(df.index,i_plot, color='orange', s=3, label='t_i instantaneuous')
    if t in ('t_u','t_l'):
        plt.scatter(df.index,df[t], color='b', s=3, label=f'{t} hourly ave')
    for x1,x2 in zip([1,2,3,4], season_dfs):
        y1 = np.full(len(x2.index), (var_threshold[k]['seasons'][x1]['lo'] - var_threshold[k]['limit']))
        y2 = np.full(len(x2.index), (var_threshold[k]['seasons'][x1]['hi'] + var_threshold[k]['limit']))
        y11 = np.full(len(x2.index), (var_threshold[k]['seasons'][x1]['lo'] ))
        y22 = np.full(len(x2.index), (var_threshold[k]['seasons'][x1]['hi'] ))
        plt.scatter(x2.index, y1, color='r',s=1)
        plt.scatter(x2.index, y2, color='r', s=1)
        plt.scatter(x2.index, y11, color='k', s=1)
        plt.scatter(x2.index, y22, color='k', s=1)
    plt.title('{} {}'.format(stid, t))
    plt.legend(loc="lower left")
    plt.show()

def _plot_percentiles(k, t, df, var_threshold, upper_thresh, lower_thresh, stid):
    '''Plot data and percentile thresholds'''
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,12))
    inst_var = t.split('_')[0] + '_i'
    if inst_var in df:
        if k == 'p_u':
            i_plot = (df[inst_var]+1000.)
        else:
            i_plot = df[inst_var]
        plt.scatter(df.index,i_plot, color='orange', s=3, label='instantaneuous')
    if t != inst_var:
        plt.scatter(df.index,df[t], color='b', s=3, label=f' {t} hourly ave')
    plt.axhline(y=upper_thresh, color='r', linestyle='-')
    plt.axhline(y=lower_thresh, color='r', linestyle='-')
    plt.axhline(y=var_threshold[k]['hi'], color='k', linestyle='--')
    plt.axhline(y=var_threshold[k]['lo'], color='k', linestyle='--')
    plt.title('{} {}'.format(stid, t))
    plt.legend(loc="lower left")
    plt.show()


if __name__ == "__main__": 
    # unittest.main() 
    pass 
