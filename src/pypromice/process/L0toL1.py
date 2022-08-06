#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import xarray as xr
import logging
import re
# logging.basicConfig(format="{asctime} : ({filename}:{lineno}) : {message} ", style="{")

def toL1(L0, vars_df, col='station_type', T_0=273.15, tilt_threshold=-100):
    '''Process one Level 0 (L0) product to Level 1

    Parameters
    ----------
    L0 : xarray.Dataset
        Level 0 dataset
    var_df : pandas.DataFrame
        Variables look-up table dataframe
    cols : list, optional
        Variables look-up table column names of interest. The default is 
        ["lo","hi","OOL"]
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

    ds['time_orig'] = ds['time']                                               # Check and shift time
    ds['time'] = _addTimeShift(ds['time'], ds.attrs['format'])
    _, index = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index)

    # ds['t_2'] = ds['t_2'] - ds.attrs['hygroclip_t_offset']                    # No hydroclip offset needed
    
    ds['dsr'] = (ds['dsr'] * 10) / ds.attrs['dsr_eng_coef']                    # Convert radiation from engineering to physical units
    ds['usr'] = (ds['usr'] * 10) / ds.attrs['usr_eng_coef']
    ds['dlr'] = ((ds['dlr'] * 10) / ds.attrs['dlr_eng_coef']) + 5.67E-8*(ds['t_rad'] + T_0)**4
    ds['ulr'] = ((ds['ulr'] * 10) / ds.attrs['ulr_eng_coef']) + 5.67E-8*(ds['t_rad'] + T_0)**4

    ds['z_boom_u'] = _reformatArray(ds['z_boom_u'])                            # Reformat boom height
    ds['z_boom_u'] = ds['z_boom_u'] * ((ds['t_u'] + T_0)/T_0)**0.5             # Adjust sonic ranger readings for sensitivity to air temperature       
    
    if ds['gps_lat'].dtype.kind == 'O':                                        # Decode and reformat GPS information
        assert('NH' in ds['gps_lat'].dropna(dim='time').values[0])
        ds = _decodeGPS(ds, ['gps_lat','gps_lon','gps_time'])
    for l in ['gps_lat', 'gps_lon', 'gps_time']:
        ds[l] = _reformatArray(ds[l])  

    ds['gps_lat'] = _reformatGPS(ds['gps_lat'], ds.attrs['latitude'])
    ds['gps_lon'] = _reformatGPS(ds['gps_lon'], ds.attrs['longitude'])

    # if ds.attrs['format'] != 'TX':                                             # Convert tilt voltage to degrees
    ds['tilt_x'] = _getTiltDegrees(ds['tilt_x'], 7)
    ds['tilt_y'] = _getTiltDegrees(ds['tilt_y'], 7)
  
    if hasattr(ds, 'tilt_y_factor'):                                           # Apply tilt factor (e.g. -1 will invert tilt angle)
        ds['tilt_y'] = ds['tilt_y']*ds.attrs['tilt_y_factor']

    ds['tilt_x']  = _filterTilt(ds['tilt_x'], tilt_threshold)                  # Filter tilt 
    ds['tilt_y']  = _filterTilt(ds['tilt_y'], tilt_threshold)                  # TODO check tilt_y inversion +ive to -ive for Gc-Net stations

    ds['wdir_u'] = ds['wdir_u'].where(ds['wspd_u'] != 0)                       # Get directional wind speed                    
    ds['wspd_x_u'], ds['wspd_y_u'] = _calcWindDir(ds['wspd_u'], ds['wdir_u']) 
    if ds.attrs['number_of_booms']==1:                                         # 1-boom processing
        if ~ds['z_pt'].isnull().all():                                         # Calculate pressure transducer fluid density                                           
            ds['z_pt_cor'],ds['z_pt']=_getPressDepth(ds['z_pt'], ds['p_u'], 
                                                     ds.attrs['pt_antifreeze'], 
                                                     ds.attrs['pt_z_factor'], 
                                                     ds.attrs['pt_z_coef'], 
                                                     ds.attrs['pt_z_p_coef'])
            ds['z_pt_cor'].attrs['long_name'] = ds['z_pt'].long_name + ' corrected'         

        names = vars_df.loc[vars_df[col] != 'two-boom']
        for v in list(names.index):
            if v not in list(ds.variables):
                ds[v] = (('time'), np.arange(ds['time'].size)*np.nan)
        ds = _removeVars(ds, ['gps_geounit', 'min_y'])                         # Remove redundant variables
            
    elif ds.attrs['number_of_booms']==2:                                       # 2-boom processing
        ds['z_boom_l'] = _reformatArray(ds['z_boom_l'])                        # Reformat boom height    
        ds['z_boom_l'] = ds['z_boom_l'] * ((ds['t_l'] + T_0)/T_0)**0.5         # Adjust sonic ranger readings for sensitivity to air temperature
        ds['wdir_l'] = ds['wdir_l'].where(ds['wspd_l'] != 0)                   # Get directional wind speed    
        ds['wspd_x_l'], ds['wspd_y_l'] = _calcWindDir(ds['wspd_l'], ds['wdir_l'])
        names = vars_df.loc[vars_df[col] != 'one-boom']
        for v in list(names.index):
            if v not in list(ds.variables):
                ds[v] = (('time'), np.arange(ds['time'].size)*np.nan) 
     
        if ~ds['msg_i'].isnull().all():                                            # Instantaneous msg processing
            ds['wdir_i'] = ds['wdir_i'].where(ds['wspd_i'] != 0)                   # Get directional wind speed                    
            ds['wspd_x_i'], ds['wspd_y_i'] = _calcWindDir(ds['wspd_i'], ds['wdir_i'])   

    return ds

def _flagNAN(ds):
    '''Read flagged data from .csv file. For each variable, and downstream 
    dependents, flag as invalid (or other) if set in the flag .csv'''
    flag_file = "./data/flags/" + ds.attrs["station_id"] + ".csv"
    
    if not os.path.isfile(flag_file):
        print(f'Flag file {flag_file} not found - no data flagged')
        return ds # no flag file
    
    df = pd.read_csv(flag_file, parse_dates=[0,1], comment="#") \
           .dropna(how='all', axis='rows')
    
    # Check format of flags.csv. Either both or neither t0 and t1 must be defined
    assert(((np.isnan(df['t0'].values).astype(int) + np.isnan(df['t1'].values).astype(int)) % 2).sum() == 0)
    
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
        if pd.isnull(t0): t0, t1 = ds['time'].values[[0,-1]]
        
        for v in varlist:
            ds[v] = ds[v].where((ds['time'] < t0) | (ds['time'] > t1))
        
        # TODO: Mark these values in the ds_flags dataset using perhaps 
        # flag_LUT.loc["NAN"]['value']
    return ds

def _addTimeShift(ds, fmt):
    '''Adjust times based on file format. For raw (10 min), values are sampled 
    instantaneously so don't call this function. For STM (1 hour), values are 
    averaged and timestamp is end so times are shifted by 1 h earlier to 
    beginning. For TX (some 10 min, some 1 hour, some 1 day?) then time is 
    shifted appropriately.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to apply time shift to
    fmt : str
        Format string to define time shift. "raw": No adjust (timestamp is at 
        start of period). "STM": Adjust timestamp from end of period to start 
        of period. "TX": Adjust timestamp start of period (hour/day) also 
        depending on season
    
    Returns
    -------
    t : xarray.DataArray
        Data array with shifted times
    '''    
    if fmt == 'raw':
        t = ds['time']                                                         # NOTE: The following line re-implements bug: https://github.com/GEUS-PROMICE/AWS_v3/issues/2
        # t = (ds['time'] + pd.to_timedelta("-24 hours"))\                     # See also https://github.com/GEUS-PROMICE/PROMICE-AWS-processing/issues/20
        #     .where((ds['time'].dt.hour == 23) &
        #            ((ds['time'].dt.dayofyear <= 300) &
        #             (ds['time'].dt.dayofyear >= 100)),
        #            other=ds['time'])
    elif fmt == 'STM':
        t = ds['time'] + pd.to_timedelta("-1 hour")
    elif fmt == 'TX':
        diff = ds['time'].diff(dim='time')
        diffarr = diff.values.astype('timedelta64[h]').astype(int)
        diffarr = np.append(0, diffarr) 
        t = (ds['time'] + pd.to_timedelta("-1 hour"))\
            .where(# (diffarr == 1) &
                   (ds['time'].dt.dayofyear <= 300) &
                   (ds['time'].dt.dayofyear >= 100),
                   other=ds['time'])
    else:
        t = ds['time']
    return t

def _removeVars(ds, v_names):
    '''Remote redundant variables if present in dataset'''
    for v in v_names:
        if v in list(ds.variables): ds = ds.drop_vars(v)
    return ds

def _getPressDepth(z_pt, p, pt_antifreeze, pt_z_factor, pt_z_coef, pt_z_p_coef): 
    '''Adjust pressure depth and calculate pressure transducer depth based on 
    pressure transducer fluid density'''
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

def _getTiltDegrees(tilt, win_size):
    '''Convert tilt-o-meter voltage to degrees. This should be implemented
    on all messages not transmitted.
    
    IDL translation:
    tiltX = smooth(tiltX,7,/EDGE_MIRROR,MISSING=-999) & tiltY = smooth(tiltY,7,/EDGE_MIRROR, MISSING=-999)
    endif
    
    In Python, this should be 
    dstxy = dstxy.rolling(time=7, win_type='boxcar', center=True).mean()
    But the /EDGE_MIRROR makes it a bit more complicated
    '''
    s = int(win_size/2)
    tdf = tilt.to_dataframe()
    return (('time'), tdf.iloc[:s][::-1] \
            .append(tdf) \
            .append(tdf.iloc[-s:][::-1]) \
            .rolling(win_size, win_type='boxcar', center=True) \
            .mean()[s:-s] \
            .values \
            .flatten())

def _filterTilt(tilt, threshold):
    '''Filter tilt with given threshold'''
    
# # notOKtiltX = where(tiltX lt -100, complement=OKtiltX) & notOKtiltY = where(tiltY lt -100, complement=OKtiltY)
    notOKtilt = (tilt < threshold)
    OKtilt = (tilt >= threshold)
    
    # tiltX = tiltX/10.
    tilt = tilt / 10
    
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


    ### NOTE / WARNING / TODO / BUG
    ## Filling w/o considering time gaps to re-create IDL/GDL outputs
    ## Should fill with coordinate not False. Also consider 'max_gap' option?
    return dst.interpolate_na(dim='time', use_coordinate=False)

def _calcWindDir(wspd, wdir, deg2rad=np.pi/180):
    '''Calculate directional wind speed from wind speed and direction'''        
    wspd_x = wspd * np.sin(wdir * deg2rad)
    wspd_y = wspd * np.cos(wdir * deg2rad) 
    return wspd_x, wspd_y
    
def _decodeGPS(ds, gps_names):
    '''Decode GPS information based on names of GPS attributes. This should be 
    applied if gps information does not consist of float values'''
    for v in gps_names:
        a = ds[v].attrs    
        str2nums = [re.findall(r"[-+]?\d*\.\d+|\d+", _) if isinstance(_, str) else [np.nan] for _ in ds[v].values]
        ds[v][:] = pd.DataFrame(str2nums).astype(float).T.values[0]
        ds[v] = ds[v].astype(float)
        ds[v].attrs = a 
    return ds

def _reformatArray(ds_arr):
    '''Reformat DataArray values and attributes'''
    a = ds_arr.attrs                                                           # Store
    ds_arr.values = pd.to_numeric(ds_arr, errors='coerce')
    ds_arr.attrs = a                                                           # Reformat
    return ds_arr   

def _reformatGPS(pos_arr, attrs):
    '''Correct position if only recorded minutes (and not degrees), and 
    reformat values and attributes'''       
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