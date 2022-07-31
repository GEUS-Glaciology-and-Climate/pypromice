#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import xarray as xr
import logging
import re
# logging.basicConfig(format="{asctime} : ({filename}:{lineno}) : {message} ", style="{")

def toL1(L0, v_file='./variables.csv', T_0=273.15):
    '''Process one Level 0 (L0) product to Level 1

    Parameters
    ----------
    L0 : xarray.Dataset
        Level 0 dataset

    Returns
    -------
    ds : xarray.Dataset
        Level 1 dataset
    '''

    assert(type(L0) == xr.Dataset)
    ds = L0
    ds['n'] = (('time'), np.arange(ds['time'].size)+1)

    ds = _flagNAN(ds)
    ds = _addBasicMeta(ds, v_file)
    
    # Create variables and fill in missing
    df = vartable()[['lo','hi','OOL']]                                         #TODO take this from attributes.getVars() function
    for v in df.index:
        if v not in list(ds.variables):
            ds[v] = (('time'), np.arange(ds['time'].size)*np.nan)
    
    # Station type checks
    if ~ds['z_pt'].isnull().all(): 
        assert("pt_antifreeze" in ds.attrs.keys())
    if 't_2' in list(ds.variables): 
        assert("hygroclip_t_offset" in ds.attrs.keys())
        
    # Calculate pressure transducer fluid density
    if ~ds['z_pt'].isnull().all():                                             #TODO: Implement function w/ reference (analytical or from LUT)
        if ds.attrs['pt_antifreeze'] == 50:                                    #TODO: Track uncertainty
            rho_af = 1092
        elif ds.attrs['pt_antifreeze'] == 100:
            rho_af = 1145
        else:
            rho_af = np.nan
            print("ERROR: Incorrect metadata: 'pt_antifreeze =' ", ds.attrs['pt_antifreeze'])
            print("Antifreeze mix only supported at 50 % or 100%")
            # assert(False)
    
    # Remove redundant variables if present
    for v in ['gps_geounit','min_y']:
        if v in list(ds.variables): ds = ds.drop_vars(v)

    # Clean z data
    for v in ['z_boom']:
        a = ds[v].attrs                                                        # Store
        ds[v].values = pd.to_numeric(ds[v], errors='coerce')
        ds[v].attrs = a                                                        # Restore

    # Check and shift time
    ds['time_orig'] = ds['time']
    ds['time'] = _addTimeShift(ds['time'], ds.attrs['format'])
    _, index = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index)

    # Remove HygroClip temperature offset
    ds['t_2'] = ds['t_2'] - ds.attrs['hygroclip_t_offset']
    
    # Convert radiation from engineering to physical units
    ds['dsr'] = (ds['dsr'] * 10) / ds.attrs['dsr_eng_coef']
    ds['usr'] = (ds['usr'] * 10) / ds.attrs['usr_eng_coef']
    ds['dlr'] = ((ds['dlr'] * 10) / ds.attrs['dlr_eng_coef']) + 5.67E-8*(ds['t_rad'] + T_0)**4
    ds['ulr'] = ((ds['ulr'] * 10) / ds.attrs['ulr_eng_coef']) + 5.67E-8*(ds['t_rad'] + T_0)**4
    
    # Adjust sonic ranger readings for sensitivity to air temperature
    ds['z_boom'] = ds['z_boom'] * ((ds['t_1'] + T_0)/T_0)**0.5
    ds['z_stake'] = ds['z_stake'] * ((ds['t_1'] + T_0)/T_0)**0.5
    
    # Adjust pressure transducer due to fluid properties
    if ~ds['z_pt'].isnull().all():                                             #TODO is this if statement needed?
        
        # Correct pressure depth
        ds['z_pt_cor'] = ds['z_pt'] * ds.attrs['pt_z_coef'] * ds.attrs['pt_z_factor'] \
            * 998.0 / rho_af + 100 * (ds.attrs['pt_z_p_coef'] - ds['p']) / (rho_af * 9.81)
        ds['z_pt_cor'].attrs['long_name'] = ds['z_pt'].long_name + " corrected"

        # Calculate pressure transducer depth
        ds['z_pt'] = ds['z_pt'] * ds.attrs['pt_z_coef'] * ds.attrs['pt_z_factor'] * 998.0 / rho_af
      
    # Decode GPS
    if ds['gps_lat'].dtype.kind == 'O':                                        # Not a float. Probably has "NH"
        assert('NH' in ds['gps_lat'].dropna(dim='time').values[0])
        for v in ['gps_lat','gps_lon','gps_time']:
            a = ds[v].attrs                                                    # Store
            str2nums = [re.findall(r"[-+]?\d*\.\d+|\d+", _) if isinstance(_, str) else [np.nan] for _ in ds[v].values]
            ds[v][:] = pd.DataFrame(str2nums).astype(float).T.values[0]
            ds[v] = ds[v].astype(float)
            ds[v].attrs = a                                                    # Restore

    for v in ['gps_lat','gps_lon','gps_time']:
        a = ds[v].attrs                                                        # Store
        ds[v].values = pd.to_numeric(ds[v], errors='coerce')
        ds[v].attrs = a                                                        # Restore
    
    # Correct position if only recorded minutes (and not degrees)        
    if np.any((ds['gps_lat'] <= 90) & (ds['gps_lat'] > 0)):  
        px = ds.attrs['longitude']
        py = ds.attrs['latitude']
        ds['gps_lat'] = ds['gps_lat'] + 100*py
    if np.any((ds['gps_lon'] <= 90) & (ds['gps_lon'] > 0)):
        ds['gps_lon'] = ds['gps_lon'] + 100*px
            
    for v in ['gps_lat','gps_lon']:
        a = ds[v].attrs                                                        # Store
        ds[v] = np.floor(ds[v] / 100) + (ds[v] / 100 - np.floor(ds[v] / 100)) * 100 / 60
        ds[v].attrs = a                                                        # Restore
    
    # Convert tilt-o-meter voltage to degrees
    # if transmitted ne 'yes' then begin
    #    tiltX = smooth(tiltX,7,/EDGE_MIRROR,MISSING=-999) & tiltY = smooth(tiltY,7,/EDGE_MIRROR, MISSING=-999)
    # endif
    
    # Should just be:
    # if ds.attrs['format'] != 'TX': 
    #   dstxy = dstxy.rolling(time=7, win_type='boxcar', center=True).mean()
    # but the /EDGE_MIRROR makes it a bit more complicated...
    if ds.attrs['format'] != 'TX':
        win_size=7
        s = int(win_size/2)
        tdf = ds['tilt_x'].to_dataframe()
        ds['tilt_x'] = (('time'), tdf.iloc[:s][::-1]\
                        .append(tdf)\
                        .append(tdf.iloc[-s:][::-1])\
                        .rolling(win_size, win_type='boxcar', center=True)\
                        .mean()[s:-s]\
                        .values\
                        .flatten())
        tdf = ds['tilt_y'].to_dataframe()
        ds['tilt_y'] = (('time'), tdf.iloc[:s][::-1]\
                        .append(tdf)\
                        .append(tdf.iloc[-s:][::-1])\
                        .rolling(win_size, win_type='boxcar', center=True)\
                        .mean()[s:-s]\
                        .values\
                        .flatten())
    
    # # notOKtiltX = where(tiltX lt -100, complement=OKtiltX) & notOKtiltY = where(tiltY lt -100, complement=OKtiltY)
    notOKtiltX = (ds['tilt_x'] < -100)
    OKtiltX = (ds['tilt_x'] >= -100)
    notOKtiltY = (ds['tilt_y'] < -100)
    OKtiltY = (ds['tilt_y'] >= -100)
    
    # tiltX = tiltX/10.
    ds['tilt_x'] = ds['tilt_x'] / 10
    ds['tilt_y'] = ds['tilt_y'] / 10
    
    # tiltnonzero = where(tiltX ne 0 and tiltX gt -40 and tiltX lt 40)
    # if n_elements(tiltnonzero) ne 1 then tiltX[tiltnonzero] = tiltX[tiltnonzero]/abs(tiltX[tiltnonzero])*(-0.49*(abs(tiltX[tiltnonzero]))^4 + 3.6*(abs(tiltX[tiltnonzero]))^3 - 10.4*(abs(tiltX[tiltnonzero]))^2 +21.1*(abs(tiltX[tiltnonzero])))
    
    # tiltY = tiltY/10.
    # tiltnonzero = where(tiltY ne 0 and tiltY gt -40 and tiltY lt 40)
    # if n_elements(tiltnonzero) ne 1 then tiltY[tiltnonzero] = tiltY[tiltnonzero]/abs(tiltY[tiltnonzero])*(-0.49*(abs(tiltY[tiltnonzero]))^4 + 3.6*(abs(tiltY[tiltnonzero]))^3 - 10.4*(abs(tiltY[tiltnonzero]))^2 +21.1*(abs(tiltY[tiltnonzero])))
    
    dstx = ds['tilt_x']
    nz = (dstx != 0) & (np.abs(dstx) < 40)
    dstx = dstx.where(~nz, other = dstx / np.abs(dstx)
                      * (-0.49
                         * (np.abs(dstx))**4 + 3.6
                         * (np.abs(dstx))**3 - 10.4
                         * (np.abs(dstx))**2 + 21.1
                         * (np.abs(dstx))))
    ds['tilt_x'] = dstx
    
    dsty = ds['tilt_y']
    nz = (dsty != 0) & (np.abs(dsty) < 40)
    dsty = dsty.where(~nz, other = dsty / np.abs(dsty)
                      * (-0.49
                         * (np.abs(dsty))**4 + 3.6
                         * (np.abs(dsty))**3 - 10.4
                         * (np.abs(dsty))**2 + 21.1
                         * (np.abs(dsty))))
    ds['tilt_y'] = dsty
    
    # if n_elements(OKtiltX) gt 1 then tiltX[notOKtiltX] = interpol(tiltX[OKtiltX],OKtiltX,notOKtiltX) ; Interpolate over gaps for radiation correction; set to -999 again below.
    # if n_elements(OKtiltY) gt 1 then tiltY[notOKtiltY] = interpol(tiltY[OKtiltY],OKtiltY,notOKtiltY) ; Interpolate over gaps for radiation correction; set to -999 again below.
    ds['tilt_x'] = ds['tilt_x'].where(~notOKtiltX)
    ds['tilt_y'] = ds['tilt_y'].where(~notOKtiltY)

    ### NOTE / WARNING / TODO / BUG
    ## Filling w/o considering time gaps to re-create IDL/GDL outputs
    ## Should fill with coordinate not False. Also consider 'max_gap' option?
    ds['tilt_x'] = ds['tilt_x'].interpolate_na(dim='time', use_coordinate=False)
    ds['tilt_y'] = ds['tilt_y'].interpolate_na(dim='time', use_coordinate=False)
    # ds['tilt_x'] = ds['tilt_x'].interpolate_na(dim='time')
    # ds['tilt_y'] = ds['tilt_y'].interpolate_na(dim='time', )
    
    deg2rad = np.pi / 180
    ds['wdir'] = ds['wdir'].where(ds['wspd'] != 0)
    ds['wspd_x'] = ds['wspd'] * np.sin(ds['wdir'] * deg2rad)
    ds['wspd_y'] = ds['wspd'] * np.cos(ds['wdir'] * deg2rad)

    return ds

def vartable():
   """load the variables.csv file"""
   varcsv = os.path.join(os.path.dirname(__file__), 'variables.csv')
   return pd.read_csv(varcsv, index_col=0, comment="#")

def _flagNAN(ds):
    '''Read flagged data from .csv file. For each variable, and downstream 
    dependents, flag as invalid (or other) if set in the flag .csv'''
    flag_file = "./data/flags/" + ds.attrs["station_id"] + ".csv"
    
    if not os.path.isfile(flag_file):
        logging.warning(f'Flag file {flag_file} not found - no data flagged')
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

def _addBasicMeta(ds, v_file):
    ''' Use a variable lookup table (variables.csv) to add the basic metadata 
    to the xarray dataset. This is later amended to finalise L3'''
    varcsv = os.path.join(os.path.dirname(__file__), v_file)
    df = pd.read_csv(varcsv, index_col=0, comment="#")                         #TODO Use attributes.getVars() function for this
    for v in df.index:
        if v == 'time': continue # coordinate variable, not normal var
        if v not in list(ds.variables): continue
        for c in ['standard_name', 'long_name', 'units']:
            if isinstance(df[c][v], float) and np.isnan(df[c][v]): continue
            ds[v].attrs[c] = df[c][v]
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

#------------------------------------------------------------------------------

if __name__ == "__main__": 
    # unittest.main() 
    pass    