#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:58:39 2024

@author: pho
"""
import logging
import numpy as np
import xarray as xr
from pypromice.process.L1toL2 import calcDirWindSpeeds
logger = logging.getLogger(__name__)

def resample_dataset(ds_h, t):
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
        Resample factor, same variable definition as in
        pandas.DataFrame.resample()

    Returns
    -------
    ds_d : xarray.Dataset
        L3 AWS dataset resampled to the frequency defined by t
    '''
    df_d = ds_h.to_dataframe().resample(t).mean()
    
    # taking the 10 min data and using it as instantaneous values:
    if (t == '60min') and (ds_h.time.diff(dim='time').isel(time=0).dt.total_seconds() == 600):
        cols_to_update = ['p_i', 't_i', 'rh_i', 'rh_i_cor', 'wspd_i', 'wdir_i','wspd_x_i','wspd_y_i']
        for col in cols_to_update:
            df_d[col] = ds_h.reindex(time=df_d.index)[col.replace('_i','_u')].values
            if col == 'p_i':
                df_d[col] = df_d[col].values-1000
            

    # recalculating wind direction from averaged directional wind speeds
    for var in ['wdir_u','wdir_l']:
        boom = var.split('_')[1]
        if var in df_d.columns:
            if ('wspd_x_'+boom in df_d.columns) & ('wspd_y_'+boom in df_d.columns):
                df_d[var] = _calcWindDir(df_d['wspd_x_'+boom], df_d['wspd_y_'+boom])
            else:
                logger.info(var+' in dataframe but not wspd_x_'+boom+' nor wspd_y_'+boom+', recalculating them')
                ds_h['wspd_x_'+boom], ds_h['wspd_y_'+boom] = calcDirWindSpeeds(ds_h['wspd_'+boom], ds_h['wdir_'+boom])
                df_d[['wspd_x_'+boom, 'wspd_y_'+boom]] = ds_h[['wspd_x_'+boom, 'wspd_y_'+boom]].to_dataframe().resample(t).mean()
                df_d[var] = _calcWindDir(df_d['wspd_x_'+boom], df_d['wspd_y_'+boom])
    
    # recalculating relative humidity from average vapour pressure and average
    # saturation vapor pressure
    for var in ['rh_u','rh_l']:
        lvl = var.split('_')[1]
        if var in df_d.columns:
            if ('t_'+lvl in ds_h.keys()):
                es_wtr, es_cor = calculateSaturationVaporPressure(ds_h['t_'+lvl])
                p_vap = ds_h[var] / 100 * es_wtr
                
                df_d[var] = (p_vap.to_series().resample(t).mean() \
                           / es_wtr.to_series().resample(t).mean())*100
                if var+'_cor' in df_d.keys():
                    df_d[var+'_cor'] = (p_vap.to_series().resample(t).mean() \
                               / es_cor.to_series().resample(t).mean())*100
    
    # passing each variable attribute to the ressample dataset
    vals = []
    for c in df_d.columns:
        if c in ds_h.data_vars:
            vals.append(xr.DataArray(
                data=df_d[c], dims=['time'],
               coords={'time':df_d.index}, attrs=ds_h[c].attrs))
        else:
            vals.append(xr.DataArray(
                data=df_d[c], dims=['time'],
               coords={'time':df_d.index}, attrs=None))
            
    ds_d = xr.Dataset(dict(zip(df_d.columns,vals)), attrs=ds_h.attrs)
    return ds_d


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
