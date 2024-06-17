#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities module for data formatting, populating and metadata handling
"""
import datetime, uuid
from importlib import metadata
import pandas as pd
import numpy as np

def roundValues(ds, df, col='max_decimals'):
    '''Round all variable values in data array based on pre-defined rounding
    value in variables look-up table DataFrame

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to round values in
    df : pd.Dataframe
        Variable look-up table with rounding values
    col : str
        Column in variable look-up table that contains rounding values. The
        default is "max_decimals"
    '''
    df = df[col]
    df = df.dropna(how='all')
    for var in df.index:
        if var not in list(ds.variables):
            continue
        if df[var] is not np.nan:
            ds[var] = ds[var].round(decimals=int(df[var]))
    return ds

def reformat_time(dataset):
    '''Re-format time'''
    t = dataset['time'].values
    dataset['time'] = list(t)
    return dataset

def reformat_lon(dataset, exempt=['UWN', 'Roof_GEUS', 'Roof_PROMICE']):
    '''Switch gps_lon to negative values (degrees_east). We do this here, and 
    NOT in addMeta, otherwise we switch back to positive when calling getMeta 
    in joinL2'''
    if 'station_id' in dataset.attrs.keys():
        id = dataset.attrs['station_id']
    else:
        id = dataset.attrs['site_id']

    if id not in exempt:
        if 'gps_lon' not in dataset.keys():
            return dataset
        dataset['gps_lon'] = dataset['gps_lon'] * -1
    return dataset

def popCols(ds, names):
    '''Populate dataset with all given variable names

    Parammeters
    -----------
    ds : xr.Dataset
        Dataset
    names : list
        List of variable names to populate
    '''
    for v in names:
        if v not in list(ds.variables):
            ds[v] = (('time'), np.arange(ds['time'].size)*np.nan)
    return ds

def addBasicMeta(ds, vars_df):
    ''' Use a variable lookup table DataFrame to add the basic metadata
    to the xarray dataset. This is later amended to finalise L3

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to add metadata to
    vars_df : pd.DataFrame
        Metadata dataframe

    Returns
    -------
    ds : xr.Dataset
        Dataset with added metadata
    '''
    for v in vars_df.index:
        if v == 'time': continue # coordinate variable, not normal var
        if v not in list(ds.variables): continue
        for c in ['standard_name', 'long_name', 'units']:
            if isinstance(vars_df[c][v], float) and np.isnan(vars_df[c][v]): continue
            ds[v].attrs[c] = vars_df[c][v]
    return ds

def populateMeta(ds, conf, skip):
    '''Populate L0 Dataset with metadata dictionary

    Parameters
    ----------
    ds : xarray.Dataset
        L0 dataset
    conf : dict
        Metadata dictionary
    skip : list
        List of column names to skip parsing to metadata

    Returns
    -------
    ds : xarray.Dataset
        L0 dataset with metadata populated as Dataset attributes
    '''
    # skip = ["columns", "skiprows"]
    for k in conf.keys():
        if k not in skip: ds.attrs[k] = conf[k]
    return ds


def addVars(ds, variables):
    '''Add variable attributes from file to dataset

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to add variable attributes to
    variables : pandas.DataFrame
        Variables lookup table file

    Returns
    -------
    ds : xarray.Dataset
        Dataset with metadata
   '''
    for k in ds.keys():
        if k not in variables.index: continue
        ds[k].attrs['standard_name'] = variables.loc[k]['standard_name']
        ds[k].attrs['long_name'] = variables.loc[k]['long_name']
        ds[k].attrs['units'] = variables.loc[k]['units']
        ds[k].attrs['coverage_content_type'] = variables.loc[k]['coverage_content_type']
        ds[k].attrs['coordinates'] = variables.loc[k]['coordinates']
    return ds

def addMeta(ds, meta):
    '''Add metadata attributes from file to dataset

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to add metadata attributes to
    meta : dict
        Metadata file

    Returns
    -------
    ds : xarray.Dataset
        Dataset with metadata
   '''
    if 'gps_lon' in ds.keys():
        ds['lon'] = ds['gps_lon'].mean()
        ds['lon'].attrs = ds['gps_lon'].attrs

        ds['lat'] = ds['gps_lat'].mean()
        ds['lat'].attrs = ds['gps_lat'].attrs

        ds['alt'] = ds['gps_alt'].mean()
        ds['alt'].attrs = ds['gps_alt'].attrs

    # for k in ds.keys(): # for each var
    #     if 'units' in ds[k].attrs:
    #         if ds[k].attrs['units'] == 'C':
    #             ds[k].attrs['units'] = 'degrees_C'

    # https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3#geospatial_bounds
    if 'station_id' in ds.attrs.keys():
        ds.attrs['id'] = 'dk.geus.promice:' + str(uuid.uuid3(uuid.NAMESPACE_DNS, ds.attrs['station_id']))
    else:
        ds.attrs['id'] = 'dk.geus.promice:' + str(uuid.uuid3(uuid.NAMESPACE_DNS, ds.attrs['site_id']))
    ds.attrs['history'] = 'Generated on ' + datetime.datetime.utcnow().isoformat()
    ds.attrs['date_created'] = str(datetime.datetime.now().isoformat())
    ds.attrs['date_modified'] = ds.attrs['date_created']
    ds.attrs['date_issued'] = ds.attrs['date_created']
    ds.attrs['date_metadata_modified'] = ds.attrs['date_created']

    ds.attrs['geospatial_bounds'] = "POLYGON((" + \
        f"{ds['lat'].min().values} {ds['lon'].min().values}, " + \
        f"{ds['lat'].min().values} {ds['lon'].max().values}, " + \
        f"{ds['lat'].max().values} {ds['lon'].max().values}, " + \
        f"{ds['lat'].max().values} {ds['lon'].min().values}, " + \
        f"{ds['lat'].min().values} {ds['lon'].min().values}))"

    ds.attrs['geospatial_lat_min'] = str(ds['lat'].min().values)
    ds.attrs['geospatial_lat_max'] = str(ds['lat'].max().values)
    ds.attrs['geospatial_lon_min'] = str(ds['lon'].min().values)
    ds.attrs['geospatial_lon_max'] = str(ds['lon'].max().values)
    ds.attrs['geospatial_vertical_min'] = str(ds['alt'].min().values)
    ds.attrs['geospatial_vertical_max'] = str(ds['alt'].max().values)
    ds.attrs['geospatial_vertical_positive'] = 'up'
    ds.attrs['time_coverage_start'] = str(ds['time'][0].values)
    ds.attrs['time_coverage_end'] = str(ds['time'][-1].values)

    try:
        ds.attrs['source']= 'pypromice v' + str(metadata.version('pypromice'))
    except:
        ds.attrs['source'] = 'pypromice'

    # https://www.digi.com/resources/documentation/digidocs/90001437-13/reference/r_iso_8601_duration_format.htm
    try:
        ds.attrs['time_coverage_duration'] = str(pd.Timedelta((ds['time'][-1] - ds['time'][0]).values).isoformat())
        ds.attrs['time_coverage_resolution'] = str(pd.Timedelta((ds['time'][1] - ds['time'][0]).values).isoformat())
    except:
        ds.attrs['time_coverage_duration'] = str(pd.Timedelta(0).isoformat())
        ds.attrs['time_coverage_resolution'] = str(pd.Timedelta(0).isoformat())

    # Note: int64 dtype (long int) is incompatible with OPeNDAP access via THREDDS for NetCDF files
    # See https://stackoverflow.com/questions/48895227/output-int32-time-dimension-in-netcdf-using-xarray
    ds.time.encoding["dtype"] = "i4" # 32-bit signed integer
    #ds.time.encoding["calendar"] = 'proleptic_gregorian' # this is default

    # Load metadata attributes and add to Dataset
    [_addAttr(ds, key, value) for key,value in meta.items()]

    # Check attribute formating
    for k,v in ds.attrs.items():
        if not isinstance(v, str) or not isinstance(v, int):
            ds.attrs[k]=str(v)
    return ds

def _addAttr(ds, key, value):
    '''Add attribute to xarray dataset

    ds : xr.Dataset
        Dataset to add attribute to
    key : str
        Attribute name, with "." denoting variable attributes
    value : str/int
        Value for attribute'''
    if len(key.split('.')) == 2:
        try:
            ds[key.split('.')[0]].attrs[key.split('.')[1]] = str(value)
        except:
            pass
            # logger.info(f'Unable to add metadata to {key.split(".")[0]}')
    else:
        ds.attrs[key] = value
