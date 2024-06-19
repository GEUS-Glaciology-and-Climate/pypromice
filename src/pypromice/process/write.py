#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing all the functions needed to prepare and AWS data
"""
import os, logging, datetime, uuid
import pandas as pd
import numpy as np
from importlib import metadata
from pypromice.process.resample import resample_dataset
from pypromice.process import load
logger = logging.getLogger(__name__)

def prepare_and_write(dataset, outpath, vars_df=None, meta_dict=None, time='60min', resample=True):
    '''Prepare data with resampling, formating and metadata population; then
    write data to .nc and .csv hourly and daily files

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to write to file
    outpath : str
        Output directory
    vars_df : pandas.DataFrame
        Variables look-up table dataframe
    meta_dict : dictionary
        Metadata dictionary to write to dataset
    time : str
        Resampling interval for output dataset
    '''
    # Resample dataset
    if resample:
        d2 = resample_dataset(dataset, time)
        logger.info('Resampling to '+str(time))
    else:
        d2 = dataset.copy()
        
    # Reformat time
    d2 = reformat_time(d2)
    
    # finding station/site name
    if 'station_id' in d2.attrs.keys():
        name = d2.attrs['station_id']
    else:
        name = d2.attrs['site_id']
        
    # Reformat longitude (to negative values)
    if 'gps_lon' in d2.keys():
        d2 = reformat_lon(d2)
    else:
        logger.info('%s does not have gpd_lon'%name)
        
    # Add variable attributes and metadata
    if vars_df is None:
        vars_df = load.getVars()
    if meta_dict is None:
        meta_dict = load.getMeta()
        
    d2 = addVars(d2, vars_df)
    d2 = addMeta(d2, meta_dict)

    # Round all values to specified decimals places
    d2 = roundValues(d2, vars_df)

    # Get variable names to write out
    col_names = getColNames(vars_df, d2, remove_nan_fields=True)

    # Define filename based on resample rate
    t = int(pd.Timedelta((d2['time'][1] - d2['time'][0]).values).total_seconds())

    # Create out directory
    outdir = os.path.join(outpath, name)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    if t == 600:
        out_csv = os.path.join(outdir, name+'_10min.csv')
        out_nc = os.path.join(outdir, name+'_10min.nc')
    elif t == 3600:
        out_csv = os.path.join(outdir, name+'_hour.csv')
        out_nc = os.path.join(outdir, name+'_hour.nc')
    elif t == 86400:
        # removing instantaneous values from daily and monthly files
        for v in col_names:
            if ('_i' in v) and ('_i_' not in v):
                col_names.remove(v)
        out_csv = os.path.join(outdir, name+'_day.csv')
        out_nc = os.path.join(outdir, name+'_day.nc')
    else:
        # removing instantaneous values from daily and monthly files
        for v in col_names:
            if ('_i' in v) and ('_i_' not in v):
                col_names.remove(v)
        out_csv = os.path.join(outdir, name+'_month.csv')
        out_nc = os.path.join(outdir, name+'_month.nc')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # Write to csv file
    logger.info('Writing to files...')
    writeCSV(out_csv, d2, col_names)

    # Write to netcdf file
    col_names = col_names + ['lat', 'lon', 'alt']
    writeNC(out_nc, d2, col_names)
    logger.info(f'Written to {out_csv}')
    logger.info(f'Written to {out_nc}')


def writeAll(outpath, station_id, l3_h, l3_d, l3_m, csv_order=None):
    '''Write L3 hourly, daily and monthly datasets to .nc and .csv
    files

    Parameters
    ----------
    outpath : str
        Output file path
    station_id : str
        Station name
    l3_h : xr.Dataset
        L3 hourly data
    l3_d : xr.Dataset
        L3 daily data
    l3_m : xr.Dataset
        L3 monthly data
    csv_order : list, optional
        List order of variables
    '''
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    outfile_h = os.path.join(outpath, station_id + '_hour')
    outfile_d = os.path.join(outpath, station_id + '_day')
    outfile_m = os.path.join(outpath, station_id + '_month')
    for o,l in zip([outfile_h, outfile_d, outfile_m], [l3_h ,l3_d, l3_m]):
        writeCSV(o+'.csv',l, csv_order)
        writeNC(o+'.nc',l)
        
def writeCSV(outfile, Lx, csv_order):
    '''Write data product to CSV file

    Parameters
    ----------
    outfile : str
        Output file path
    Lx : xr.Dataset
        Dataset to write to file
    csv_order : list
        List order of variables
    '''
    Lcsv = Lx.to_dataframe().dropna(how='all')
    if csv_order is not None:
        names = [c for c in csv_order if c in list(Lcsv.columns)]
        Lcsv = Lcsv[names]
    Lcsv.to_csv(outfile)

def writeNC(outfile, Lx, col_names=None):
    '''Write data product to NetCDF file

    Parameters
    ----------
    outfile : str
        Output file path
    Lx : xr.Dataset
        Dataset to write to file
    '''
    if os.path.isfile(outfile):
        os.remove(outfile)
    if col_names is not None:
        names = [c for c in col_names if c in list(Lx.keys())]
    else:
        names = list(Lx.keys())

    Lx[names].to_netcdf(outfile, mode='w', format='NETCDF4', compute=True)

def getColNames(vars_df, ds, remove_nan_fields=False):
    '''
    Get variable names for a given dataset with respect to its type and processing level
    
    The dataset must have the the following attributes: 
    * level
    * number_of_booms when the processing level is <= 2
    
    This is mainly for exporting purposes. 
   
   Parameters
    -------
    list
        Variable names
    '''
    # selecting variable list based on level
    vars_df = vars_df.loc[vars_df[ds.attrs['level']] == 1]

    # selecting variable list based on geometry
    if ds.attrs['level'] in ['L0', 'L1', 'L2']:
        if ds.attrs['number_of_booms']==1:
            vars_df = vars_df.loc[vars_df['station_type'].isin(['one-boom','all'])]
        elif ds.attrs['number_of_booms']==2:
            vars_df = vars_df.loc[vars_df['station_type'].isin(['two-boom','all'])]
            
    var_list = list(vars_df.index)
    if remove_nan_fields:
        for v in var_list:
             if v not in ds.keys():
                 var_list.remove(v)
                 continue
             if ds[v].isnull().all():
                 var_list.remove(v)
    return var_list

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

    # Attribute convention for data discovery
    # https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3
    
    # Determine the temporal resolution
    time_diff = pd.Timedelta((ds['time'][1] - ds['time'][0]).values)
    if time_diff == pd.Timedelta('10min'):
        sample_rate  = "10min"
    elif time_diff == pd.Timedelta('1H'):
        sample_rate  = "hourly"
    elif time_diff == pd.Timedelta('1D'):
        sample_rate  = "daily"
    elif time_diff == pd.Timedelta('1M'):
        sample_rate  = "monthly"
    else:
        sample_rate  = "unknown_sample_rate"
        
    if 'station_id' in ds.attrs.keys():
        ds.attrs['id'] = 'dk.geus.promice.station.' + ds.attrs['station_id']+'.'+sample_rate
    else:
        ds.attrs['id'] = 'dk.geus.promice.site.' + ds.attrs['site_id'] +'.'+sample_rate
        
    ds.attrs['history'] = 'Generated on ' + datetime.datetime.utcnow().isoformat()
    ds.attrs['date_created'] = str(datetime.datetime.now().isoformat())
    ds.attrs['date_modified'] = ds.attrs['date_created']
    ds.attrs['date_issued'] = ds.attrs['date_created']
    ds.attrs['date_metadata_modified'] = ds.attrs['date_created']
    ds.attrs['processing_level'] = ds.attrs['level'].replace('L','level ')

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