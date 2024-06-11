#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write dataset module
"""
import os, logging
import pandas as pd
logger = logging.getLogger(__name__)

from pypromice.process.resample import resample_dataset
from pypromice.process import utilities, write

def prepare_and_write(dataset, outpath, vars_df, meta_dict, time='60min'):
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
    d2 = resample_dataset(dataset, time)
    logger.info('Resampling to '+str(time))

    # Reformat time
    d2 = utilities.reformat_time(d2)
    
    # Reformat longitude (to negative values)
    d2 = utilities.reformat_lon(d2)
    
    # Add variable attributes and metadata
    d2 = utilities.addVars(d2, vars_df)
    d2 = utilities.addMeta(d2, meta_dict)

    # Round all values to specified decimals places
    d2 = utilities.roundValues(d2, vars_df)
    
    # Create out directory
    outdir = os.path.join(outpath, d2.attrs['station_id'])
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    # Get variable names to write out
    col_names = write.getColNames(
        vars_df,
        d2.attrs['number_of_booms'],
        d2.attrs['format'],
        d2.attrs['bedrock'],
    )
    
    # Define filename based on resample rate
    t = int(pd.Timedelta((d2['time'][1] - d2['time'][0]).values).total_seconds())
    if t == 600:
        out_csv = os.path.join(outdir, d2.attrs['station_id']+'_10min.csv')
        out_nc = os.path.join(outdir, d2.attrs['station_id']+'_10min.nc')
    elif t == 3600:
        out_csv = os.path.join(outdir, d2.attrs['station_id']+'_hour.csv')
        out_nc = os.path.join(outdir, d2.attrs['station_id']+'_hour.nc')
    else:
        out_csv = os.path.join(outdir, d2.attrs['station_id']+'_month.csv')
        out_nc = os.path.join(outdir, d2.attrs['station_id']+'_month.nc')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # Write to csv file
    logger.info('Writing to files...')
    write.writeCSV(out_csv, d2, col_names)
    
    # Write to netcdf file
    col_names = col_names + ['lat', 'lon', 'alt']
    write.writeNC(out_nc, d2, col_names)
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

def getColNames(vars_df, booms=None, data_type=None, bedrock=False):
    '''Get all variable names for a given data type, based on a variables
    look-up table. This is mainly for exporting purposes

    Parameters
    ----------
    vars_df : pd.DataFrame
        Variables look-up table
    booms : int, optional
        Number of booms. If this parameter is empty then all variables
        regardless of boom type will be passed. The default is None.
    data_type : str, optional
        Data type, "tx", "STM" or "raw". If this parameter is empty then all
        variables regardless of data type will be passed. The default is None.

    Returns
    -------
    list
        Variable names
    '''
    if booms==1:
        vars_df = vars_df.loc[vars_df['station_type'].isin(['one-boom','all'])]
    elif booms==2:
        vars_df = vars_df.loc[vars_df['station_type'].isin(['two-boom','all'])]

    if data_type=='TX':
        vars_df = vars_df.loc[vars_df['data_type'].isin(['TX','all'])]
    elif data_type=='STM' or data_type=='raw':
        vars_df = vars_df.loc[vars_df['data_type'].isin(['raw','all'])]

    col_names = list(vars_df.index)
    if isinstance(bedrock, str):
        bedrock = (bedrock.lower() == 'true')
    if bedrock == True:
        col_names.remove('cc')
        for v in ['dlhf_u', 'dlhf_l', 'dshf_u', 'dshf_l']:
            try:
                col_names.remove(v)
            except:
                pass
    return col_names
