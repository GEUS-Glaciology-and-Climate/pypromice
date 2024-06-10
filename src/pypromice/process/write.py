#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write dataset module
"""
import os

def writeAll(outpath, station_id, l3_h, l3_d, l3_m, csv_order=None):
    '''Write L3 hourly, daily and monthly datasets to .nc and .csv
    files

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