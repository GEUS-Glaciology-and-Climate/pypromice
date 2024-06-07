#!/usr/bin/env python
import os, unittest, pkg_resources
import pandas as pd
import numpy as np
import xarray as xr
from argparse import ArgumentParser
from pypromice.process import getVars, getMeta, addMeta, getColNames, \
    roundValues, resampleL2, writeAll
from pypromice.process.L1toL2 import correctPrecip
from pypromice.process.L2toL3 import toL3
from sys import exit

def parse_arguments_l2_to_l3(debug_args=None):
    parser = ArgumentParser(description="AWS L3 script for the processing L3 data from L2 and merging the L3 data with its historical site. An hourly, daily and monthly L3 data product is outputted to the defined output path")
    parser.add_argument('-s', '--file1', type=str, required=True, nargs='+',
                        help='Path to source L2 file')
    # here will come additional arguments for the merging with historical stations
    parser.add_argument('-v', '--variables', default=None, type=str, required=False, 
    			 help='Path to variables look-up table .csv file for variable name retained'''),
    parser.add_argument('-m', '--metadata', default=None, type=str, required=False, 
    			 help='Path to metadata table .csv file for metadata information'''),
    parser.add_argument('-d', '--datatype', default='raw', type=str, required=False, 
    			 help='Data type to output, raw or tx')
    args = parser.parse_args(args=debug_args)
    args.file1 = ' '.join(args.file1)
    args.folder_gcnet = ' '.join(args.folder_gcnet)
    args.folder_promice = ' '.join(args.folder_promice)
    return args


def loadArr(infile):
    if infile.split('.')[-1].lower() in 'csv':
        df = pd.read_csv(infile)
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df = df.set_index('time')
        ds = xr.Dataset.from_dataframe(df)

    elif infile.split('.')[-1].lower() in 'nc':
        ds = xr.open_dataset(infile)
    
    try:
        name = ds.attrs['station_name'] 
    except:
        name = infile.split('/')[-1].split('.')[0].split('_hour')[0].split('_10min')[0]
        
    print(f'{name} array loaded from {infile}')
    return ds, name

def get_l3():
    args = parse_arguments_l2_to_l3()
                    
    # Check files    
    if os.path.isfile(args.file1): 
        # Load L2 data arrays
        ds1, n1 = loadArr(args.file1)
        
        # converts to L3:
        # - derives sensible heat fluxes
        # - more to come
        ds1 = toL3(ds1)
        
        # here will come the merging with historical data    
    else:
        print(f'Invalid file {args.file1}')
        exit()

    # Get hourly, daily and monthly datasets
    print('Resampling L3 data to hourly, daily and monthly resolutions...')
    l3_h = resampleL2(ds1, '60min')
    l3_d = resampleL2(ds1, '1D')
    l3_m = resampleL2(ds1, 'M')
    
    print(f'Adding variable information from {args.variables}...')
        
    # Load variables look-up table
    var = getVars(args.variables)
        	
    # Round all values to specified decimals places
    l3_h = roundValues(l3_h, var)
    l3_d = roundValues(l3_d, var)
    l3_m = roundValues(l3_m, var)
        
    # Get columns to keep
    if hasattr(ds1, 'p_l'):
        col_names = getColNames(var, 2, args.datatype.lower())  
    else:
        col_names = getColNames(var, 1, args.datatype.lower())    

    # Assign station id
    for l in [l3_h, l3_d, l3_m]:
        l.attrs['station_id'] = n1
    
    # Assign metadata
    print(f'Adding metadata from {args.metadata}...')
    m = getMeta(args.metadata)
    l3_h = addMeta(l3_h, m)
    l3_d = addMeta(l3_d, m)
    l3_m = addMeta(l3_m, m)
      
    # Set up output path
    out = os.path.join(args.outpath, site_id)
    
    # Write to files
    writeAll(out, site_id, l3_h, l3_d, l3_m, col_names)
    print(f'Files saved to {os.path.join(out, site_id)}...')
# %%
if __name__ == "__main__":  
    l2_to_l3()

