#!/usr/bin/env python
import os, unittest
import pandas as pd
import xarray as xr
from argparse import ArgumentParser
from pypromice.process.load import getVars, getMeta
from pypromice.process.utilities import addMeta, roundValues
from pypromice.process.write import prepare_and_write
from pypromice.process.L1toL2 import correctPrecip

def parse_arguments_join():
    parser = ArgumentParser(description="AWS L2 joiner for merging together two L2 products, for example an L2 RAW and L2 TX data product. An hourly, daily and monthly L2 data product is outputted to the defined output path")
    parser.add_argument('-s', '--file1', type=str, required=True,
                        help='Path to source L2 file, which will be preferenced in merge process')
    parser.add_argument('-t', '--file2', type=str, required=True, 
                        help='Path to target L2 file, which will be used to fill gaps in merge process')
    parser.add_argument('-o', '--outpath', default=os.getcwd(), type=str, required=True, 
                        help='Path where to write output')
    parser.add_argument('-v', '--variables', default=None, type=str, required=False, 
    			 help='Path to variables look-up table .csv file for variable name retained'''),
    parser.add_argument('-m', '--metadata', default=None, type=str, required=False, 
    			 help='Path to metadata table .csv file for metadata information'''),
    args = parser.parse_args()
    return args

def loadArr(infile):
    if infile.split('.')[-1].lower() in 'csv':
        df = pd.read_csv(infile, index_col=0, parse_dates=True)
        ds = xr.Dataset.from_dataframe(df)  
    
    elif infile.split('.')[-1].lower() in 'nc':
        ds = xr.open_dataset(infile)
    
    try:
        name = ds.attrs['station_name'] 
    except:
        name = infile.split('/')[-1].split('.')[0].split('_hour')[0].split('_10min')[0]
        
    print(f'{name} array loaded from {infile}')
    return ds, name
    

def join_levels():
    args = parse_arguments_join()

    # Define variables and metadata (either from file or pypromice package defaults)
    v = getVars(args.variables)
    m = getMeta(args.metadata)
            
    # Check files
    if os.path.isfile(args.file1) and os.path.isfile(args.file2): 

        # Load data arrays
        ds1, n1 = loadArr(args.file1)
        ds2, n2 = loadArr(args.file2)    	
        
        # Check stations match
        if n1.lower() == n2.lower():
            
        	# Merge arrays
            print(f'Combining {args.file1} with {args.file2}...')
            name = n1
            all_ds = ds1.combine_first(ds2)
            
            # Re-calculate corrected precipitation
            if hasattr(all_ds, 'precip_u_cor'):
                if ~all_ds['precip_u_cor'].isnull().all():
                    all_ds['precip_u_cor'],  _ = correctPrecip(all_ds['precip_u'], 
                                                                all_ds['wspd_u'])
            if hasattr(all_ds, 'precip_l_cor'):
                if ~all_ds['precip_l_cor'].isnull().all():
                    all_ds['precip_l_cor'],  _ = correctPrecip(all_ds['precip_l'], 
                                                                all_ds['wspd_l'])                    
        else:
            print(f'Mismatched station names {n1}, {n2}')
            exit()            
    
    elif os.path.isfile(args.file1):  
        ds1, name = loadArr(args.file1)
        print(f'Only one file found {args.file1}...')
        all_ds = ds1  

    elif os.path.isfile(args.file2):
        ds2, name = loadArr(args.file2)
        print(f'Only one file found {args.file2}...')
        all_ds = ds2  
    
    else:
        print(f'Invalid files {args.file1}, {args.file2}')
        exit()
    
    # Define output directory subfolder
    out = os.path.join(args.output, name)
    
    # Resample to hourly, daily and monthly datasets and write to file
    prepare_and_write(all_ds, out, v, m, '60min')
    prepare_and_write(all_ds, out, v, m, '1D')
    prepare_and_write(all_ds, out, v, m, 'M')
           
    print(f'Files saved to {os.path.join(out, name)}...')

if __name__ == "__main__":  
    join_levels()
