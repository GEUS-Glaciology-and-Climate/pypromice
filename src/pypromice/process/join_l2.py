#!/usr/bin/env python
import logging, sys, os, unittest
import pandas as pd
import xarray as xr
from argparse import ArgumentParser
from pypromice.process.L1toL2 import correctPrecip
from pypromice.process.write import prepare_and_write
logger = logging.getLogger(__name__)

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
    if infile.split('.')[-1].lower() == 'csv':
        df = pd.read_csv(infile, index_col=0, parse_dates=True)
        ds = xr.Dataset.from_dataframe(df)  
    elif infile.split('.')[-1].lower() == 'nc':
        with xr.open_dataset(infile) as ds:
            ds.load()
        # Remove encoding attributes from NetCDF
        for varname in ds.variables:
            if ds[varname].encoding!={}:
                ds[varname].encoding = {}

    try:
        name = ds.attrs['station_id'] 
    except:
        name = infile.split('/')[-1].split('.')[0].split('_hour')[0].split('_10min')[0]
        ds.attrs['station_id'] = name
    if 'bedrock' in ds.attrs.keys():
        ds.attrs['bedrock'] = ds.attrs['bedrock'] == 'True'
    if 'number_of_booms' in ds.attrs.keys():
        ds.attrs['number_of_booms'] = int(ds.attrs['number_of_booms'])

    logger.info(f'{name} array loaded from {infile}')
    return ds, name
    

def join_l2(file1,file2,outpath,variables,metadata) -> xr.Dataset:
    logging.basicConfig(
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    # Check files
    if os.path.isfile(file1) and os.path.isfile(file2): 

        # Load data arrays
        ds1, n1 = loadArr(file1)
        ds2, n2 = loadArr(file2)    	
        
        # Check stations match
        if n1.lower() == n2.lower():
            
        	# Merge arrays
            logger.info(f'Combining {file1} with {file2}...')
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
            logger.info(f'Mismatched station names {n1}, {n2}')
            exit()            
    
    elif os.path.isfile(file1):  
        ds1, name = loadArr(file1)
        logger.info(f'Only one file found {file1}...')
        all_ds = ds1  

    elif os.path.isfile(file2):
        ds2, name = loadArr(file2)
        logger.info(f'Only one file found {file2}...')
        all_ds = ds2  
    
    else:
        logger.info(f'Invalid files {file1}, {file2}')
        exit()

    all_ds.attrs['format'] = 'merged RAW and TX'

    # Resample to hourly, daily and monthly datasets and write to file
    prepare_and_write(all_ds, outpath, variables, metadata, resample = False)
    
    logger.info(f'Files saved to {os.path.join(outpath, name)}...')
    return all_ds

def main():
    args = parse_arguments_join()
    _ = join_l2(args.file1, args.file2, args.outpath, args.variables, args.metadata)
    
if __name__ == "__main__":  
    main()
