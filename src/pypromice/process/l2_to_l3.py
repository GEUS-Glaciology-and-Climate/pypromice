#!/usr/bin/env python
import os, logging, sys
import xarray as xr
from argparse import ArgumentParser
import pypromice
from pypromice.process.aws import AWS

def parse_arguments_l2_to_l3(debug_args=None):
    parser = ArgumentParser(description="AWS L3 script for the processing L3 "+
                            "data from L2 and merging the L3 data with its "+
                            "historical site. An hourly, daily and monthly L3 "+
                            "data product is outputted to the defined output path")
    parser.add_argument('-c', '--config_file', type=str, required=True,
                        help='Path to config (TOML) file')
    parser.add_argument('-i', '--inpath', type=str, required=True, 
                        help='Path to input data')
    parser.add_argument('-l', '--level_2', type=str, required=True,
                        help='Path to Level 2 .nc data file')
    parser.add_argument('-o', '--outpath', default=None, type=str, required=False, 
                        help='Path where to write output')
    parser.add_argument('-v', '--variables', default=None, type=str, 
                        required=False, help='File path to variables look-up table')
    parser.add_argument('-m', '--metadata', default=None, type=str, 
                        required=False, help='File path to metadata')
    parser.add_argument('-g', '--gcnet_historical', default=None, type=str, 
                        required=False, help='File path to historical GC-Net data file')

    # here will come additional arguments for the merging with historical stations
    args = parser.parse_args(args=debug_args)
    return args

def l2_to_l3():
    args = parse_arguments_l2_to_l3()
    logging.basicConfig(
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    # Define variables (either from file or pypromice package defaults)
    if args.variables is None:
        v = os.path.join(os.path.dirname(pypromice.__file__),'process/variables.csv')
    else:
        v = args.variables
        
    # Define metadata (either from file or pypromice package defaults)
    if args.variables is None:
        m = os.path.join(os.path.dirname(pypromice.__file__),'process/metadata.csv')
    else:
        m = args.metadata

    # Define input path
    station_name = args.config_file.split('/')[-1].split('.')[0] 
    station_path = os.path.join(args.inpath, station_name)
    if os.path.exists(station_path):
        aws = AWS(args.config_file, station_path, v, m)
    else:
        aws = AWS(args.config_file, args.inpath, v, m)

    
    # Define Level 2 dataset from file
    aws.L2 = xr.open_dataset(args.level_2)
    
    # Perform Level 3 processing
    aws.getL3()
    
    # Write Level 3 dataset to file if output directory given
    if args.outpath is not None:
        aws.writeL3(args.outpath)

if __name__ == "__main__":  
    l2_to_l3()
