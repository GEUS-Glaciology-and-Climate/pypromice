#!/usr/bin/env python
import os, logging, sys
import xarray as xr
from argparse import ArgumentParser
import pypromice
from pypromice.process.L2toL3 import toL3
from pypromice.process.load import getVars, getMeta
from pypromice.process.write import prepare_and_write
logger = logging.getLogger(__name__)

def parse_arguments_l2tol3(debug_args=None):
    parser = ArgumentParser(description="AWS L3 script for the processing L3 "+
                            "data from L2 and merging the L3 data with its "+
                            "historical site. An hourly, daily and monthly L3 "+
                            "data product is outputted to the defined output path")
    parser.add_argument('-i', '--inpath', type=str, required=True, 
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

def get_l2tol3():
    args = parse_arguments_l2tol3()
    logging.basicConfig(
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
  
    # Define Level 2 dataset from file
    with xr.open_dataset(args.inpath) as l2:
        l2.load()
    for varname in l2.variables:
        if 'encoding' in l2[varname].attrs:
            del l2[varname].attrs['encoding']        
    if 'bedrock' in l2.attrs.keys():
        l2.attrs['bedrock'] = l2.attrs['bedrock'] == 'True'
    if 'number_of_booms' in l2.attrs.keys():
        l2.attrs['number_of_booms'] = int(l2.attrs['number_of_booms'])
    
    # Perform Level 3 processing
    l3 = toL3(l2)

    # Write Level 3 dataset to file if output directory given
    v = getVars(args.variables)
    m = getMeta(args.metadata)
    if args.outpath is not None:
        prepare_and_write(l3, args.outpath, v, m, '60min')
        prepare_and_write(l3, args.outpath, v, m, '1D')
        prepare_and_write(l3, args.outpath, v, m, 'M')

if __name__ == "__main__":  
    get_l2tol3()
