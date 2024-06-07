#!/usr/bin/env python
import logging, os, sys, unittest
from argparse import ArgumentParser
import pypromice
from pypromice.process.aws import AWS

def parse_arguments_level():
    parser = ArgumentParser(description="AWS L2/L3 processor")

    parser.add_argument('-c', '--config_file', type=str, required=True,
                        help='Path to config (TOML) file')
    parser.add_argument('-i', '--inpath', type=str, required=True, 
                        help='Path to input data')
    parser.add_argument('-o', '--outpath', default=None, type=str, required=False, 
                        help='Path where to write output')
    parser.add_argument('-l', '--level', default=3, type=int, required=False, 
                        help='Processing level')
    parser.add_argument('-v', '--variables', default=None, type=str, 
                        required=False, help='File path to variables look-up table')
    parser.add_argument('-m', '--metadata', default=None, type=str, 
                        required=False, help='File path to metadata')
    args = parser.parse_args()
    return args

def get_l3():
    args = parse_arguments_level()

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

    # Perform level 1 and 2 processing
    aws.getL1()
    aws.getL2() 
    
    # Write out level 2 
    if int(args.level)==2:
        if args.outpath is not None:
            aws.writeL2(args.outpath)
    
    # Perform level 3 processing and write out
    elif int(args.level)==3:
        aws.getL3()
        if args.outpath is not None:
            aws.writeL3(args.outpath)
    
    # Do nothing else if different level number is given (potential to expand to
    # level 4 processing option) - PHO
    else:
        print('Invalid level number given')
        
if __name__ == "__main__":  
    get_l3()
        
