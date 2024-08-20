#!/usr/bin/env python
import logging, os, sys, unittest
from argparse import ArgumentParser
from pypromice.process.aws import AWS

def parse_arguments_l3():
    parser = ArgumentParser(description="AWS L3 processor")

    parser.add_argument('-c', '--config_file', type=str, required=True,
                        help='Path to config (TOML) file')
    parser.add_argument('-i', '--inpath', default='data', type=str, required=True, 
                        help='Path to input data')
    parser.add_argument('-o', '--outpath', default=None, type=str, required=False, 
                        help='Path where to write output')
    parser.add_argument('-v', '--variables', default=None, type=str, 
                        required=False, help='File path to variables look-up table')
    parser.add_argument('-m', '--metadata', default=None, type=str, 
                        required=False, help='File path to metadata')
    args = parser.parse_args()
    return args

def get_l3():
    args = parse_arguments_l3()

    logging.basicConfig(
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    station_name = args.config_file.split('/')[-1].split('.')[0] 
    station_path = os.path.join(args.inpath, station_name)

    if os.path.exists(station_path):
        aws = AWS(args.config_file, station_path, args.variables, args.metadata)
    else:
        aws = AWS(args.config_file, args.inpath, args.variables, args.metadata)

    aws.process() 
     
    if args.outpath is not None:
        aws.write(args.outpath)
        
if __name__ == "__main__":  
    get_l3()
        
