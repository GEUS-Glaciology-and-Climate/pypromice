#!/usr/bin/env python
import logging, os, sys, unittest
from argparse import ArgumentParser
import pypromice
from pypromice.process.aws import AWS
from pypromice.process.write import prepare_and_write

def parse_arguments_l2():
    parser = ArgumentParser(description="AWS L2 processor")

    parser.add_argument('-c', '--config_file', type=str, required=True,
                        help='Path to config (TOML) file')
    parser.add_argument('-i', '--inpath', type=str, required=True, 
                        help='Path to input data')
    parser.add_argument('-o', '--outpath', default=None, type=str, required=False, 
                        help='Path where to write output')
    parser.add_argument('-v', '--variables', default=None, type=str, 
                        required=False, help='File path to variables look-up table')
    parser.add_argument('-m', '--metadata', default=None, type=str, 
                        required=False, help='File path to metadata')
    args = parser.parse_args()
    return args

def get_l2():
    args = parse_arguments_l2()

    logging.basicConfig(
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    
    # Define input path
    station_name = args.config_file.split('/')[-1].split('.')[0] 
    station_path = os.path.join(args.inpath, station_name)
    if os.path.exists(station_path):
        aws = AWS(args.config_file, station_path, args.variables, args.metadata)
    else:
        aws = AWS(args.config_file, args.inpath, args.variables, args.metadata)

    # Perform level 1 and 2 processing
    aws.getL1()
    aws.getL2() 
    
    # Write out level 2
    if args.outpath is not None:
        if not os.path.isdir(args.outpath):
            os.mkdir(args.outpath)
        if aws.L2.attrs['format'] == 'raw':
            prepare_and_write(aws.L2, args.outpath, args.variables, args.metadata, '10min')
        prepare_and_write(aws.L2, args.outpath, args.variables, args.metadata, '60min')


if __name__ == "__main__":  
    get_l2()
        