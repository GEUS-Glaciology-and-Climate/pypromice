#!/usr/bin/env python
import logging, os, sys, unittest
from argparse import ArgumentParser
import pypromice
from pypromice.process.aws import AWS
from pypromice.process.load import getVars, getMeta
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


def get_l2(config_file, inpath, outpath, variables, metadata):
    logging.basicConfig(
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    
    # Define input path
    station_name = config_file.split('/')[-1].split('.')[0] 
    station_path = os.path.join(inpath, station_name)
    if os.path.exists(station_path):
        aws = AWS(config_file, station_path, variables, metadata)
    else:
        aws = AWS(config_file, inpath, variables, metadata)

    # Perform level 1 and 2 processing
    aws.getL1()
    aws.getL2() 
    v = getVars(variables)
    m = getMeta(metadata)
    # Write out level 2
    if outpath is not None:
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        if aws.L2.attrs['format'] == 'raw':
            prepare_and_write(aws.L2, outpath, v, m, '10min')
        prepare_and_write(aws.L2, outpath, v, m, '60min')
    return aws

def main():
    args = parse_arguments_l2()
    _ = get_l2(args.config_file, args.inpath, args.outpath, args.variables, args.metadata)


if __name__ == "__main__":  
    main()
        