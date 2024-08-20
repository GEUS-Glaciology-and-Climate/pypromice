#!/usr/bin/env python
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

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
    parser.add_argument('--data_issues_path', '--issues', default=None, help="Path to data issues repository")
    args = parser.parse_args()
    return args


def get_l2(config_file, inpath, outpath, variables, metadata, data_issues_path: Path) -> AWS:
    # Define input path
    station_name = config_file.split('/')[-1].split('.')[0] 
    station_path = os.path.join(inpath, station_name)
    
    # checking that data_issues_path is valid
    if data_issues_path is None:
        data_issues_path = Path("../PROMICE-AWS-data-issues")
        if data_issues_path.exists():
            logging.warning(f"data_issues_path is missing. Using default data issues path: {data_issues_path}")
        else:
            raise ValueError("data_issues_path is missing. Please provide a valid path to the data issues repository")

    if os.path.exists(station_path):
        aws = AWS(config_file, 
                  station_path,
                  data_issues_repository=data_issues_path, 
                  var_file=variables, 
                  meta_file=metadata)
    else:
        aws = AWS(config_file, 
                  inpath, 
                  data_issues_repository=data_issues_path, 
                  var_file=variables, 
                  meta_file=metadata)

    # Perform level 1 and 2 processing
    aws.getL1()
    aws.getL2()
    # Write out level 2
    if outpath is not None:
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        if aws.L2.attrs['format'] == 'raw':
            prepare_and_write(aws.L2, outpath, aws.vars, aws.meta, '10min')
        prepare_and_write(aws.L2, outpath, aws.vars, aws.meta, '60min')
    return aws


def main():
    args = parse_arguments_l2()

    logging.basicConfig(
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    _ = get_l2(
        args.config_file,
        args.inpath,
        args.outpath,
        args.variables,
        args.metadata,
        args.data_issues_path,
    )


if __name__ == "__main__":  
    main()
        