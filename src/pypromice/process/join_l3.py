#!/usr/bin/env python
import logging
import os
import pandas as pd
import xarray as xr
from argparse import ArgumentParser
from pypromice.process import (
    getVars,
    getMeta,
    addMeta,
    getColNames,
    roundValues,
    resampleL3,
    writeAll,
)
from pypromice.process.L1toL2 import correctPrecip

logger = logging.getLogger(__name__)

def parse_arguments_join() -> ArgumentParser:
    parser = ArgumentParser(description="AWS L3 joiner for merging together two L3 products, for example an L3 RAW and L3 TX data product. An hourly, daily and monthly L3 data product is outputted to the defined output path")
    parser.add_argument('-s', '--file1', type=str, required=True,
                        help='Path to source L3 file, which will be preferenced in merge process')
    parser.add_argument('-t', '--file2', type=str, required=True,
                        help='Path to target L3 file, which will be used to fill gaps in merge process')
    parser.add_argument('-o', '--outpath', default=os.getcwd(), type=str, required=True,
                        help='Path where to write output')
    parser.add_argument('-v', '--variables', default=None, type=str, required=False,
    			 help='Path to variables look-up table .csv file for variable name retained'''),
    parser.add_argument('-m', '--metadata', default=None, type=str, required=False,
    			 help='Path to metadata table .csv file for metadata information'''),
    parser.add_argument('-d', '--datatype', default='raw', type=str, required=False,
    			 help='Data type to output, raw or tx')
    return parser


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

    logger.info(f'{name} array loaded from {infile}')
    return ds, name


def join_l3(
    file1_path,
    file2_path,
    data_type,
    output_path,
    args_metadata=None,
    args_variables=None,
):

    # Check files
    if os.path.isfile(file1_path) and os.path.isfile(file2_path):

        # Load data arrays
        ds1, n1 = loadArr(file1_path)
        ds2, n2 = loadArr(file2_path)

        # Check stations match
        if n1.lower() == n2.lower():

            # Merge arrays
            logger.info(f'Combining {file1_path} with {file2_path}...')
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

    elif os.path.isfile(file1_path):
        ds1, name = loadArr(file1_path)
        logger.info(f'Only one file found {file1_path}...')
        all_ds = ds1

    elif os.path.isfile(file2_path):
        ds2, name = loadArr(file2_path)
        logger.info(f'Only one file found {file2_path}...')
        all_ds = ds2

    else:
        logger.info(f'Invalid files {file1_path}, {file2_path}')
        exit()

    # Get hourly, daily and monthly datasets
    logger.info('Resampling L3 data to hourly, daily and monthly resolutions...')
    l3_h = resampleL3(all_ds, '60min')
    l3_d = resampleL3(all_ds, '1D')
    l3_m = resampleL3(all_ds, 'M')

    logger.info(f'Adding variable information from {args_variables}...')

    # Load variables look-up table
    var = getVars(args_variables)

    # Round all values to specified decimals places
    l3_h = roundValues(l3_h, var)
    l3_d = roundValues(l3_d, var)
    l3_m = roundValues(l3_m, var)

    # Get columns to keep
    if hasattr(all_ds, 'p_l'):
        col_names = getColNames(var, 2, data_type.lower())
    else:
        col_names = getColNames(var, 1, data_type.lower())

    # Assign station id
    for l in [l3_h, l3_d, l3_m]:
        l.attrs['station_id'] = name

    # Assign metadata
    logger.info(f'Adding metadata from {args_metadata}...')
    m = getMeta(args_metadata)
    l3_h = addMeta(l3_h, m)
    l3_d = addMeta(l3_d, m)
    l3_m = addMeta(l3_m, m)

    # Set up output path
    out = os.path.join(output_path, name)

    # Write to files
    writeAll(out, name, l3_h, l3_d, l3_m, col_names)
    logger.info(f'Files saved to {os.path.join(out, name)}...')

if __name__ == "__main__":
    args = parse_arguments_join().parse_args()
    join_l3(
        data_type=args.datatype,
        file1_path=args.file1,
        file2_path=args.file2,
        args_metadata=args.metadata,
        output_path=args.outpath,
        args_variables=args.variables,
    )
