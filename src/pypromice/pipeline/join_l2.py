#!/usr/bin/env python
import logging, sys, os, unittest
import pandas as pd
import numpy as np
import xarray as xr
from argparse import ArgumentParser
from pypromice.core.qc.common import finalize_qc, has_qc_flags
from pypromice.io.write import prepare_and_write
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
    parser.add_argument('--keep_flagged_data', action='store_true',
                        help='Keep the original values of QC-flagged samples '
                             'and write the "<var>_qc" flag variables in the '
                             'mixed-resolution file (default: flagged data '
                             'removed, no flag variables)')
    parser.add_argument('--write_qc_flags', action='store_true',
                        help='Write the "<var>_qc" flag variables in the '
                             'mixed-resolution file while still removing '
                             'flagged data (default: no flag variables)')
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

def _combine_qc_flags(ds1, ds2, all_ds):
    """Merge the "<var>_qc" flag variables of ds1 and ds2 consistently with
    the data values chosen by combine_first (ds1 preferred, ds2 fills gaps).

    combine_first would turn the int8 flag codes into floats padded with NaN.
    Here each flag follows the data it describes: ds1's flag where ds1 has a
    value, else ds2's flag where ds2 has a value, else (no value left, i.e.
    the flagged samples were already removed) whichever flag is set. A
    missing flag variable means "never flagged" (code 0, OK).
    """
    time = all_ds.time
    qc_names = sorted({v for ds in (ds1, ds2) for v in ds.data_vars
                       if v.endswith("_qc")})
    for q in qc_names:
        v = q[:-3]
        if v not in all_ds:
            all_ds = all_ds.drop_vars(q, errors="ignore")
            continue

        def _qc_and_values(ds):
            if v in ds:
                val = ds[v].reindex(time=time)
            else:
                val = xr.full_like(all_ds[v], np.nan)
            if q in ds:
                qc = ds[q].reindex(time=time, fill_value=0)
            else:
                qc = xr.zeros_like(all_ds[v], dtype="int8")
            return qc, val

        qc1, v1 = _qc_and_values(ds1)
        qc2, v2 = _qc_and_values(ds2)

        if v in ("precip_u", "precip_l") and v in ds1 and v in ds2:
            # precipitation is concatenated in block (see join_l2): the flag
            # follows the same block
            tx_no_overlap = (qc2.sel(time=slice(ds1.time.values[-1], ds2.time.values[-1]))
                             .isel(time=slice(1, None)))
            qc = (xr.concat([ds1[q] if q in ds1 else qc1.reindex(time=ds1.time),
                             tx_no_overlap], dim="time")
                  .sortby("time").reindex(time=time, fill_value=0))
        else:
            qc = xr.where(v1.notnull(), qc1,
                          xr.where(v2.notnull(), qc2,
                                   xr.where(qc1 != 0, qc1, qc2)))

        attrs = (ds1[q].attrs if q in ds1 else ds2[q].attrs).copy()
        all_ds[q] = qc.astype("int8")
        all_ds[q].attrs = attrs
    return all_ds


def join_l2(file1,file2,outpath,variables,metadata,
            keep_flagged_data: bool = False,
            write_qc_flags: bool = False) -> xr.Dataset:
    """Merge two L2 files (file1 preferred, file2 fills gaps) and write them.

    By default, as before QC flags were introduced, flagged samples are
    removed and no "<var>_qc" variable is written. If the input files carry
    "<var>_qc" flag variables:
      * keep_flagged_data=True keeps the original values of flagged samples
        and writes the flags (in the mixed-resolution file);
      * write_qc_flags=True writes the flags but still removes the flagged
        values.
    Resampled (hourly) output never carries flags and is always clean.
    """
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

            # combine_first works terrible for accumulated values
            # we rather combine semi-accumulated precipitation in block
            for var in ['precip_u', 'precip_l']:
                if hasattr(all_ds, var):
                    if all_ds[var].notnull().any():
                        tx_data_no_overlap =(ds2[var]
                                             .sel(time=slice(ds1.time.values[-1], ds2.time.values[-1]))
                                             .isel(time=slice(1, None))) # this line prevents redundant timestamps
                        all_ds[var] = xr.concat(
                                        [ds1[var], tx_data_no_overlap], dim='time'
                                    ).sortby('time')
            if has_qc_flags(ds1) or has_qc_flags(ds2):
                all_ds = _combine_qc_flags(ds1, ds2, all_ds)
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

    # Applying the QC flags carried by the input files (if any). Without
    # either switch, flagged samples are removed and flag variables dropped.
    if has_qc_flags(all_ds):
        all_ds = finalize_qc(all_ds,
                             keep_flagged_data=keep_flagged_data,
                             keep_qc_flags=write_qc_flags)

    # Writing the mixed temporal resolution file
    prepare_and_write(all_ds, outpath, variables, metadata, 'mixed', resample = False,
                      include_qc_flags=keep_flagged_data or write_qc_flags)
    # Writing the resampled hourly file (never carries flags, always clean)
    prepare_and_write(all_ds, outpath, variables, metadata, '60min')

    logger.info(f'Files saved to {os.path.join(outpath, name)}...')
    return all_ds

def main():
    args = parse_arguments_join()
    _ = join_l2(args.file1, args.file2, args.outpath, args.variables, args.metadata,
                keep_flagged_data=args.keep_flagged_data,
                write_qc_flags=args.write_qc_flags)

if __name__ == "__main__":
    main()
