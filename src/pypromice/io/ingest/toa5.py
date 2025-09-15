"""
This module provides functionality to read and convert Campbell Scientific TOA5 files into xarray
datasets. It extracts metadata, variable names, units, and statistical types, and formats the
data for further analysis.
"""
from pathlib import Path
from typing import Dict

import pandas as pd
import xarray as xr


def read_metadata(filepath: Path|str, raise_exception_on_error: bool = False)  -> Dict | None:
    # 1) Read the first four lines manually
    with open(filepath, 'r', encoding='utf-8') as f:
        # strip quotes and newline
        meta_vals = next(f).strip().replace('"', '').split(',')
        names     = next(f).strip().replace('"', '').split(',')
        units     = next(f).strip().replace('"', '').split(',')
        stats     = next(f).strip().replace('"', '').split(',')

    # Verify the format
    if meta_vals[0] != 'TOA5':
        if raise_exception_on_error:
            raise ValueError(f"Unsupported file format: {meta_vals[0]}")
        else:
            return None

    # 2) Map the first-line values to a set of metadata keys
    attrs = {
        "format"           : meta_vals[0], # e.g. TOA5
        "station_name"     : meta_vals[1], # e.g. qas_l_21_correct
        "datalogger"       : meta_vals[2], # e.g. CR1000
        "serial_number"    : meta_vals[3], # e.g. E6745
        "os_version"       : meta_vals[4], # e.g. CR1000.Std.16
        "program_name"     : meta_vals[5], # e.g. Promice2015e.CR1
        "program_signature": meta_vals[6], # e.g. 65241
        "table_name"       : meta_vals[7], # e.g. SlimTableMem
    }

    return dict(
        names=names,
        units=units,
        stats=stats,
        attrs=attrs,
    )


def read(filepath: Path, **kwargs) -> xr.DataArray | None:
    """
    Read a Campbell TOA5 file and return as an xarray.Dataset.

    - Line 1 → dataset.attrs (metadata)
    - Line 2 → variable names
    - Line 3 → variable units
    - Line 4 → statistic/type (e.g. Avg, Smp)
    - Remaining lines → data (with TIMESTAMP parsed as datetime index)
    """

    metadata = read_metadata(filepath, **kwargs)
    if metadata is None:
        return None


    # 3) Read the rest of the file into a DataFrame
    df = pd.read_csv(
        filepath,
        skiprows=4,
        header=None,
        names=metadata['names'],
        parse_dates=["TIMESTAMP"],
        index_col="TIMESTAMP",
        na_values=('NAN', '')
    )

    # 4) Build an xarray.Dataset
    ds = xr.Dataset.from_dataframe(df)
    ds.attrs.update(metadata['attrs'])

    # 5) Attach per-variable attributes
    for name, unit, stat in zip(metadata['names'], metadata['units'], metadata['stats']):
        # skip if the column wasn't read (e.g. extra blank columns)
        if name in ds:
            ds[name].attrs["units"]     = unit
            ds[name].attrs["statistic"] = stat

    return ds
