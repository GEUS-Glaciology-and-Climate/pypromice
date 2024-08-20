#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load module
"""
from datetime import timedelta
from typing import Sequence, Optional

import logging
import os
import pandas as pd
import toml
import xarray as xr

logger = logging.getLogger(__name__)


def getConfig(
    config_file, inpath, default_columns: Sequence[str] = ("msg_lat", "msg_lon")
):
    """Load configuration from .toml file. PROMICE .toml files support defining
    features at the top level which apply to all nested properties, but do not
    overwrite nested properties if they are defined

    Parameters
    ----------
    config_file : str
        TOML file path
    inpath : str
        Input folder directory where L0 files can be found

    Returns
    -------
    conf : dict
        Configuration dictionary
    """
    conf = toml.load(config_file)  # Move all top level keys to nested properties,
    top = [
        _ for _ in conf.keys() if not type(conf[_]) is dict
    ]  # if they are not already defined in the nested properties
    subs = [
        _ for _ in conf.keys() if type(conf[_]) is dict
    ]  # Insert the section name (config_file) as a file property and config file
    for s in subs:
        for t in top:
            if t not in conf[s].keys():
                conf[s][t] = conf[t]

        conf[s]["conf"] = config_file
        conf[s]["file"] = os.path.join(inpath, s)
        conf[s]["columns"].extend(default_columns)

    for t in top:
        conf.pop(t)  # Delete all top level keys beause each file
    # should carry all properties with it
    for k in conf.keys():  # Check required fields are present
        for field in ["columns", "station_id", "format", "skiprows"]:
            assert field in conf[k].keys(), field + " not in config keys"
    return conf


def getL0(
    infile: str,
    nodata,
    cols,
    skiprows,
    file_version,
    delimiter=",",
    comment="#",
    time_offset: Optional[float] = None,
) -> xr.Dataset:
    """Read L0 data file into pandas DataFrame object

    Parameters
    ----------
    infile : str
        L0 file path
    nodata : list
        List containing value for nan values and reassigned value
    cols : list
        List of columns in file
    skiprows : int
        Skip rows value
    file_version : int
        Version of L0 file
    delimiter : str
        String delimiter for L0 file
    comment : str
        Notifier of commented sections in L0 file
    time_offset : Optional[float]
        Time offset in hours for correcting for non utc time data.
    Returns
    -------
    ds : xarray.Dataset
        L0 Dataset
    """
    if file_version == 1:
        df = pd.read_csv(
            infile,
            comment=comment,
            index_col=0,
            na_values=nodata,
            names=cols,
            sep=delimiter,
            skiprows=skiprows,
            skip_blank_lines=True,
            usecols=range(len(cols)),
            low_memory=False,
        )
        df["time"] = pd.to_datetime(
            df.year.astype(str)
            + df.doy.astype(str).str.zfill(3)
            + df.hhmm.astype(str).str.zfill(4),
            format="%Y%j%H%M",
        )
        df = df.set_index("time")

    else:
        df = pd.read_csv(
            infile,
            comment=comment,
            index_col=0,
            na_values=nodata,
            names=cols,
            parse_dates=True,
            sep=delimiter,
            skiprows=skiprows,
            skip_blank_lines=True,
            usecols=range(len(cols)),
            low_memory=False,
        )
        try:
            df.index = pd.to_datetime(df.index)
        except ValueError as e:
            logger.info("\n" + infile)
            logger.info("\nValueError:")
            logger.info(e)
            logger.info("\t\t> Trying pd.to_datetime with format=mixed")
            try:
                df.index = pd.to_datetime(df.index, format="mixed")
            except Exception as e:
                logger.info("\nDateParseError:")
                logger.info(e)
                logger.info(
                    "\t\t> Trying again removing apostrophes in timestamp (old files format)"
                )
                df.index = pd.to_datetime(df.index.str.replace('"', ""))

    if time_offset is not None:
        df.index = df.index + timedelta(hours=time_offset)

    # Drop SKIP columns
    for c in df.columns:
        if c[0:4] == "SKIP":
            df.drop(columns=c, inplace=True)

    # Carry relevant metadata with ds
    ds = xr.Dataset.from_dataframe(df)
    ds.attrs["level"] = "L0"

    return ds
