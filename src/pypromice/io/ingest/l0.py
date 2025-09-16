"""
Module for handling configuration loading and parsing of L0 data files.

This module provides the functionalities to interpret configuration files,
detect file types for data parsing, and process L0 data into xarray.Dataset
objects with associated metadata.

The module implements explicit input file type detection and parsing logic
for different data file types including `csv_v1`, `toa5`, and `csv_default`.
Additionally, it supports post-processing for time offsets and metadata
enrichment.

Functions
---------
- load_data_files: Reads and processes multiple data files given a configuration dictionary.
- load_config: Parses a TOML configuration file and produces a dictionary of configurations.
"""
import logging
import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import toml
import xarray as xr

from . import toa5

__all__ = [
    "load_data_files",
    "load_config",
]

logger = logging.getLogger(__name__)

DELIMITER = ","
COMMENT = "#"


# ---------------------------------------------------------------------
# Explicit input file type detection
# ---------------------------------------------------------------------


def _detect_file_type(conf: Dict) -> str:
    """Classify input file type explicitly.

    Returns one of:
      - 'csv_v1'      : legacy layout with year, doy, hhmm columns
      - 'toa5'        : Campbell Scientific TOA5
      - 'csv_default' : default CSV-like with timestamp in first column
    """
    infile = conf["file"]

    # 1) Respect explicit version hint from config
    file_version = conf.get("file_version", -1)
    if file_version == 1:
        return "csv_v1"

    # 2) Peek file header to detect TOA5
    try:
        with open(infile, "r", encoding="utf-8", errors="ignore") as f:
            # Read a handful of lines to detect markers
            header_lines = []
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                header_lines.append(line.strip())
    except Exception as e:
        logger.debug(f"Failed reading header for detection from {infile}: {e}")
        # Fall back to default if we cannot read
        return "csv_default"

    # Normalize: skip blank lines
    header_nonblank = [ln for ln in header_lines if ln]

    if header_nonblank:
        first = header_nonblank[0]

        # TOA5 files have a first line starting with 'TOA5'
        if re.match(r'^["]?TOA5', first):
            return "toa5"

    # Default CSV-like parser as a safe fallback
    return "csv_default"


def _parse_csv_v1(conf) -> pd.DataFrame:
    df = pd.read_csv(
        conf["file"],
        comment=COMMENT,
        parse_dates=True,
        na_values=conf["nodata"],
        names=conf["columns"],
        sep=DELIMITER,
        skiprows=conf["skiprows"],
        skip_blank_lines=True,
        usecols=range(len(conf["columns"])),
        low_memory=False,
    )
    df["time"] = pd.to_datetime(
        df.year.astype(str)
        + df.doy.astype(str).str.zfill(3)
        + df.hhmm.astype(str).str.zfill(4),
        format="%Y%j%H%M",
    )
    return df.set_index("time")


def _parse_csv_default(conf) -> pd.DataFrame:
    df = pd.read_csv(
        conf["file"],
        comment=COMMENT,
        index_col=0,
        parse_dates=True,
        na_values=conf["nodata"],
        names=conf["columns"],
        sep=DELIMITER,
        skiprows=conf["skiprows"],
        skip_blank_lines=True,
        usecols=range(len(conf["columns"])),
        low_memory=False,
    )
    try:
        df.index = pd.to_datetime(df.index)
    except ValueError as e:
        logger.info("\n" + conf["file"])
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

    return df


def _parse_toa5(conf) -> pd.DataFrame:
    df = _parse_csv_default(conf)
    # TODO: Convert to xr.DataSet to allow for metadata enrichment
    try:
        meta = toa5.read_metadata(conf["file"])
        tao5_attrs = meta.get("attrs", {})
        tao5_attrs["file_format"] = tao5_attrs.pop("format")
    except Exception as e:
        logger.warning(f"Failed to enrich TOA5 metadata for {conf['file']}: {e}")
    return df


def load_data_file(conf: Dict) -> xr.Dataset:
    """Read L0 data file to xarray.Dataset using config dictionary and
    populate with initial metadata. The file type is detected automatically.

    Parameters
    ----------
    conf : dict
        Configuration parameters
    delimiter : str
    comment: str

    Returns
    -------
    ds : xr.Dataset
        L0 data
    """
    file_type = _detect_file_type(conf)
    logger.info(f"Detected L0 file type '{file_type}' for {conf.get('file')}")

    if file_type == "csv_v1":
        df = _parse_csv_v1(conf)
    elif file_type == "csv_default":
        df = _parse_csv_default(conf)
    elif file_type == "toa5":
        df = _parse_toa5(conf)
    else:
        raise ValueError(f"Unknown file type: {file_type}")

    df = _postprocess_dataframe(df, time_offset=conf.get("time_offset"))

    # Carry relevant metadata with ds
    ds = xr.Dataset.from_dataframe(df)
    ds.attrs["level"] = "L0"
    ds.attrs["detected_file_type"] = file_type
    ds.attrs["filename"] = Path(conf["file"]).name

    # populate meta with config keys
    skip = ["columns", "skiprows", "modem", "file", "conf", "nodata"]
    for k, v in conf.items():
        if k not in skip:
            ds.attrs[k] = v

    return ds


def load_data_files(config: Dict[str, Dict]) -> List[xr.Dataset]:
    """Load level 0 (L0) data from config mapping file names to configuration.

    Tries read_l0_file() using the config with msg_lat & msg_lon appended.
    If a pandas.errors.ParserError occurs due to mismatched columns, removes
    msg_lat & msg_lon from the config and tries again.

    Parameters
    ----------
    config : Dict[str, Dict]
        Configuration dictionary as returned by pypromice.io.load.getConfig

    Returns
    -------
    List[xr.Dataset]
        List of L0 datasets
    """
    ds_list: List[xr.Dataset] = []
    for k in config.keys():
        target = config[k]
        try:
            ds_list.append(load_data_file(target))
        except pd.errors.ParserError:
            for item in ["msg_lat", "msg_lon"]:
                if item in target["columns"]:
                    target["columns"].remove(item)
            ds_list.append(load_data_file(target))
        logger.info(f"L0 data successfully loaded from {k}")
    return ds_list


def _postprocess_dataframe(
    df: pd.DataFrame, time_offset: Optional[float] = None
) -> pd.DataFrame:
    """Apply common post-processing to parsed L0 dataframe."""
    if time_offset is not None:
        df.index = df.index + timedelta(hours=time_offset)
    # Drop SKIP columns
    for c in list(df.columns):
        if c.startswith("SKIP"):
            df.drop(columns=c, inplace=True)
    return df


def load_config(
    config_file: str | Path,
    inpath: str | Path,
    default_columns: Sequence[str] = ("msg_lat", "msg_lon"),
):
    """Load configuration from .toml file. PROMICE .toml files support defining
    features at the top level which apply to all nested properties, but do not
    overwrite nested properties if they are defined

    Parameters
    ----------
    config_file
        TOML file path
    inpath
        Input folder directory where L0 files can be found

    Returns
    -------
    conf : dict
        Configuration dictionary
    """
    config_file = Path(config_file)
    inpath = Path(inpath)

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

        conf[s]["conf"] = config_file.as_posix()
        conf[s]["file"] = os.path.join(inpath, s)
        conf[s]["columns"].extend(default_columns)

    for t in top:
        conf.pop(t)  # Delete all top level keys beause each file
    # should carry all properties with it
    for k in conf.keys():  # Check required fields are present
        for field in ["columns", "station_id", "format", "skiprows"]:
            assert field in conf[k].keys(), field + " not in config keys"
    return conf
