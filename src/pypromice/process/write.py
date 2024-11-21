#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing all the functions needed to prepare and AWS data
"""
import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pypromice.process.resample import resample_dataset
import pypromice.resources

logger = logging.getLogger(__name__)


def prepare_and_write(
    dataset,
        output_path: Path | str,
        vars_df=None,
        meta_dict=None,
        time="60min",
        resample=True,
        nc_compression:bool=False,
):
    """Prepare data with resampling, formating and metadata population; then
    write data to .nc and .csv hourly and daily files

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to write to file
    output_path : Path|str
        Output directory
    vars_df : pandas.DataFrame
        Variables look-up table dataframe
    meta_dict : dictionary
        Metadata dictionary to write to dataset
    time : str
        Resampling interval for output dataset
    """
    # Resample dataset
    if isinstance(output_path, str):
        output_path = Path(output_path)

    if resample:
        d2 = resample_dataset(dataset, time)
        logger.info("Resampling to " + str(time))
        if len(d2.time) == 1:
            logger.warning(
                "Output of resample has length 1. Not enough data to calculate daily/monthly average."
            )
            return None
    else:
        d2 = dataset.copy()

    # Reformat time
    d2 = reformat_time(d2)

    # finding station/site name
    if "station_id" in d2.attrs.keys():
        name = d2.attrs["station_id"]
    else:
        name = d2.attrs["site_id"]

    # Reformat longitude (to negative values)
    if "gps_lon" in d2.keys():
        d2 = reformat_lon(d2)
    else:
        logger.info("%s does not have gps_lon" % name)

    # Add variable attributes and metadata
    if vars_df is None:
        vars_df = pypromice.resources.load_variables()
    if meta_dict is None:
        meta_dict = pypromice.resources.load_metadata()

    d2 = addVars(d2, vars_df)
    d2 = addMeta(d2, meta_dict)

    # Round all values to specified decimals places
    d2 = roundValues(d2, vars_df)

    # Get variable names to write out
    if "site_id" in d2.attrs.keys():
        remove_nan_fields = True
    else:
        remove_nan_fields = False
    col_names = getColNames(vars_df, d2, remove_nan_fields=remove_nan_fields)

    # Define filename based on resample rate
    t = int(pd.Timedelta((d2["time"][1] - d2["time"][0]).values).total_seconds())

    # Create out directory
    output_dir = output_path / name
    output_dir.mkdir(exist_ok=True, parents=True)

    if t == 600:
        out_csv = output_dir / f"{name}_10min.csv"
        out_nc = output_dir / f"{name}_10min.nc"
    elif t == 3600:
        out_csv = output_dir / f"{name}_hour.csv"
        out_nc = output_dir / f"{name}_hour.nc"
    elif t == 86400:
        # removing instantaneous values from daily and monthly files
        for v in col_names:
            if v in ['p_i', 't_i', 'rh_i', 'wspd_i', 'wdir_i', 'wspd_x_i', 'wspd_y_i']:
                col_names.remove(v)
        out_csv = output_dir / f"{name}_day.csv"
        out_nc = output_dir / f"{name}_day.nc"
    else:
        # removing instantaneous values from daily and monthly files
        for v in col_names:
            if v in ['p_i', 't_i', 'rh_i', 'wspd_i', 'wdir_i', 'wspd_x_i', 'wspd_y_i']:
                col_names.remove(v)
        out_csv = output_dir / f"{name}_month.csv"
        out_nc = output_dir / f"{name}_month.nc"

    # Write to csv file
    logger.info("Writing to files...")
    writeCSV(out_csv, d2, col_names)

    # Write to netcdf file
    writeNC(out_nc, d2, col_names, compression=nc_compression)
    logger.info(f"Written to {out_csv}")
    logger.info(f"Written to {out_nc}")


def writeCSV(outfile, Lx, csv_order):
    """Write data product to CSV file

    Parameters
    ----------
    outfile : str
        Output file path
    Lx : xr.Dataset
        Dataset to write to file
    csv_order : list
        List order of variables
    """
    Lcsv = Lx.to_dataframe().dropna(how="all")
    if csv_order is not None:
        names = [c for c in csv_order if c in list(Lcsv.columns)]
        Lcsv = Lcsv[names]
    Lcsv.to_csv(outfile)


def writeNC(outfile, Lx, col_names=None, compression=False):
    """Write data product to NetCDF file with compression

    Parameters
    ----------
    outfile : str
        Output file path
    Lx : xr.Dataset
        Dataset to write to file
    """
    if os.path.isfile(outfile):
        os.remove(outfile)
    if col_names is not None:
        names = [c for c in col_names if c in list(Lx.keys())]
    else:
        names = list(Lx.keys())

    encoding = {var: dict() for var in names}

    if compression:
        comp = dict(zlib=True, complevel=4)
        for var in names:
            encoding[var].update(comp)

    Lx[names].to_netcdf(outfile, mode="w", format="NETCDF4", compute=True, encoding=encoding)


def getColNames(vars_df, ds, remove_nan_fields=False):
    """
     Get variable names for a given dataset with respect to its type and processing level

     The dataset must have the the following attributes:
     * level
     * number_of_booms when the processing level is <= 2

     This is mainly for exporting purposes.

    Parameters
     -------
     list
         Variable names
    """
    # selecting variable list based on level
    vars_df = vars_df.loc[vars_df[ds.attrs["level"]] == 1]

    # selecting variable list based on geometry
    if ds.attrs["level"] in ["L0", "L1", "L2"]:
        if ds.attrs["number_of_booms"] == 1:
            vars_df = vars_df.loc[vars_df["station_type"].isin(["one-boom", "all"])]
        elif ds.attrs["number_of_booms"] == 2:
            vars_df = vars_df.loc[vars_df["station_type"].isin(["two-boom", "all"])]

    var_list = list(vars_df.index)
    if remove_nan_fields:
        for v in var_list:
            if v not in ds.keys():
                var_list.remove(v)
                continue
            if ds[v].isnull().all():
                var_list.remove(v)
    return var_list


def addVars(ds, variables):
    """Add variable attributes from file to dataset

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to add variable attributes to
    variables : pandas.DataFrame
        Variables lookup table file

    Returns
    -------
    ds : xarray.Dataset
        Dataset with metadata
    """
    for k in ds.keys():
        if k not in variables.index:
            continue
        ds[k].attrs["standard_name"] = variables.loc[k]["standard_name"]
        ds[k].attrs["long_name"] = variables.loc[k]["long_name"]
        ds[k].attrs["units"] = variables.loc[k]["units"]
        ds[k].attrs["coverage_content_type"] = variables.loc[k]["coverage_content_type"]
        ds[k].attrs["coordinates"] = variables.loc[k]["coordinates"]
    return ds


def addMeta(ds, meta):
    """Add metadata attributes from file to dataset

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to add metadata attributes to
    meta : dict
        Metadata file

    Returns
    -------
    ds : xarray.Dataset
        Dataset with metadata
    """

    # a static latitude, longitude and altitude is saved as attribute along its origin
    var_alias = {"lat": "latitude", "lon": "longitude", "alt": "altitude"}
    for v in ["lat", "lon", "alt"]:
        # saving the reference latitude/longitude/altitude
        original_value = np.nan
        if var_alias[v] in ds.attrs.keys():
            original_value = ds.attrs[var_alias[v]]
        if v in ds.keys():
            # if possible, replacing it with average coordinates based on the extra/interpolated coords
            ds.attrs[var_alias[v]] = ds[v].mean().item()
            ds.attrs[var_alias[v] + "_origin"] = (
                "average of gap-filled postprocessed " + v
            )
        elif "gps_" + v in ds.keys():
            # if possible, replacing it with average coordinates based on the measured coords (can be gappy)
            ds.attrs[var_alias[v]] = ds["gps_" + v].mean().item()
            ds.attrs[var_alias[v] + "_origin"] = (
                "average of GPS-measured " + v + ", potentially including gaps"
            )

        if np.isnan(ds.attrs[var_alias[v]]):
            # if no better data was available to update the coordinate, then we
            # re-use the original value
            ds.attrs[var_alias[v]] = original_value
            ds.attrs[var_alias[v] + "_origin"] = "reference value, origin unknown"

    # Attribute convention for data discovery
    # https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3

    # Determine the temporal resolution
    sample_rate = "unknown_sample_rate"
    if len(ds["time"]) > 1:
        time_diff = pd.Timedelta((ds["time"][1] - ds["time"][0]).values)
        if time_diff == pd.Timedelta("10min"):
            sample_rate = "10min"
        elif time_diff == pd.Timedelta("1h"):
            sample_rate = "hourly"
        elif time_diff == pd.Timedelta("1D"):
            sample_rate = "daily"
        elif 28 <= time_diff.days <= 31:
            sample_rate = "monthly"

    if "station_id" in ds.attrs.keys():
        id_components = [
            "dk",
            "geus",
            "promice",
            "station",
            ds.attrs["station_id"],
            ds.attrs["level"],
            sample_rate,
        ]
        ds.attrs["id"] = ".".join(id_components)
    else:
        id_components = [
            "dk",
            "geus",
            "promice",
            "site",
            ds.attrs["site_id"],
            ds.attrs["level"],
            sample_rate,
        ]
        ds.attrs["id"] = ".".join(id_components)

    ds.attrs["history"] = "Generated on " + datetime.datetime.utcnow().isoformat()
    ds.attrs["date_created"] = str(datetime.datetime.now().isoformat())
    ds.attrs["date_modified"] = ds.attrs["date_created"]
    ds.attrs["date_issued"] = ds.attrs["date_created"]
    ds.attrs["date_metadata_modified"] = ds.attrs["date_created"]
    ds.attrs["processing_level"] = ds.attrs["level"].replace("L", "Level ")

    id = ds.attrs.get('station_id', ds.attrs.get('site_id'))
    title_string_format = "AWS measurements from {id} processed to {processing_level}. {sample_rate} average."
    ds.attrs["title"] = title_string_format.format(
        id=id,
        processing_level=ds.attrs["processing_level"].lower(),
        sample_rate=sample_rate.capitalize(),
    )

    if "lat" in ds.keys():
        lat_min = ds["lat"].min().values
        lat_max = ds["lat"].max().values
    elif "gps_lat" in ds.keys():
        lat_min = ds["gps_lat"].min().values
        lat_max = ds["gps_lat"].max().values
    elif "latitude" in ds.attrs.keys():
        lat_min = ds.attrs["latitude"]
        lat_max = ds.attrs["latitude"]
    else:
        lat_min = np.nan
        lat_max = np.nan

    if "lon" in ds.keys():
        lon_min = ds["lon"].min().values
        lon_max = ds["lon"].max().values
    elif "gps_lon" in ds.keys():
        lon_min = ds["gps_lon"].min().values
        lon_max = ds["gps_lon"].max().values
    elif "longitude" in ds.attrs.keys():
        lon_min = ds.attrs["longitude"]
        lon_max = ds.attrs["longitude"]
    else:
        lon_min = np.nan
        lon_max = np.nan

    if "alt" in ds.keys():
        alt_min = ds["alt"].min().values
        alt_max = ds["alt"].max().values
    elif "gps_alt" in ds.keys():
        alt_min = ds["gps_alt"].min().values
        alt_max = ds["gps_alt"].max().values
    elif "altitude" in ds.attrs.keys():
        alt_min = ds.attrs["altitude"]
        alt_max = ds.attrs["altitude"]
    else:
        alt_min = np.nan
        alt_max = np.nan

    ds.attrs["geospatial_bounds"] = (
        "POLYGON(("
        + f"{lat_min} {lon_min}, "
        + f"{lat_min} {lon_max}, "
        + f"{lat_max} {lon_max}, "
        + f"{lat_max} {lon_min}, "
        + f"{lat_min} {lon_min}))"
    )

    ds.attrs["geospatial_lat_min"] = str(lat_min)
    ds.attrs["geospatial_lat_max"] = str(lat_max)
    ds.attrs["geospatial_lon_min"] = str(lon_min)
    ds.attrs["geospatial_lon_max"] = str(lon_max)
    ds.attrs["geospatial_vertical_min"] = str(alt_min)
    ds.attrs["geospatial_vertical_max"] = str(alt_max)

    ds.attrs["geospatial_vertical_positive"] = "up"
    ds.attrs["time_coverage_start"] = str(ds["time"][0].values)
    ds.attrs["time_coverage_end"] = str(ds["time"][-1].values)

    # https://www.digi.com/resources/documentation/digidocs/90001437-13/reference/r_iso_8601_duration_format.htm
    try:
        ds.attrs["time_coverage_duration"] = str(
            pd.Timedelta((ds["time"][-1] - ds["time"][0]).values).isoformat()
        )
        ds.attrs["time_coverage_resolution"] = str(
            pd.Timedelta((ds["time"][1] - ds["time"][0]).values).isoformat()
        )
    except:
        ds.attrs["time_coverage_duration"] = str(pd.Timedelta(0).isoformat())
        ds.attrs["time_coverage_resolution"] = str(pd.Timedelta(0).isoformat())

    # Note: int64 dtype (long int) is incompatible with OPeNDAP access via THREDDS for NetCDF files
    # See https://stackoverflow.com/questions/48895227/output-int32-time-dimension-in-netcdf-using-xarray
    ds.time.encoding["dtype"] = "i4"  # 32-bit signed integer
    # ds.time.encoding["calendar"] = 'proleptic_gregorian' # this is default

    # Load metadata attributes and add to Dataset
    [_addAttr(ds, key, value) for key, value in meta.items()]

    # Check attribute formating
    for k, v in ds.attrs.items():
        if not isinstance(v, str) or not isinstance(v, int):
            ds.attrs[k] = str(v)
    return ds


def _addAttr(ds, key, value):
    """Add attribute to xarray dataset

    ds : xr.Dataset
        Dataset to add attribute to
    key : str
        Attribute name, with "." denoting variable attributes
    value : str/int
        Value for attribute"""
    if len(key.split(".")) == 2:
        try:
            ds[key.split(".")[0]].attrs[key.split(".")[1]] = str(value)
        except:
            pass
            # logger.info(f'Unable to add metadata to {key.split(".")[0]}')
    else:
        ds.attrs[key] = value


def roundValues(ds, df, col="max_decimals"):
    """Round all variable values in data array based on pre-defined rounding
    value in variables look-up table DataFrame

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to round values in
    df : pd.Dataframe
        Variable look-up table with rounding values
    col : str
        Column in variable look-up table that contains rounding values. The
        default is "max_decimals"
    """
    df = df[col]
    df = df.dropna(how="all")
    for var in df.index:
        if var not in list(ds.variables):
            continue
        if df[var] is not np.nan:
            ds[var] = ds[var].round(decimals=int(df[var]))
    return ds


def reformat_time(dataset):
    """Re-format time"""
    t = dataset["time"].values
    dataset["time"] = list(t)
    return dataset


def reformat_lon(dataset, exempt=["UWN", "Roof_GEUS", "Roof_PROMICE", "ORO"]):
    """Switch gps_lon to negative values (degrees_east). We do this here, and
    NOT in addMeta, otherwise we switch back to positive when calling getMeta
    in joinL2"""
    if "station_id" in dataset.attrs.keys():
        id = dataset.attrs["station_id"]
    else:
        id = dataset.attrs["site_id"]

    if id not in exempt:
        if "gps_lon" not in dataset.keys():
            return dataset
        dataset["gps_lon"] = np.abs(dataset["gps_lon"]) * -1
        if "lon" not in dataset.keys():
            return dataset
        dataset["lon"] = np.abs(dataset["lon"]) * -1
    return dataset
