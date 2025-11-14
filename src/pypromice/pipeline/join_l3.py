#!/usr/bin/env python
import json
import logging, os, sys, toml
from argparse import ArgumentParser

from pypromice.io.ingest.git import get_commit_hash_and_check_dirty
from pypromice.pipeline.L2toL3 import post_processing_z_ice_surf

import pypromice.resources
from pypromice.io.write import prepare_and_write

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_arguments_joinl3(debug_args=None):
    parser = ArgumentParser(
        description="AWS L3 script for the processing L3 data from L2 and merging the L3 data with its historical site. An hourly, daily and monthly L3 data product is outputted to the defined output path"
    )
    parser.add_argument(
        "-c",
        "--config_folder",
        type=str,
        required=True,
        help="Path to folder with sites configuration (TOML) files",
    )
    parser.add_argument(
        "-s",
        "--site",
        default=None,
        type=str,
        required=False,
        help="Name of site to process (default: all sites are processed)",
    )

    parser.add_argument(
        "-l3", "--folder_l3", type=str, required=True, help="Path to level 3 folder"
    )
    parser.add_argument(
        "-gc",
        "--folder_gcnet",
        type=str,
        required=False,
        help="Path to GC-Net historical L1 folder",
    )
    parser.add_argument(
        "-gb",
        "--folder_glaciobasis",
        type=str,
        required=False,
        help="Path to GlacioBasis historical data folder",
    )

    parser.add_argument(
        "-o",
        "--outpath",
        default=os.getcwd(),
        type=str,
        required=True,
        help="Path where to write output",
    )

    parser.add_argument(
        "-v",
        "--variables",
        default=None,
        type=str,
        required=False,
        help="Path to variables look-up table .csv file for variable name retained" "",
    ),
    parser.add_argument(
        "-m",
        "--metadata",
        default=None,
        type=str,
        required=False,
        help="Path to metadata table .csv file for metadata information" "",
    ),

    args = parser.parse_args(args=debug_args)
    return args


def readNead(infile):
    with open(infile) as f:
        fmt = f.readline()
        assert fmt[0] == "#"
        assert fmt.split("#")[1].split()[0] == "NEAD"
        assert fmt.split("#")[1].split()[1] == "1.0"
        assert fmt.split("#")[1].split()[2] == "UTF-8"

        line = f.readline()
        assert line[0] == "#"
        assert line.split("#")[1].strip() == "[METADATA]"

        meta = {}
        fields = {}
        section = "meta"
        while True:
            line = f.readline()
            if line.strip(" ") == "#":
                continue
            if line == "# [DATA]\n":
                break  # done reading header
            if line == "# [FIELDS]\n":
                section = "fields"
                continue  # done reading header

            if line[0] == "\n":
                continue  # blank line
            assert line[0] == "#"  # if not blank, must start with "#"

            key_eq_val = line.split("#")[1].strip()
            if key_eq_val == "" or key_eq_val == None:
                continue  # Line is just "#" or "# " or "#   #"...
            assert "=" in key_eq_val, print(line, key_eq_val)
            key = key_eq_val.split("=")[0].strip()
            val = key_eq_val.split("=")[1].strip()

            # Convert from string to number if it is a number
            if val.strip("-").strip("+").replace(".", "").isdigit():
                val = float(val)
                if val == int(val):
                    val = int(val)

            if section == "meta":
                meta[key] = val
            if section == "fields":
                fields[key] = val
        # done reading header

        # Find delimiter and fields for reading NEAD as simple CSV
        assert "field_delimiter" in meta.keys()
        assert "fields" in fields.keys()
        FD = meta["field_delimiter"]
        names = [_.strip() for _ in fields.pop("fields").split(FD)]

        df = pd.read_csv(
            infile,
            comment="#",
            names=names,
            sep=FD,
            usecols=np.arange(len(names)),
            skip_blank_lines=True,
        )
        df["timestamp"] = pd.to_datetime(df.timestamp).dt.tz_localize(None)
        df = df.set_index("timestamp")
        ds = df.to_xarray()
        ds.attrs = meta

        # renaming variables if GC-Net
        file_path = pypromice.resources.DEFAULT_VARIABLES_ALIASES_GCNET_PATH
        var_name = pd.read_csv(file_path)
        var_name = var_name.set_index("old_name").GEUS_name
        msk = [v for v in var_name.index if v in ds.data_vars]
        var_name = var_name.loc[msk].to_dict()

        # combining thermocouple and CS100 temperatures
        ds["TA1"] = ds["TA1"].combine_first(ds["TA3"])
        ds["TA2"] = ds["TA2"].combine_first(ds["TA4"])

        # renaming variables to the GEUS names
        ds = ds.rename(var_name)

        # variables always dropped from the historical GC-Net files
        # could be move to the config files at some point
        standard_vars_to_drop = [
            "NR",
            "TA3",
            "TA4",
            "TA5",
            "NR_cor",
            "TA2m",
            "RH2m",
            "VW10m",
            "SZA",
            "SAA",
        ]
        standard_vars_to_drop = standard_vars_to_drop + [
            v for v in list(ds.keys()) if v.endswith("_adj_flag")
        ]

        # Drop the variables if they are present in the dataset
        ds = ds.drop_vars([var for var in standard_vars_to_drop if var in ds])

        ds = ds.rename({"timestamp": "time"})

        # in the historical GC-Net processing, periods with missing z_surf_combined
        # are filled with a constant value, these values should be removed to
        # allow a better alignement with the z_surf_combined estimated for the GEUS stations
        ds["z_surf_combined"] = ds["z_surf_combined"].where(
            ds["z_surf_combined"].diff(dim="time") != 0
        )
    return ds


def loadArr(infile, isNead, vars_to_drop=[]):
    if infile.split(".")[-1].lower() in "csv":
        if isNead:
            ds = readNead(infile)
        else:
            df = pd.read_csv(infile)
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            df = df.set_index("time")

            # renaming variables if GlacioBasis
            file_path = pypromice.resources.DEFAULT_VARIABLES_ALIASES_GLACIOBASIS_PATH
            var_name = pd.read_csv(file_path)
            var_name = var_name.set_index("old_name").GEUS_name
            msk = [v for v in var_name.index if v in df.columns]
            var_name = var_name.loc[msk].to_dict()

            # postprocessing for glaciobasis
            if 'ice_ablation' in df.columns:
                df['ice_ablation'] = (-df['ice_ablation'].diff()).cumsum()
                df['z_surf_combined'] = df['ice_ablation'].values

            # renaming variables to the GEUS names
            df = df.rename(columns=var_name)

            # variables always dropped from the historical GC-Net files
            # could be move to the config files at some point
            standard_vars_to_drop = ["I"]

            # Drop the variables if they are present in the dataset
            df = df.drop(columns=[var for var in standard_vars_to_drop if var in df.columns])
            ds = xr.Dataset.from_dataframe(df)

    elif infile.split(".")[-1].lower() in "nc":
        with xr.open_dataset(infile) as ds:
            ds.load()
        # Remove encoding attributes from NetCDF
        for varname in ds.variables:
            if ds[varname].encoding != {}:
                ds[varname].encoding = {}

    try:
        name = ds.attrs["station_name"]
    except:
        name = infile.split("/")[-1].split(".")[0].split("_hour")[0].split("_10min")[0]

    # removing specific variable from a given file
    if len(vars_to_drop) > 0:
        logger.info("Skipping %s" % (vars_to_drop))
        ds = ds.drop_vars([var for var in vars_to_drop if var in ds])

    t0 = pd.to_datetime(ds.time.values[0]).strftime("%Y-%m-%d")
    t1 = pd.to_datetime(ds.time.values[-1]).strftime("%Y-%m-%d")
    logger.info(f"loaded {infile} covering {t0} to {t1}")
    return ds, name


def align_surface_heights(data_series_new, data_series_old):
    """
    Align two surface height time series based on the gap between their end and
    start.

    If the gap between the end of `data_series_old` and the start of `data_series_new`
    is less than a week, the function aligns them based on the median value of
    the last week of `data_series_old` and the first week of `data_series_new`.
    If the gap is larger than a week, it aligns them using a linear fit. If
    there is overlap, the function uses the overlapping period to adjust the
    newer time series.

    Parameters
    ----------
    data_series_old : pandas.Series
        The older time series data.
    data_series_new : pandas.Series
        The newer time series data.

    Returns
    -------
    numpy.ndarray
        Array containing the aligned newer time series data.
    """
    # Get the first and last valid indices of both series
    last_old_idx = data_series_old.last_valid_index()
    first_new_idx = data_series_new.first_valid_index()

    # Check for overlap
    if first_new_idx <= last_old_idx:
        # Find the overlapping period
        overlap_start = first_new_idx
        overlap_end = min(last_old_idx, overlap_start + pd.to_timedelta("7D"))

        # Compute the median values for the overlapping period
        overlap_old = data_series_old[overlap_start:overlap_end].median()
        overlap_new = data_series_new[overlap_start:overlap_end].median()

        if np.isnan(overlap_old) or np.isnan(overlap_new):
            overlap_end = min(last_old_idx, data_series_new.last_valid_index())

            # Compute the median values for the overlapping period
            overlap_old = data_series_old[overlap_start:overlap_end].median()
            overlap_new = data_series_new[overlap_start:overlap_end].median()

        # Align based on the overlapping median values
        data_series_new = data_series_new - overlap_new + overlap_old

    elif (first_new_idx - last_old_idx).days <= 7:
        # Compute the median of the last week of data in the old series
        last_week_old = data_series_old[
            last_old_idx - pd.Timedelta(weeks=1) : last_old_idx
        ].median()

        # Compute the median of the first week of data in the new series
        first_week_new = data_series_new[
            first_new_idx : first_new_idx + pd.Timedelta(weeks=1)
        ].median()

        # Align based on the median values
        data_series_new = data_series_new - first_week_new + last_week_old
    else:
        # Perform a linear fit on the last 5x365x24 non-nan values
        hours_in_5_years = 5 * 365 * 24

        # Drop NaN values and extract the last `hours_in_5_years` non-NaN data points
        data_series_old_nonan = data_series_old.dropna()
        data_series_old_last_5_years = data_series_old_nonan.iloc[
            -min(len(data_series_old), hours_in_5_years):
        ]

        # Perform a linear fit on the last 5 years of data
        fit = np.polyfit(
            data_series_old_last_5_years.index.astype("int64"),
            data_series_old_last_5_years.values,
            1,
        )
        fit_fn = np.poly1d(fit)

        data_series_new = (
            data_series_new.values
            + fit_fn(data_series_new.index.astype("int64")[0])
            - data_series_new[first_new_idx]
        )

    return data_series_new

def load_station_configs(config_folder: str, target_station_site: str) -> list:
    """
    Load station configuration files and extract those matching the target site.

    Args:
        config_folder (str): Path to the folder containing the TOML configuration files.
        target_station_site (str): Name of the site to search for.

    Returns:
        list: List of station information dictionaries matching the target site.
    """
    station_info_list = []
    found_as_station = False
    for filename in os.listdir(config_folder):
        if not filename.endswith(".toml"):
            continue
        with open(os.path.join(config_folder, filename), "r") as f:
            data = toml.load(f)
        stid = data.get("stid")
        station_site = data.get("station_site")

        if stid == target_station_site:
            found_as_station = True
        if station_site == target_station_site and stid:
            station_info_list.append(data.copy())

    if len(station_info_list) == 0 and not found_as_station:
        logger.error(
            f"\n***\nNo station_configuration file found for {target_station_site}.\nProcessing it as a single-station PROMICE site.\n***"
        )
        station_info_list.append({
            "stid": target_station_site,
            "station_site": target_station_site,
            "project": "PROMICE",
            "location_type": "ice sheet",
        })
    elif len(station_info_list) == 0:
        logger.error(
            f'\n***\nThe name "{target_station_site}" passed to join_l3 is a station name and not a site name.\n***'
        )
    return station_info_list


def find_station_filepath(station_info: dict, folder_l3: str,
                          folder_gcnet: str = None, folder_glaciobasis: str = None) -> tuple:
    """
    Determine the path to a station data file based on project type and available folders.

    Args:
        station_info (dict): Dictionary containing station metadata (project, stid, etc.).
        folder_l3 (str): Path to the PROMICE Level 3 data folder.
        folder_gcnet (str, optional): Path to the GC-Net data folder.
        folder_glaciobasis (str, optional): Path to the GlacioBasis data folder.

    Returns:
        tuple: (filepath, isNead)
            filepath (str or None): Path to the located data file.
            isNead (bool): Whether the file is NEAD formatted.
    """
    stid = station_info["stid"]
    filepath = os.path.join(folder_l3, f"{stid}/{stid}_hour.nc")
    isNead = False

    if not os.path.isfile(filepath):
        if folder_gcnet and station_info["project"].lower() in ["historical gc-net"]:
            filepath = os.path.join(folder_gcnet, f"{stid}.csv")
            isNead = True
        elif folder_glaciobasis and station_info["project"].lower() in ["glaciobasis"]:
            filepath = os.path.join(folder_glaciobasis, f"{stid.replace('_hist','')}.csv")

    if not os.path.isfile(filepath):
        logger.error(
            f"\n***\n{stid} listed as a station but not found in {folder_l3}, {folder_gcnet} nor {folder_glaciobasis}\n***"
        )
        return None, isNead

    return filepath, isNead

def get_valid_time_block(station_info: dict, filepath: str, isNead: bool,
                         tested_vars: list = ["t_u", "dsr"],
                         min_gap: str = "30D") -> list[dict]:
    """
    Load one station dataset and split it into continuous time blocks
    where tested variables are not simultaneously NaN for more than `min_gap`.

    Args:
        station_info (dict): Station configuration dictionary (includes 'stid' and 'project').
        filepath (str): Path to the dataset file.
        isNead (bool): Whether the file follows the NEAD format.
        tested_vars (list): Variables used to determine data validity.
        min_gap (str): Minimum NaN-only gap duration (e.g., '30D') that triggers a split.

    Returns:
        list[dict]: List of valid sub-blocks with dataset, time span, and station metadata.
    """
    try:
        ds, _ = loadArr(filepath, isNead, vars_to_drop=station_info.get("skipped_variables", []))
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return []

    if "time" not in ds:
        logger.error(f"No time coordinate in {filepath}")
        return []

    # build mask where at least one tested variable is valid
    available_mask = np.zeros(ds.time.size, dtype=bool)
    for var in tested_vars:
        if var in ds:
            available_mask |= ~np.isnan(ds[var].values)

    if not available_mask.any():
        logger.warning(f"No valid data for {station_info['stid']}")
        return []

    time_index = pd.to_datetime(ds.time.values)
    valid_df = pd.Series(available_mask, index=time_index)

    # find start and end of gaps longer than threshold
    gap_threshold = pd.Timedelta(min_gap)
    gap_boundaries = []
    last_valid = None
    for t, ok in valid_df.items():
        if ok:
            last_valid = t
        elif last_valid is not None:
            # detect next valid after gap
            next_valids = valid_df.loc[t:].index[valid_df.loc[t:].values]
            if len(next_valids) > 0:
                next_valid = next_valids[0]
                gap_len = next_valid - last_valid
                if gap_len > gap_threshold:
                    gap_boundaries.append((last_valid, next_valid))
                    last_valid = None

    # compute segments based on those long gaps
    all_times = [time_index.min()] + [b[0] for b in gap_boundaries] + [b[1] for b in gap_boundaries] + [time_index.max()]
    all_times = sorted(list(set(all_times)))

    segments = [(all_times[i], all_times[i + 1]) for i in range(len(all_times) - 1)]

    blocks = []
    for seg_start, seg_end in segments:
        ds_seg = ds.sel(time=slice(seg_start, seg_end))
        if ds_seg.time.size == 0:
            continue

        blocks.append({
            "stid": station_info["stid"],
            "start_time": np.datetime64(seg_start),
            "end_time": np.datetime64(seg_end),
            "dataset": ds_seg,
            **station_info
        })

    return blocks

def resolve_block_overlap(blocks: list) -> list:
    """
    Slice station data blocks into non-overlapping time segments,
    preferring newer data where periods overlap and filling gaps
    with older station data.

    Args:
        blocks (list): List of dicts, each containing:
                       - 'dataset' (xarray.Dataset)
                       - 'start_time' (datetime64)
                       - 'end_time' (datetime64)
                       - metadata fields

    Returns:
        list: List of new, non-overlapping blocks ordered from newest to oldest,
              each with consistent metadata and trimmed datasets.
    """
    # Sort by start time (earliest first) and ensure unique objects
    blocks = sorted(blocks, key=lambda x: x["start_time"])
    # Collect all unique time breakpoints across datasets
    all_edges = set()
    for b in blocks:
        if np.isnat(b["start_time"]) or np.isnat(b["end_time"]):
            continue
        all_edges.add(pd.to_datetime(b["start_time"]))
        all_edges.add(pd.to_datetime(b["end_time"]))
    # Sort breakpoints chronologically
    all_edges = sorted(all_edges)

    # Build consecutive time intervals between edges
    segments = [(all_edges[i], all_edges[i + 1]) for i in range(len(all_edges) - 1)]

    resolved_blocks = []

    # For each segment, determine which datasets cover it
    for seg_start, seg_end in reversed(segments):  # newest last → reversed gives newer first
        covering = [
            b for b in blocks
            if (not np.isnat(b["start_time"]) and not np.isnat(b["end_time"]))
            and (b["start_time"] <= seg_start) and (b["end_time"] >= seg_end)
        ]
        if not covering:
            continue

        # prefer the newest dataset (last in original order or latest start_time)
        chosen = covering[-1]
        ds_seg = chosen["dataset"].sel(time=slice(seg_start, seg_end))
        ds_seg.attrs = chosen["dataset"].attrs

        # skip empty slices
        if ds_seg.time.size == 0:
            continue

        new_block = {
            **{k: v for k, v in chosen.items() if k != "dataset"},
            "dataset": ds_seg,
            "start_time": np.datetime64(seg_start),
            "end_time": np.datetime64(seg_end)
        }
        resolved_blocks.append(new_block)

    # Sort descending (newest first)
    resolved_blocks = sorted(resolved_blocks, key=lambda b: b["start_time"], reverse=True)
    return resolved_blocks


def build_station_data_blocks(config_folder: str, target_station_site: str,
                              folder_l3: str, folder_gcnet: str = None,
                              folder_glaciobasis: str = None) -> list:
    """
    Build a list of station datasets and metadata blocks for a specific site.

    Args:
        config_folder (str): Path to folder containing station TOML configuration files.
        target_station_site (str): Target site name to process (e.g., "ZAC").
        folder_l3 (str): Path to PROMICE Level 3 data folder.
        folder_gcnet (str, optional): Path to GC-Net data folder.
        folder_glaciobasis (str, optional): Path to GlacioBasis data folder.

    Returns:
        list: List of tuples (xarray.Dataset, dict) where each dataset corresponds
              to one station data block and its associated metadata.
    """
    station_info_list = load_station_configs(config_folder, target_station_site)
    blocks = []

    for station_info in station_info_list:
        filepath, isNead = find_station_filepath(station_info, folder_l3, folder_gcnet, folder_glaciobasis)
        if not filepath:
            continue

        blocks.extend(get_valid_time_block(station_info, filepath, isNead))


    blocks = resolve_block_overlap(blocks)
    logger.info("Resolved non-overlapping station data blocks:")
    for b in blocks:
        stid = b.get("stid", "unknown")
        t0 = pd.to_datetime(b["start_time"]).strftime("%Y-%m-%d")
        t1 = pd.to_datetime(b["end_time"]).strftime("%Y-%m-%d")
        logger.info(f"  {stid:10s}  {t0}  →  {t1}")

    list_station_data = [(b["dataset"], {k: v for k, v in b.items() if k != "dataset"}) for b in blocks]

    return list_station_data

def join_l3(config_folder, site, folder_l3, folder_gcnet, outpath, variables, metadata,
            folder_glaciobasis="undefined_folder_glaciobasis"):

    # Read the datasets and store them into a list along with their latest timestamp and station info
    sorted_list_station_data = build_station_data_blocks(
        config_folder, site, folder_l3, folder_gcnet, folder_glaciobasis
    )

    sorted_stids = [info["stid"] for _, info in sorted_list_station_data]
    logger.info("joining %s" % " ".join(sorted_stids))

    l3_merged = None

    for l3, station_info in sorted_list_station_data:

        stid = station_info["stid"]

        if l3_merged is None:
            # saving attributes of stid
            st_attrs = {}
            st_attrs[stid] = l3.attrs.copy()
            # adding timestamps info
            st_attrs[stid]["first_timestamp"] = (
                l3.time.isel(time=0).dt.strftime(date_format="%Y-%m-%d %H:%M:%S").item()
            )
            st_attrs[stid]["last_timestamp"] = (
                l3.time.isel(time=-1)
                .dt.strftime(date_format="%Y-%m-%d %H:%M:%S")
                .item()
            )

            # then stripping attributes
            attrs_list = list(l3.attrs.keys())
            for k in attrs_list:
                del l3.attrs[k]

            # initializing l3_merged with l3
            l3_merged = l3.copy()

            # creating the station_attributes attribute in l3_merged
            l3_merged.attrs["stations_attributes"] = st_attrs

        else:
            # if l3 (older data) is missing variables compared to l3_merged (newer data)
            # , then we fill them with nan
            for v in l3_merged.data_vars:
                if v not in l3.data_vars:
                    l3[v] = l3.t_u * np.nan
            for v in l3.data_vars:
                if v not in l3_merged.data_vars:
                    l3_merged[v] = l3_merged.t_u * np.nan

            # saving attributes of station under an attribute called $stid
            st_attrs = l3_merged.attrs.get("stations_attributes", {})
            st_attrs[stid] = l3.attrs.copy()
            l3_merged.attrs["stations_attributes"] = st_attrs

            # then stripping attributes
            attrs_list = list(l3.attrs.keys())
            for k in attrs_list:
                del l3.attrs[k]

            l3_merged.attrs["stations_attributes"][stid]["first_timestamp"] = (
                l3.time.isel(time=0).dt.strftime(date_format="%Y-%m-%d %H:%M:%S").item()
            )
            l3_merged.attrs["stations_attributes"][stid]["last_timestamp"] = (
                l3_merged.time.isel(time=0)
                .dt.strftime(date_format="%Y-%m-%d %H:%M:%S")
                .item()
            )
            logger.info(
    f"adding {stid} {st_attrs[stid]['first_timestamp']} → {st_attrs[stid]['last_timestamp']}"
)


            # adjusting surface height in the most recent data (l3_merged)
            # so that it shows continuity with the older data (l3)
            if "z_surf_combined" in l3_merged.keys() and "z_surf_combined" in l3.keys():
                if (
                    l3_merged.z_surf_combined.notnull().any()
                    and l3.z_surf_combined.notnull().any()
                ):
                    l3_merged["z_surf_combined"] = (
                        "time",
                        align_surface_heights(
                            l3_merged.z_surf_combined.to_series(),
                            l3.z_surf_combined.to_series(),
                        ),
                    )
            if "z_ice_surf" in l3_merged.keys() and "z_ice_surf" in l3.keys():
                if (
                    l3_merged.z_ice_surf.notnull().any()
                    and l3.z_ice_surf.notnull().any()
                ):
                    l3_merged["z_ice_surf"] = (
                        "time",
                        align_surface_heights(
                            l3_merged.z_ice_surf.to_series(), l3.z_ice_surf.to_series()
                        ),
                    )

            # if st_attrs[stid]['site_type'] == 'accumulation':
            #     if "z_ice_surf" in l3_merged.keys() and "z_ice_surf" in l3.keys():
            #         if (
            #             l3_merged.z_ice_surf.notnull().any()
            #             and l3.z_ice_surf.notnull().any()
            #         ):
            #             l3_merged["z_ice_surf"] = (
            #                 "time",
            #                 align_surface_heights(
            #                     l3_merged.z_ice_surf.to_series(), l3.z_ice_surf.to_series()
            #                 ),
            #             )

            # saves attributes
            attrs = l3_merged.attrs
            # merging by time block
            t_start = l3.time.values[0]
            t_stop = l3_merged.time.isel(time=0) - pd.Timedelta(minutes=5)
            # subtracting 5 minutes to make sure timestamps are not added once
            # from l3 and once from l3_merge

            l3_part = l3.sel(time=slice(t_start, t_stop))

            # we don't want the first timestamp of l3_merged to also be in l3
            if l3_part.time.values[-1] == t_stop:
                l3_part = l3_part.isel(time=slice(None, -1))

            l3_merged = xr.concat([l3_part, l3_merged], dim="time")

            # restauring attributes
            l3_merged.attrs = attrs

    # Assign site id
    if not l3_merged:
        logger.error("No level 3 station data file found for " + site)
        return None, sorted_list_station_data
    l3_merged.attrs["site_id"] = site
    l3_merged.attrs["stations"] = " ".join(sorted_stids)
    l3_merged.attrs["level"] = "L3"
    l3_merged.attrs["project"] = sorted_list_station_data[0][1]["project"]
    l3_merged.attrs["location_type"] = sorted_list_station_data[0][1]["location_type"]

    first_station_name = list(l3_merged.attrs["stations_attributes"].keys())[0]
    if l3_merged.attrs["stations_attributes"][first_station_name]['site_type'] == 'ablation':
        l3_merged["z_ice_surf"] = post_processing_z_ice_surf(l3_merged["z_ice_surf"],
                                                         l3_merged["z_surf_combined"],
                                                         l3_merged["z_ice_surf"])

    # adding commit version for the site config and historical data
    site_source = dict(
        site_config_source_hash=get_commit_hash_and_check_dirty(config_folder),
        gcnet_source_hash=get_commit_hash_and_check_dirty(folder_gcnet),
    )

    for stid, station_attributes in l3_merged.attrs["stations_attributes"].items():
        if "source" in station_attributes.keys():
            station_source = json.loads(station_attributes["source"])
            for k, v in station_source.items():
                if k in site_source and site_source[k] != v:
                    site_source[k] = "multiple"
                else:
                    site_source[k] = v
    l3_merged.attrs["source"] = json.dumps(site_source)


    # writing to file
    v = pypromice.resources.load_variables(variables)
    m = pypromice.resources.load_metadata(metadata)
    if outpath is not None:
        prepare_and_write(l3_merged, outpath, v, m, "60min", nc_compression=True)
        prepare_and_write(l3_merged, outpath, v, m, "1D", nc_compression=True)
        prepare_and_write(l3_merged, outpath, v, m, "MS", nc_compression=True)
    return l3_merged, sorted_list_station_data


def main():
    args = parse_arguments_joinl3()
    _, _ = join_l3(
        args.config_folder,
        args.site,
        args.folder_l3,
        args.folder_gcnet,
        args.folder_glaciobasis,
        args.outpath,
        args.variables,
        args.metadata,
    )


if __name__ == "__main__":
    main()
