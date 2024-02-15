#!/usr/bin/env python

"""
Command-line script for running BUFR file generation
Created: Dec 20, 2022
Author: Patrick Wright, GEUS
"""
import argparse
import glob
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Mapping, Any

import numpy as np
import pandas as pd

from pypromice.postprocess import wmo_config
from pypromice.postprocess.bufr_utilities import write_bufr_message, BUFRVariables
from pypromice.postprocess.real_time_utilities import get_latest_data

logger = logging.getLogger(__name__)


def parse_arguments_bufr() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--store_positions",
        "--positions",
        action="store_true",
        required=False,
        default=False,
        help="If included (True), make a positions dict and output AWS_latest_locations.csv file.",
    )

    parser.add_argument(
        "--positions-filepath",
        default="../aws-l3/AWS_latest_locations.csv",
        type=str,
        required=False,
        help="Path to write AWS_latest_locations.csv file.",
    )

    parser.add_argument(
        "--time-limit",
        default="3M",
        type=str,
        required=False,
        help="Previous time to limit dataframe before applying linear regression.",
    )

    parser.add_argument(
        "--l3-filepath",
        default="../aws-l3/tx/*/*_hour.csv",
        type=str,
        required=False,
        help="Path to L3 tx .csv files.",
    )

    parser.add_argument(
        "--bufr-out",
        default="src/pypromice/postprocess/BUFR_out/",
        type=str,
        required=False,
        help="Path to the BUFR out directory.",
    )

    parser.add_argument(
        "--timestamps-pickle-filepath",
        default="../pypromice/src/pypromice/postprocess/latest_timestamps.pickle",
        type=str,
        required=False,
        help="Path to the latest_timestamps.pickle file.",
    )

    return parser


def get_bufr(
        bufr_out,
        l3_filepath,
        positions_filepath,
        timestamps_pickle_filepath,
        now_timestamp: datetime,
        stid_to_skip: Dict[str, List[str]],
        station_dimension_table: Mapping[str, Mapping[str, float]] = None,
        earliest_date: datetime = None,
        store_positions: bool = False,
        time_limit: str = "3M",
):
    if earliest_date is None:
        earliest_date = now_timestamp - timedelta(days=2)

    if station_dimension_table is None:
        station_dimension_table = load_station_dimension_table()

    # Get list of relative file paths
    fpaths = glob.glob(l3_filepath)

    # Make out dir
    outFiles = bufr_out
    if os.path.exists(outFiles) is False:
        os.mkdir(outFiles)

    # Read existing timestamps pickle to dictionary
    if os.path.isfile(timestamps_pickle_filepath):
        with open(timestamps_pickle_filepath, "rb") as handle:
            latest_timestamps = pickle.load(handle)
    else:
        logger.info("latest_timestamps.pickle not found!")
        latest_timestamps = {}

    # Initiate a new dict for current timestamps
    current_timestamps = {}

    if store_positions:
        # Initiate a dict to store station positions
        # (seeded with initial positions from wmo_config.positions_seed)
        # Used to retrieve a static set of positions to register stations with DMI/WMO
        # Also used to write AWS_latest_locations.csv to aws-L3 repo
        positions = wmo_config.positions_seed
    else:
        positions = None

    # Define stations to skip
    to_skip = []
    for k, v in stid_to_skip.items():
        to_skip.extend(v)
    to_skip = set(to_skip)  # Get rid of any duplicates

    # Setup diagnostic lists (logger.info at end)
    skipped = []
    no_recent_data = []
    no_entry_latest_timestamps = []
    failed_min_data_wx = []
    failed_min_data_pos = []

    # Iterate through csv files
    for file_path in fpaths:
        last_index = file_path.rfind("_")
        first_index = file_path.rfind("/")
        stid = file_path[first_index + 1: last_index]
        # stid = f.split('/')[-1].split('.csv')[0][:-5]
        logger.info("####### Processing {} #######".format(stid))
        if ("Roof" in file_path) or (stid in to_skip):
            logger.info(f"----> Skipping {stid} as per stid_to_skip config")
            skipped.append(stid)
            continue

        # TODO: The station dimension related meta data should be fetched from a station spefic configuration file or
        #  from header data from a NetCDF data source.
        if stid not in station_dimension_table:
            logger.info(f"Station id {stid} not in dimensions table")
            skipped.append(stid)
            continue
        station_dimensions = station_dimension_table[stid]

        bufrname = stid + ".bufr"
        logger.info(f"Generating {bufrname} from {file_path}")

        # Read csv file
        df1: pd.DataFrame = (
            pd.read_csv(file_path, delimiter=",")
            .assign(time=lambda df: pd.to_datetime(df["time"]))
            .set_index("time", drop=False)
            .sort_index()
            .assign(time=lambda df: df.index)
        )
        df1 = df1[:now_timestamp]

        s1_current = get_latest_data(
            df1,
            stid,
            earliest_date=earliest_date,
            lin_reg_time_limit=time_limit,
            positions=positions,
        )
        if s1_current is None:
            no_recent_data.append(stid)
            continue
        s1_current = filter_skipped_variables(s1_current, stid=s1_current["stid"])

        current_timestamp = s1_current.name
        current_timestamps[stid] = current_timestamp
        if store_positions:
            position = dict(
                timestamp=s1_current.name,
                lat=s1_current.get("gps_lat_fit"),
                lon=s1_current.get("gps_lon_fit"),
                alt=s1_current.get("gps_alt_fit"),
            )
            positions[stid] = position

        if stid in latest_timestamps and current_timestamp <= latest_timestamps[stid]:
            logger.info("Current data is not newer than latest")
            continue

        # Construct and export BUFR file
        bufr_variables = get_bufr_variables(
            s1_current=s1_current,
            station_type=station_dimensions["station_type"],
            wmo_id=station_dimensions["wmo_id"],
            barometer_height_relative_to_gps=station_dimensions["barometer_from_gps"],
            anemometer_height_relative_to_sonic_ranger=station_dimensions["anemometer_from_sonic_ranger"],
            temp_rh_height_relative_to_sonic_ranger=station_dimensions["temperature_from_sonic_ranger"],
            height_of_gps_from_station_ground=station_dimensions['height_of_gps_from_station_ground'],
        )
        with Path(outFiles, bufrname).open('bw') as fp:
            write_bufr_message(variables=bufr_variables, file=fp)

    # Write the most recent timestamps back to the pickle on disk
    logger.info("writing latest_timestamps.pickle")
    with open(timestamps_pickle_filepath, "wb") as handle:
        pickle.dump(current_timestamps, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if store_positions:
        positions_df = pd.DataFrame.from_dict(
            positions,
            orient="index",
            # columns=['timestamp','lat','lon','alt','lat_source','lon_source']
            columns=["timestamp", "lat", "lon", "alt"],
        )
        positions_df.sort_index(inplace=True)
        positions_df.to_csv(positions_filepath, index_label="stid")

    logger.info("--------------------------------")
    not_processed_wx_pos = set(failed_min_data_wx + failed_min_data_pos)
    not_processed_count = (
            len(skipped)
            + len(no_recent_data)
            + len(no_entry_latest_timestamps)
            + len(not_processed_wx_pos)
    )
    logger.info(
        "BUFR exported for {} of {} fpaths.".format(
            (len(fpaths) - not_processed_count), len(fpaths)
        )
    )
    logger.info("")
    logger.info("skipped: {}".format(skipped))
    logger.info("no_recent_data: {}".format(no_recent_data))
    logger.info("no_entry_latest_timestamps: {}".format(no_entry_latest_timestamps))
    logger.info("failed_min_data_wx: {}".format(failed_min_data_wx))
    logger.info("failed_min_data_pos: {}".format(failed_min_data_pos))
    logger.info("--------------------------------")


def filter_skipped_variables(row: pd.Series, stid: str) -> pd.Series:
    stid_vars_to_skip = wmo_config.vars_to_skip.get(stid, [])
    for var_key in row.keys():
        if var_key in stid_vars_to_skip:
            row[var_key] = np.nan
            logger.info("----> Skipping var: {} {}".format(stid, var_key))
    return row


def get_bufr_variables(
        s1_current: pd.Series,
        station_type: str,
        wmo_id: str,
        barometer_height_relative_to_gps: float,
        height_of_gps_from_station_ground: float,
        temp_rh_height_relative_to_sonic_ranger: float,
        anemometer_height_relative_to_sonic_ranger: float,
) -> BUFRVariables:
    output_row = BUFRVariables(
        wmo_id=wmo_id,
        station_type=station_type,
        timestamp = s1_current.name,
        # DMI wants non-corrected rh
        relativeHumidity = s1_current.rh_i,
        # Convert air temp, C to Kelvin
        airTemperature = s1_current.t_i + 273.15,
        # Convert pressure, correct the -1000 offset, then hPa to Pa
        # note that instantaneous pressure has 0.1 hPa precision
        pressure = (s1_current.p_i + 1000.0) * 100.0,
        windDirection = s1_current.wdir_i,
        windSpeed = s1_current.wspd_i,
        latitude = s1_current.gps_lat_fit,
        longitude = s1_current.gps_lon_fit,
        # TODO: This might need to be relative to snow height instead.
        heightOfStationGroundAboveMeanSeaLevel = (
                s1_current["gps_alt_fit"] - height_of_gps_from_station_ground
        ),
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH = (
                s1_current["z_boom_u_smooth"] + temp_rh_height_relative_to_sonic_ranger
        ),
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD = (
                s1_current["z_boom_u_smooth"] + anemometer_height_relative_to_sonic_ranger
        ),
        heightOfBarometerAboveMeanSeaLevel = (
                s1_current["gps_alt_fit"] + barometer_height_relative_to_gps
        ),

    )
    return output_row


def load_station_dimension_table() -> Mapping[str, Mapping[str, Any]]:
    station_dimension_table_path = Path(__file__).parent.joinpath("station_dimensions.csv")
    station_dimension_df = pd.read_csv(
        station_dimension_table_path, index_col=0,
        dtype={'wmo_id': str},
    )

    return station_dimension_df.to_dict('index')


if __name__ == "__main__":
    args = parse_arguments_bufr().parse_args()

    get_bufr(
        bufr_out=args.bufr_out,
        l3_filepath=args.l3_filepath,
        store_positions=args.store_positions,
        positions_filepath=args.positions_filepath,
        time_limit=args.time_limit,
        timestamps_pickle_filepath=args.timestamps_pickle_filepath,
        now_timestamp=datetime.utcnow(),
        stid_to_skip=wmo_config.stid_to_skip,
        station_dimension_table=load_station_dimension_table(),
    )
