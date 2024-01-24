#!/usr/bin/env python

"""
Command-line script for running BUFR file generation
Created: Dec 20, 2022
Author: Patrick Wright, GEUS
"""
from typing import List, Dict

import logging

import pandas as pd
import glob, os
import argparse
from datetime import datetime, timedelta
import pickle

from pypromice.postprocess.wmo_config import (
    ibufr_settings,
    stid_to_skip,
    positions_seed,
    positions_update_timestamp_only,
)
from pypromice.postprocess.csv2bufr import (
    getBUFR,
    linear_fit,
    rolling_window,
    round_values,
    find_positions,
    min_data_check,
)
from pypromice.postprocess import wmo_config
from pypromice.postprocess.wmo_config import ibufr_settings, positions_seed, positions_update_timestamp_only
from pypromice.postprocess.csv2bufr import getBUFR, linear_fit, rolling_window, round_values, \
										   find_positions, min_data_check

logger = logging.getLogger(__name__)


def parse_arguments_bufr() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dev",
        action="store_true",
        required=False,
        default=True,
        help="If included (True), run in dev mode. Useful for repeated runs of script between transmissions.",
    )

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
        dev: bool = False,
        store_positions: bool = False,
        time_limit: str = "3M",
    ):
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
        positions = positions_seed

    # Define stations to skip
    to_skip = []
    for k, v in stid_to_skip.items():
        to_skip.extend(v)
    to_skip = set(to_skip)  # Get rid of any duplicates

    # Setup diagnostic lists (logger.info at end)
    skipped = []
    no_recent_data = []
    no_valid_data = []
    no_entry_latest_timestamps = []
    failed_min_data_wx = []
    failed_min_data_pos = []

    land_stids = ibufr_settings["land"]["station"]["stationNumber"].keys()

    # Iterate through csv files
    for file_path in fpaths:
        last_index = file_path.rfind("_")
        first_index = file_path.rfind("/")
        stid = file_path[first_index + 1 : last_index]
        # stid = f.split('/')[-1].split('.csv')[0][:-5]
        logger.info("####### Processing {} #######".format(stid))
        if ("Roof" not in file_path) and (stid not in to_skip):
            bufrname = stid + ".bufr"
            logger.info(f"Generating {bufrname} from {file_path}")

            # TODO: Consider the store positions
            if store_positions and (stid not in positions_update_timestamp_only):
                positions[stid] = {}
                # Optionally include source flag columns, useful to indicate if position
                # comes from current transmission, or older data. This could also be used
                # to differentiate GPS from modem, but using the combine_first method for
                # the modem positions currently prevents us from easily knowing which source
                # was used.
                # positions[stid]['lat_source'] = ''
                # positions[stid]['lon_source'] = ''

            # Read csv file
            df1 = (
                pd.read_csv(file_path, delimiter=",")
                .assign(time=lambda df: pd.to_datetime(df["time"]))
                .set_index("time")
                .sort_index()
                .assign(time=lambda df: df.index)
            )
            # df1.set_index(pd.to_datetime(df1['time']), inplace=True)
            # df1.sort_index(inplace=True) # make sure we are time-sorted

            # Check that the last valid index for all instantaneous values match
            # Note: we cannot always use the single most-recent timestamp in the dataframe
            # e.g. for 6-hr transmissions, *_u will have hourly data while *_i is nan
            # Need to check for last valid (non-nan) index instead
            last_valid_indices = {
                "t_i": df1["t_i"].last_valid_index(),
                "p_i": df1["p_i"].last_valid_index(),
                "rh_i": df1["rh_i"].last_valid_index(),
                "wspd_i": df1["wspd_i"].last_valid_index(),
                "wdir_i": df1["wdir_i"].last_valid_index(),
            }

            two_days_ago = now_timestamp - timedelta(days=2)

            if len(set(last_valid_indices.values())) != 1:
                # instantaneous vars have different timestamps
                recent = {}
                for k, v in last_valid_indices.items():
                    if (v is not None) and (v >= two_days_ago):
                        recent[k] = v
                if len(recent) == 0:
                    logger.info("No recent instantaneous timestamps!")
                    no_recent_data.append(stid)
                    if store_positions:
                        df1_limited, positions = find_positions(df1, stid, time_limit, positions=positions)
                    continue
                else:
                    # we have partial data, just use the most recent row
                    current_timestamp = max(recent.values())
                    # We will throw this obset down the line, and there is a final min_data_check
                    # to make sure we have minimum data requirements before writing to BUFR
            else:
                if all(i is None for i in last_valid_indices.values()) is True:
                    logger.info("All instantaneous timestamps are None!")
                    no_valid_data.append(stid)
                    if store_positions:
                        df1_limited, positions = find_positions(
                            df1, stid, time_limit, positions=positions
                        )
                    continue
                else:
                    # all values are present, with matching timestamps, so just use t_i
                    current_timestamp = df1["t_i"].last_valid_index()

            logger.info(f"TIMESTAMP: {current_timestamp}")

            # set in dict, will be written to disk at end
            current_timestamps[stid] = current_timestamp

            if stid in latest_timestamps:
                latest_timestamp = latest_timestamps[stid]

                if dev is True:
                    logger.info("----> Running in dev mode!")
                    # If we want to run repeatedly (before another transmission comes in), then don't
                    # check the actual latest timestamp, and just set to two_days_ago
                    latest_timestamp = two_days_ago

                if (current_timestamp > latest_timestamp) and (
                    current_timestamp > two_days_ago
                ):
                    logger.info("Time checks passed.")

                    if store_positions:
                        # return positions dict for writing to csv file after processing finished
                        df1_limited, positions = find_positions(
                            df1, stid, time_limit, current_timestamp, positions
                        )
                    else:
                        # we only need to add positions to the BUFR file
                        df1_limited, _ = find_positions(
                            df1, stid, time_limit, current_timestamp
                        )

                    # Apply smoothing to z_boom_u
                    # require at least 2 hourly obs? Sometimes seeing once/day data for z_boom_u
                    df1_limited = rolling_window(df1_limited, "z_boom_u", "72H", 2, 1)

                    # limit to single most recent valid row (convert to series)
                    s1_current = df1_limited.loc[current_timestamp]

                    # Convert air temp, C to Kelvin
                    s1_current.t_i = s1_current.t_i + 273.15

                    # Convert pressure, correct the -1000 offset, then hPa to Pa
                    # note that instantaneous pressure has 0.1 hPa precision
                    s1_current.p_i = (s1_current.p_i + 1000.0) * 100.0

                    s1_current = round_values(s1_current)

                    # Check that we have minimum required valid data
                    min_data_wx_result, min_data_pos_result = min_data_check(
                        s1_current, stid
                    )
                    if min_data_wx_result is False:
                        failed_min_data_wx.append(stid)
                        continue
                    elif min_data_pos_result is False:
                        failed_min_data_pos.append(stid)
                        continue

                    # Construct and export BUFR file
                    outBUFR_path = os.path.join(outFiles, bufrname)
                    file_removed = getBUFR(s1_current, outBUFR_path, stid, land_stids)

                    if file_removed is False:
                        logger.info(
                            f"Successfully exported bufr file to {outFiles+bufrname}"
                        )
                else:
                    logger.info("----> Time checks failed for {}".format(stid))
                    logger.info(f"      current: {current_timestamp}")
                    if dev is True:
                        logger.info(f" latest (DEV): {latest_timestamp}")
                    else:
                        logger.info(f"       latest: {latest_timestamp}")
                    no_recent_data.append(stid)
                    if store_positions:
                        current_timestamp = None
                        df1_limited, positions = find_positions(
                            df1, stid, time_limit, current_timestamp, positions
                        )
            else:
                logger.info("{} not found in latest_timestamps".format(stid))
                no_entry_latest_timestamps.append(stid)
        else:
            logger.info("----> Skipping {} as per stid_to_skip config".format(stid))
            skipped.append(stid)
            if store_positions and stid not in ("XXX",):
                # still will be useful to have all stations in AWS_station_location.csv,
                # regardless if they were skipped for the DMI upload
                if stid not in positions_update_timestamp_only:
                    positions[stid] = {}
                df_skipped = pd.read_csv(file_path, delimiter=",")
                df_skipped.set_index(pd.to_datetime(df_skipped["time"]), inplace=True)
                df_skipped.sort_index(inplace=True)  # make sure we are time-sorted
                df_skipped_limited, positions = find_positions(
                    df_skipped, stid, time_limit, positions=positions
                )

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
        + len(no_valid_data)
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
    logger.info("no_valid_data: {}".format(no_valid_data))
    logger.info("no_entry_latest_timestamps: {}".format(no_entry_latest_timestamps))
    logger.info("failed_min_data_wx: {}".format(failed_min_data_wx))
    logger.info("failed_min_data_pos: {}".format(failed_min_data_pos))
    logger.info("--------------------------------")


if __name__ == "__main__":
    args = parse_arguments_bufr().parse_args()
    get_bufr(
        bufr_out=args.bufr_out,
        dev=args.dev,
        l3_filepath=args.l3_filepath,
        store_positions=args.store_positions,
        positions_filepath=args.positions_filepath,
        time_limit=args.time_limit,
        timestamps_pickle_filepath=args.timestamps_pickle_filepath,
        now_timestamp=datetime.utcnow(),
        stid_to_skip=wmo_config.stid_to_skip,
    )
