"""
Command-line script for running BUFR file generation

Post-processing functions for AWS station data, such as converting PROMICE and GC-Net data files to WMO-compliant BUFR files

"""
__all__ = [
    "get_bufr",
    "main",
    "DEFAULT_POSITION_SEED_PATH",
    "DEFAULT_LIN_REG_TIME_LIMIT",
]

import argparse
import glob
import logging
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Mapping

import numpy as np
import pandas as pd

from pypromice.postprocess.bufr_utilities import write_bufr_message, BUFRVariables
from pypromice.postprocess.real_time_utilities import get_latest_data


from pypromice.station_configuration import (
    StationConfiguration,
    load_station_configuration_mapping,
)

logger = logging.getLogger(__name__)

DEFAULT_POSITION_SEED_PATH = Path(__file__).parent.joinpath("positions_seed.csv")
DEFAULT_LIN_REG_TIME_LIMIT = "91d"
REQUIRED_KEYS = (
    "t_i",
    "p_i",
    "rh_i",
    "wdir_i",
    "wspd_i",
    "gps_lat_fit",
    "gps_lon_fit",
    "gps_alt_fit",
    "z_boom_u_smooth",
)


def load_data(file_path: Path, latest_timestamp: datetime) -> pd.DataFrame:
    """
    Read AWS data from csv file using time as index and filter all rows after latest_timestamp

    Parameters
    ----------
    file_path
    latest_timestamp

    Returns
    -------
    Dataframe with all columns from csv file and time as index
    """
    # Read csv file
    df: pd.DataFrame = (
        pd.read_csv(file_path, delimiter=",", parse_dates=["time"])
        .set_index("time")
        .sort_index()
    )
    df = df[:latest_timestamp]
    return df


def get_bufr(
    bufr_out: Path,
    input_files: Sequence[Path],
    positions_filepath: Optional[Path],
    timestamps_pickle_filepath: Optional[Path],
    station_configuration_mapping: Mapping[str, StationConfiguration],
    target_timestamp: Optional[datetime] = None,
    positions_seed_path: Optional[Path] = None,
    time_window_length: timedelta = timedelta(days=2),
    store_positions: bool = False,
    linear_regression_time_limit: str = "91d",
    break_on_error: bool = False,
):
    """
    Main function for generating BUFR files and determine latest positions from a sequence of csv files

    The file timestamps_pickle_filepath is used to maintain a local state in the execution environment to ensure the
    same data is not processed multiple times.


    Parameters
    ----------
    bufr_out
        Path to the BUFR out directory.
    input_files
        List of csv file paths.
    positions_filepath
        Path to write latest positions. Used to retrieve a static set of positions to register stations with DMI/WMO
    timestamps_pickle_filepath
        Path to pickle file used for storing latest timestamp
    station_configuration_mapping
        Mapping of station id to StationConfiguration object
    target_timestamp
        get_bufr will export the latest data before target_timestamp. Default datetime.utcnow()
    positions_seed_path
        Path to csv file with position data used as default values for the output position.
    time_window_length
        The length of the time window to consider for the latest data. Default 2 days
    store_positions
        Flag determine if latest positions are exported.
    linear_regression_time_limit
        Previous time to limit dataframe before applying linear regression.
    break_on_error
        If True, the function will raise an exception if an error occurs during processing.

    """
    if target_timestamp is None:
        target_timestamp = datetime.utcnow()

    # Prepare (latest) positions
    positions = dict()
    if positions_seed_path:
        positions_seed = pd.read_csv(
            positions_seed_path,
            index_col="stid",
            delimiter=",",
            parse_dates=["timestamp"],
        ).to_dict(orient="index")
        logger.info(f"Seed positions for {positions_seed.keys()}")
        positions.update(positions_seed)

    # Prepare bufr output dir
    bufr_out.mkdir(parents=True, exist_ok=True)

    # Read existing timestamps pickle to dictionary
    if timestamps_pickle_filepath and timestamps_pickle_filepath.exists():
        with timestamps_pickle_filepath.open("rb") as handle:
            latest_timestamps = pickle.load(handle)
    else:
        logger.info("latest_timestamps.pickle not found!")
        latest_timestamps = {}

    # Setup diagnostic lists (logger.info at end)
    skipped = []
    no_recent_data = []

    # Iterate through csv files
    for file_path in input_files:
        # TODO: This split is explicitly requiring the filename to have sampleate at suffix. This shuld be more robust
        stid = file_path.stem.rsplit("_", 1)[0]
        logger.info("####### Processing {} #######".format(stid))

        if stid not in station_configuration_mapping:
            logger.info(f"Station id {stid} not in configuration mapping.")
            station_configuration = StationConfiguration(stid=stid)
            skipped.append(stid)
        else:
            station_configuration = station_configuration_mapping[stid]

        output_path = bufr_out / f"{stid}.bufr"
        logger.info(f"Generating {output_path} from {file_path}")

        time_window_start = target_timestamp - time_window_length
        # Use only newer data than the latest timestamp
        if stid in latest_timestamps:
            time_window_start = max(latest_timestamps[stid], time_window_start)

        try:
            input_data = load_data(file_path, target_timestamp)

            # Select current data
            latest_data = get_latest_data(
                input_data,
                lin_reg_time_limit=linear_regression_time_limit,
                vars_to_skip=station_configuration.skipped_variables,
            )
            if latest_data is None:
                logger.info("No valid instantaneous timestamps!")
                skipped.append(stid)
                continue

            # Create station positions
            station_position = get_station_positions(latest_data)
            if stid not in positions:
                positions[stid] = dict()
            if station_configuration.positions_update_timestamp_only:
                positions[stid]["timestamp"] = station_position["timestamp"]
            else:
                positions[stid].update(station_position)

            # Create BUFR File
            if (
                station_configuration.export_bufr
                and latest_data.name > time_window_start
            ):
                latest_timestamps[stid] = latest_data.name
                bufr_variables = get_bufr_variables(latest_data, station_configuration)
                if bufr_variables:
                    with output_path.open("bw") as output_file:
                        write_bufr_message(bufr_variables, output_file)
            else:
                logger.info(f"No new data {latest_data.name} <= {time_window_start}")
                no_recent_data.append(stid)

        except Exception:
            logger.exception(f"Failed processing {stid}")
            if output_path.exists():
                output_path.unlink()
            if break_on_error:
                raise
            skipped.append(stid)
            continue

    # Write the most recent timestamps back to the pickle on disk
    logger.info(f"writing latest_timestamps to {timestamps_pickle_filepath}")
    if timestamps_pickle_filepath:
        with timestamps_pickle_filepath.open("wb") as handle:
            pickle.dump(latest_timestamps, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    not_processed_count = len(skipped) + len(no_recent_data)
    logger.info(
        "BUFR exported for {} of {} fpaths.".format(
            (len(input_files) - not_processed_count), len(input_files)
        )
    )
    logger.info("")
    logger.info("skipped: {}".format(skipped))
    logger.info("no_recent_data: {}".format(no_recent_data))
    logger.info("--------------------------------")


def get_station_positions(latest_data: pd.Series) -> Dict:
    station_position = dict()
    station_position["timestamp"] = latest_data.name
    station_position["lat"] = latest_data["gps_lat_fit"]
    station_position["lon"] = latest_data["gps_lon_fit"]
    station_position["alt"] = latest_data["gps_alt_fit"]
    if any(
        [
            pd.isna(station_position["lat"]),
            pd.isna(station_position["lon"]),
            pd.isna(station_position["alt"]),
        ]
    ):
        logger.warning("Insufficient position data")
        station_position["lat"] = None
        station_position["lon"] = None
        station_position["alt"] = None
    return station_position


def get_bufr_variables(
    data: pd.Series,
    station_configuration: StationConfiguration,
) -> Optional[BUFRVariables]:
    """
    Helper function for converting our variables to the variables needed for bufr export.

    Raises AttributeError if station_configuration don't have the minimum dimension fields since they are required to determine barometer heights.
    * height_of_gps_from_station_ground
    * barometer_from_gps



    Parameters
    ----------
    data
        Series with processed variables from get_latest_datas

    station_configuration

    Returns
    -------
    BUFRVariables used by bufr_utilities

    """

    if not all(key in data.index for key in REQUIRED_KEYS):
        raise ValueError(
            f"Failed to process BUFRVariables. Missing required keys: {REQUIRED_KEYS}"
        )

    # Check that we have minimum required fields to proceed with writing to BUFR
    # Always require minimum a valid air temp or a valid pressure.
    # If both air temp and pressure are nan, do not submit.
    # This will allow the case of having only one or the other.
    if data[["t_i", "p_i"]].isna().all():
        logger.warning("Failed to process BUFRVariables - insufficient data")
        return None

    # Always require a valid position data
    if data[["gps_lat_fit", "gps_lon_fit", "gps_alt_fit"]].isna().any():
        logger.warning("Failed to process BUFRVariables - insufficient position data")
        return None

    if station_configuration.height_of_gps_from_station_ground is None:
        raise AttributeError(
            "height_of_gps_from_station_ground is required for BUFR export"
        )
    if station_configuration.barometer_from_gps is None:
        raise AttributeError("barometer_from_gps is required for BUFR export")

    if station_configuration.static_height_of_gps_from_mean_sea_level is None:
        height_of_gps_above_mean_sea_level = data["gps_alt_fit"]
    else:
        height_of_gps_above_mean_sea_level = (
            station_configuration.static_height_of_gps_from_mean_sea_level
        )

    heightOfStationGroundAboveMeanSeaLevel = (
        height_of_gps_above_mean_sea_level
        - station_configuration.height_of_gps_from_station_ground
    )

    heightOfBarometerAboveMeanSeaLevel = (
        height_of_gps_above_mean_sea_level + station_configuration.barometer_from_gps
    )

    if station_configuration.temperature_from_sonic_ranger is None:
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH = np.nan
    else:
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH = (
            data["z_boom_u_smooth"]
            + station_configuration.temperature_from_sonic_ranger
        )

    if station_configuration.anemometer_from_sonic_ranger is None:
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD = np.nan
    else:
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD = (
            data["z_boom_u_smooth"] + station_configuration.anemometer_from_sonic_ranger
        )

    output_row = BUFRVariables(
        wmo_id=station_configuration.wmo_id,
        station_type=station_configuration.station_type,
        timestamp=data.name,
        # DMI wants non-corrected rh
        relativeHumidity=data.rh_i,
        # Convert air temp, C to Kelvin
        airTemperature=data.t_i + 273.15,
        # Convert pressure, correct the -1000 offset, then hPa to Pa
        # note that instantaneous pressure has 0.1 hPa precision
        nonCoordinatePressure=(data.p_i + 1000.0) * 100.0,
        windDirection=data.wdir_i,
        windSpeed=data.wspd_i,
        latitude=data.gps_lat_fit,
        longitude=data.gps_lon_fit,
        # TODO: This might need to be relative to snow height instead.
        heightOfStationGroundAboveMeanSeaLevel=heightOfStationGroundAboveMeanSeaLevel,
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH,
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD,
        heightOfBarometerAboveMeanSeaLevel=heightOfBarometerAboveMeanSeaLevel,
    )
    return output_row


def main():
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
        "-p",
        type=Path,
        required=False,
        help="Path to write AWS_latest_locations.csv file.",
    )
    parser.add_argument(
        "--linear_regression_time_limit",
        "--time-limit",
        default=DEFAULT_LIN_REG_TIME_LIMIT,
        type=str,
        required=False,
        help="Previous time to limit dataframe before applying linear regression.",
    )
    parser.add_argument(
        "--input_files",
        "-i",
        type=Path,
        nargs="+",
        required=True,
        help="Path to input files .csv files. Can be direct paths or glob patterns",
    )
    parser.add_argument(
        "--bufr-out",
        "-o",
        type=Path,
        required=True,
        help="Path to the BUFR out directory.",
    )
    parser.add_argument(
        "--timestamps-pickle-filepath",
        type=Path,
        required=False,
        help="Path to the latest_timestamps.pickle file.",
    )
    parser.add_argument(
        "--station_configurations_root",
        type=Path,
        required=True,
        help="Path to root directory containing station configuration toml files",
    )
    parser.add_argument(
        "--position_seed",
        default=DEFAULT_POSITION_SEED_PATH,
        type=Path,
        required=False,
        help="Path to csv file with seed values for output positions.",
    )
    parser.add_argument(
        "--target_timestamp",
        "--now-timestamp",
        default=datetime.utcnow(),
        type=pd.Timestamp,
        help="Timestamp used to determine latest data. Default utcnow.",
    )
    parser.add_argument("--verbose", "-v", default=False, action="store_true")

    args = parser.parse_args()

    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=log_level,
    )

    # Interpret all input file paths as glob patterns if they don't exist
    input_files: List[Path] = list()
    for path in args.input_files:
        if path.exists():
            input_files.append(path)
        else:
            # The input path might be a glob pattern
            input_files += map(Path, glob.glob(path.as_posix()))

    station_configuration_mapping = load_station_configuration_mapping(
        args.station_configurations_root,
        skip_unexpected_fields=True,
    )

    get_bufr(
        bufr_out=args.bufr_out,
        input_files=input_files,
        store_positions=args.store_positions,
        positions_filepath=args.positions_filepath,
        linear_regression_time_limit=args.linear_regression_time_limit,
        timestamps_pickle_filepath=args.timestamps_pickle_filepath,
        target_timestamp=args.target_timestamp,
        station_configuration_mapping=station_configuration_mapping,
        positions_seed_path=args.position_seed,
    )


if __name__ == "__main__":
    main()
