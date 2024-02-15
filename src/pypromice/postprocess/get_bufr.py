#!/usr/bin/env python

"""
Command-line script for running BUFR file generation

Post-processing functions for AWS station data, such as converting PROMICE and GC-Net data files to WMO-compliant BUFR files

"""
import argparse
import glob
import logging
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Mapping, Optional, Collection, Sequence, Union, TextIO

import attrs
import numpy as np
import pandas as pd
import toml

from pypromice.postprocess.bufr_utilities import write_bufr_message, BUFRVariables
from pypromice.postprocess.real_time_utilities import get_latest_data

logger = logging.getLogger(__name__)

DEFAULT_STATION_CONFIGURATION_PATH = Path(__file__).parent.joinpath(
    "station_configurations.toml"
)
DEFAULT_POSITION_SEED_PATH = Path(__file__).parent.joinpath("positions_seed.csv")


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
        type=Path,
        required=False,
        help="Path to write AWS_latest_locations.csv file.",
    )

    parser.add_argument(
        "--time-limit",
        default="91d",
        type=str,
        required=False,
        help="Previous time to limit dataframe before applying linear regression.",
    )

    parser.add_argument(
        "--input_files",
        "--l3-filepath",
        type=Path,
        nargs="+",
        help="Path to L3 tx .csv files. Can be direct paths or glob patterns",
    )

    parser.add_argument(
        "--bufr-out",
        default="src/pypromice/postprocess/BUFR_out/",
        type=Path,
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

    parser.add_argument(
        "--station_configuration_mapping",
        default=DEFAULT_STATION_CONFIGURATION_PATH,
        type=Path,
        required=False,
        help="Path to csv file with station meta data and BUFR export configuration",
    )

    parser.add_argument(
        "--position_seed",
        default=DEFAULT_POSITION_SEED_PATH,
        type=Path,
        required=False,
        help="Path to csv file with seed values for output positions.",
    )

    parser.add_argument("--verbose", "-v", default=False, action="store_true")

    return parser


@attrs.define
class StationConfiguration:
    """
    Helper class for storing station specific configurations with respect to

    * Installation specific distance measurements such as height differences between instruments
    * Reference strings such as stid, station_site and wmo_id
    * BUFR export specific parameters

    # TODO: The station related meta data should be fetched from a station specific configuration files in the future or
    # from header data in data source.
    """

    stid: str
    station_site: str = None
    project: Optional[str] = None
    station_type: Optional[str] = None
    wmo_id: Optional[str] = None
    barometer_from_gps: Optional[float] = None
    anemometer_from_sonic_ranger: Optional[float] = None
    temperature_from_sonic_ranger: Optional[float] = None
    height_of_gps_from_station_ground: Optional[float] = None
    sonic_ranger_from_gps: Optional[float] = None

    # The station data will be exported to BUFR if True. Otherwise, it will only export latest position
    export_bufr: bool = False
    comment: Optional[str] = None

    # skip specific variables for stations
    # If a variable has known bad data, use this collection to skip the variable
    # Note that if a station is not reporting both air temp and pressure it will be skipped,
    # as currently implemented in csv2bufr.min_data_check().
    # ['p_i'], # EXAMPLE
    skipped_variables: List[str] = attrs.field(factory=list)

    positions_update_timestamp_only: bool = False

    def as_dict(self) -> Dict:
        return attrs.asdict(self)


def load_station_configuration_mapping(
    fp: Union[str, Path, TextIO]
) -> Mapping[str, StationConfiguration]:
    """
    Read station configurations from toml file

    Parameters
    ----------
    fp :
        Path to or open toml file

    Returns
    -------
    Mapping from stid to StationConfiguration

    """
    return {
        stid: StationConfiguration(**config_dict)
        for stid, config_dict in toml.load(fp).items()
    }


def write_station_configuration_mapping(
    config_mapping: Mapping[str, StationConfiguration], fp: TextIO
):
    """
    Write station configuration to toml file

    Parameters
    ----------
    config_mapping
        Mapping from stid to StationConfiguration
    fp
        open writable TextIO
    """
    config_mapping = {
        config.stid: config.as_dict() for config in config_mapping.values()
    }
    toml.dump(config_mapping, fp)


def process_station(
    file_path: Path,
    output_path: Path,
    now_timestamp: datetime,
    latest_timestamp: Optional[datetime],
    time_limit: str,
    stid: str,
    station_configuration: StationConfiguration,
) -> Optional[Dict]:
    df = load_data(file_path, now_timestamp)

    # Select current data
    latest_data = get_latest_data(
        df,
        lin_reg_time_limit=time_limit,
    )

    if latest_data is None:
        logger.info("No valid instantaneous timestamps!")
        return None

    latest_data = filter_skipped_variables(
        latest_data, vars_to_skip=station_configuration.skipped_variables
    )

    # Check that we have minimum required valid data
    sufficient_wx_data, sufficient_position_data = min_data_check(latest_data)

    station_position = dict()
    station_position["timestamp"] = latest_data.name
    if sufficient_position_data:
        station_position["lon"] = latest_data.get("gps_lon_fit")
        station_position["lat"] = latest_data.get("gps_lat_fit")
        station_position["alt"] = latest_data.get("gps_alt_fit")
    else:
        logger.warning("Insufficient position data")
        # Don't use any position attributes from latest_data
        station_position["lon"] = None
        station_position["lat"] = None
        station_position["alt"] = None
        return station_position

    if station_configuration.export_bufr:
        if not sufficient_wx_data:
            logger.warning(f"Failed min data wx {stid}")
            return station_position

        # Store current timest
        if latest_data.name <= latest_timestamp:
            return station_position

        # Construct and export BUFR file
        bufr_variables = get_bufr_variables(
            data=latest_data,
            station_configuration=station_configuration,
        )
        with output_path.open("bw") as fp:
            write_bufr_message(variables=bufr_variables, file=fp)

    return station_position


def load_data(file_path: Path, now_timestamp: datetime) -> pd.DataFrame:
    """
    Read AWS data from csv file using time as index and filter all rows after now_timestamp

    Parameters
    ----------
    file_path
    now_timestamp

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
    df = df[:now_timestamp]
    return df


def get_bufr(
    bufr_out: Path,
    input_files: Sequence[Path],
    positions_filepath: Path,
    timestamps_pickle_filepath: Optional[Path],
    station_configuration_path: Optional[Path],
    now_timestamp: Optional[datetime] = None,
    positions_seed_path: Optional[Path] = None,
    earliest_timestamp: datetime = None,
    store_positions: bool = False,
    time_limit: str = "91d",
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
        List of L3 csv file paths.
    positions_filepath
        Path to write latest positions. Used to retrieve a static set of positions to register stations with DMI/WMO
    timestamps_pickle_filepath
        Path to pickle file used for storing latest timestamp
    station_configuration_path
        Path to toml file with configuration entries for each station
    now_timestamp
        get_bufr will export the latest data before now_timestamp. Default datetime.utcnow()
    positions_seed_path
        Path to csv file with position data used as default values for the output position.
    earliest_timestamp
        The earliest allowed timestamp for data to be included in the output. Default now_timestamp - 2 days
    store_positions
        Flag determine if latest positions are exported.
    time_limit
        Previous time to limit dataframe before applying linear regression.

    """
    if now_timestamp is None:
        now_timestamp = datetime.utcnow()

    if earliest_timestamp is None:
        earliest_timestamp = now_timestamp - timedelta(days=2)

    # Prepare (latest) positions
    positions = dict()
    if positions_seed_path:
        positions_seed = pd.read_csv(
            positions_seed_path, index_col=0, delimiter=",", parse_dates=["timestamp"]
        ).to_dict(orient="index")
        logger.info(f"Seed positions for {positions_seed.keys()}")
        positions.update(positions_seed)

    # Prepare station configurations
    if station_configuration_path is None:
        station_configuration_mapping = dict()
    else:
        station_configuration_mapping = load_station_configuration_mapping(
            station_configuration_path
        )

    # Prepare bufr output dir
    bufr_out.mkdir(parents=True, exist_ok=True)

    # Read existing timestamps pickle to dictionary
    if timestamps_pickle_filepath and timestamps_pickle_filepath.exists():
        with timestamps_pickle_filepath.open("rb") as handle:
            latest_timestamps = pickle.load(handle)
    else:
        logger.info("latest_timestamps.pickle not found!")
        latest_timestamps = {}

    # Initiate a new dict for current timestamps
    current_timestamps = {}

    # Setup diagnostic lists (logger.info at end)
    skipped = []
    no_recent_data = []
    no_entry_latest_timestamps = []
    failed_min_data_wx = []
    failed_min_data_pos = []

    # Iterate through csv files
    for file_path in input_files:
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
        latest_timestamp = latest_timestamps.get(stid, earliest_timestamp)
        latest_timestamp = max(earliest_timestamp, latest_timestamp)

        try:
            station_position = process_station(
                file_path=file_path,
                output_path=output_path,
                now_timestamp=now_timestamp,
                latest_timestamp=latest_timestamp,
                time_limit=time_limit,
                stid=stid,
                station_configuration=station_configuration,
            )
        except Exception:
            logger.exception(f"Failed processing {stid}")
            continue

        if station_position is None:
            logger.warning(f"No position information available for {stid}")

        else:
            if stid not in positions:
                positions[stid] = dict()

            if station_configuration.positions_update_timestamp_only:
                positions[stid]["timestamp"] = station_position["timestamp"]
            else:
                positions[stid].update(station_position)

    # Write the most recent timestamps back to the pickle on disk
    logger.info(f"writing latest_timestamps to {timestamps_pickle_filepath}")
    if timestamps_pickle_filepath:
        with timestamps_pickle_filepath.open("wb") as handle:
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
            (len(input_files) - not_processed_count), len(input_files)
        )
    )
    logger.info("")
    logger.info("skipped: {}".format(skipped))
    logger.info("no_recent_data: {}".format(no_recent_data))
    logger.info("no_entry_latest_timestamps: {}".format(no_entry_latest_timestamps))
    logger.info("failed_min_data_wx: {}".format(failed_min_data_wx))
    logger.info("failed_min_data_pos: {}".format(failed_min_data_pos))
    logger.info("--------------------------------")


def filter_skipped_variables(
    row: pd.Series, vars_to_skip: Collection[str]
) -> pd.Series:
    """
    Mutate input series by setting var_to_skip to np.nan

    Parameters
    ----------
    row
    vars_to_skip
        List of variable names to be skipped

    Returns
    -------
    Input series

    """
    vars_to_skip = set(row.keys()) & set(vars_to_skip)
    for var_key in vars_to_skip:
        row[var_key] = np.nan
        logger.info("----> Skipping var: {}".format(var_key))
    return row


def get_bufr_variables(
    data: pd.Series,
    station_configuration: StationConfiguration,
) -> BUFRVariables:
    """
    Helper function for converting our l3 variables to the variables needed for bufr export.

    Parameters
    ----------
    data
        Series with processed l3 variables from get_latest_datas

    station_configuration

    Returns
    -------
    BUFRVariables used by bufr_utilities

    """
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
        pressure=(data.p_i + 1000.0) * 100.0,
        windDirection=data.wdir_i,
        windSpeed=data.wspd_i,
        latitude=data.gps_lat_fit,
        longitude=data.gps_lon_fit,
        # TODO: This might need to be relative to snow height instead.
        heightOfStationGroundAboveMeanSeaLevel=(
            data["gps_alt_fit"]
            - station_configuration.height_of_gps_from_station_ground
        ),
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=(
            data["z_boom_u_smooth"]
            + station_configuration.temperature_from_sonic_ranger
        ),
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=(
            data["z_boom_u_smooth"] + station_configuration.anemometer_from_sonic_ranger
        ),
        heightOfBarometerAboveMeanSeaLevel=(
            data["gps_alt_fit"] + station_configuration.barometer_from_gps
        ),
    )
    return output_row


def min_data_check(s):
    """Check that we have minimum required fields to proceed with writing to BUFR
    For wx vars, we currently require both air temp and pressure to be non-NaN.
    If you know a specific var is reporting bad data, you can ignore just that var
    using the vars_to_skip dict in wmo_config.

    Parameters
    ----------
    s : pandas series
        The current obset we are working with (for BUFR submission)

    Returns
    -------
    min_data_wx_result : bool
        True (default), the test for min wx data passed. False, the test failed.
    min_data_pos_result : bool
        True (default), the test for min position data passed. False, the test failed.
    """
    min_data_wx_result = True
    min_data_pos_result = True

    # Can use pd.isna() or math.isnan() below...

    # Always require valid air temp and valid pressure (both must be non-nan)
    # if (pd.isna(s['t_i']) is False) and (pd.isna(s['p_i']) is False):
    #     pass
    # else:
    #     print('----> Failed min_data_check for air temp and pressure!')
    #     min_data_wx_result = False

    # If both air temp and pressure are nan, do not submit.
    # This will allow the case of having only one or the other.
    if (pd.isna(s["t_i"]) is True) and (pd.isna(s["p_i"]) is True):
        logger.warning("----> Failed min_data_check for air temp and pressure!")
        min_data_wx_result = False

    # Missing just elevation OK
    # if (pd.isna(s['gps_lat_fit']) is False) and (pd.isna(s['gps_lon_fit']) is False):
    #     pass
    # Require all three: lat, lon, elev
    if (
        (pd.isna(s["gps_lat_fit"]) is False)
        and (pd.isna(s["gps_lon_fit"]) is False)
        and (pd.isna(s["gps_alt_fit"]) is False)
    ):
        pass
    else:
        logger.warning("----> Failed min_data_check for position!")
        min_data_pos_result = False

    return min_data_wx_result, min_data_pos_result


if __name__ == "__main__":
    args = parse_arguments_bufr().parse_args()

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

    get_bufr(
        bufr_out=args.bufr_out,
        input_files=input_files,
        store_positions=args.store_positions,
        positions_filepath=args.positions_filepath,
        time_limit=args.time_limit,
        timestamps_pickle_filepath=args.timestamps_pickle_filepath,
        now_timestamp=datetime.utcnow(),
        station_configuration_path=args.station_configuration_mapping,
        positions_seed_path=args.position_seed,
    )
