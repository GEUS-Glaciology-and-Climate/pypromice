"""
Utility functions writing and reading BUFR files from AWS data

see documentation here:
https://confluence.ecmwf.int/display/ECC/Documentation

BUFR element table for WMO master table version 32
https://confluence.ecmwf.int/display/ECC/WMO%3D32+element+table

"""
import datetime
import logging
import math
from os import PathLike
from pathlib import Path
from typing import BinaryIO, Optional

import attrs
import numpy as np
import pandas as pd
from eccodes import (
    codes_set,
    codes_write,
    codes_release,
    codes_bufr_new_from_samples,
    CodesInternalError,
    codes_is_defined,
    codes_bufr_new_from_file,
    codes_get,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BUFRVariables",
    "write_bufr_message",
    "read_bufr_message",
    "read_bufr_file",
]


def round_converter(decimals: int):
    def round(value: float):
        return np.round(value, decimals=decimals)

    return round


# Enforce precision
# Note the sensor accuracies listed here:
# https://essd.copernicus.org/articles/13/3819/2021/#section8
# In addition to sensor accuracy, WMO requires pressure and heights
# to be reported at 0.1 precision.
@attrs.define
class BUFRVariables:
    wmo_id: str
    station_type: str
    timestamp: datetime.datetime
    relativeHumidity: float = attrs.field(converter=round_converter(0))
    airTemperature: float = attrs.field(converter=round_converter(1))
    pressure: float = attrs.field(converter=round_converter(1))
    windDirection: float = attrs.field(converter=round_converter(0))
    windSpeed: float = attrs.field(converter=round_converter(1))
    latitude: float = attrs.field(converter=round_converter(6))
    longitude: float = attrs.field(converter=round_converter(6))
    heightOfStationGroundAboveMeanSeaLevel: float = attrs.field(
        converter=round_converter(2)
    )
    heightOfBarometerAboveMeanSeaLevel: float = attrs.field(
        converter=round_converter(2)
    )
    heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH: float = attrs.field(
        converter=round_converter(4)
    )
    heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD: float = attrs.field(
        converter=round_converter(4)
    )


STATION_CONFIGURATIONS = {
    "mobile": {
        # 'blockNumber': 4, #4 is Greenland, 6 is Denmark; not valid with synopMobil template
        "regionNumber": 6,  # 6 is Europe, 7 is MISSING VALUE; not valid with synopLand template
        "centre": 94,  # 94 is Copenhagen
        # 'agencyInChargeOfOperatingObservingPlatform': , #nothing for DMI or GEUS in code table
        # 'wmoRegionSubArea': 1,
        # 'stationOrSiteName': , #not valid with synopMobil template
        # 'shortStationName': , #not valid with synopMobil template
        # 'longStationName': , #not valid with synopMobil template
        # 'directionOfMotionOfMovingObservingPlatform': ,
        # 'movingObservingPlatformSpeed': ,
        "stationType": 0,  # automatic station
        "instrumentationForWindMeasurement": 8,  # certified instruments
        "stationElevationQualityMarkForMobileStations": 1,  # Excellent - within 3m; not valid with synopLand template
    },
    "land": {
        "blockNumber": 4,  # 4 is Greenland, 6 is Denmark; not valid with synopMobil template
        # 'regionNumber': 6, #6 is Europe, 7 is MISSING VALUE; not valid with synopLand template
        "centre": 94,  # 94 is Copenhagen
        # 'agencyInChargeOfOperatingObservingPlatform': , #nothing for DMI or GEUS in code table
        # 'wmoRegionSubArea': 1,
        # 'stationOrSiteName': , #not valid with synopMobil template
        # 'shortStationName': , #not valid with synopMobil template
        # 'longStationName': , #not valid with synopMobil template
        "stationType": 0,  # automatic station
        "instrumentationForWindMeasurement": 8,  # certified instruments
        # 'stationElevationQualityMarkForMobileStations': 1, #Excellent - within 3m; not valid with synopLand template
    },
}

BUFR_TEMPLATES = {
    "mobile": {
        "unexpandedDescriptors": (307090),  # message template, "synopMobil"
        "edition": 4,  # latest edition
        "masterTableNumber": 0,
        "masterTablesVersionNumber": 32,  # DMI recommends any table version between 28-32
        "localTablesVersionNumber": 0,
        "bufrHeaderCentre": 94,  # originating centre 98=ECMWF, 94=DMI
        # 'bufrHeaderSubCentre': 0,
        "updateSequenceNumber": 0,  # 0 is original message, incremented by 1 for updates
        "dataCategory": 0,  # surface data - land
        "internationalDataSubCategory": 3,  # hourly synoptic observations from mobile-land stations (SYNOP MOBIL)
        # 'dataSubCategory': 0,
        "observedData": 1,
        "compressedData": 0,
    },
    "land": {
        "unexpandedDescriptors": (307080),  # message template, "synopLand"
        "edition": 4,  # latest edition
        "masterTableNumber": 0,
        "masterTablesVersionNumber": 32,  # DMI recommends any table version between 28-32
        "localTablesVersionNumber": 0,
        "bufrHeaderCentre": 94,  # originating centre 98=ECMWF, 94=DMI
        # 'bufrHeaderSubCentre': 0,
        "updateSequenceNumber": 0,  # 0 is original message, incremented by 1 for updates
        "dataCategory": 0,  # surface data - land
        "internationalDataSubCategory": 0,  # Hourly synoptic observations from fixed-land stations (SYNOP)
        # 'dataSubCategory': 0,
        "observedData": 1,
        "compressedData": 0,
    },
}


def write_bufr_message(
    variables: BUFRVariables,
    file: BinaryIO,
):
    """Construct and export .bufr message to file from pandas Series.

    Parameters
    ----------
    variables : pandas.Series
        Pandas series of single most recent obset for a station
    file
        Binary writable file object
    """

    # Create new bufr message to write to
    ibufr = codes_bufr_new_from_samples("BUFR4")

    try:
        # we must pass all the following functions without error.
        # If handled (or unhandled) errors occur, we re-raise and
        # the exceptions below will set remove_file to True.
        set_template(ibufr, variables.timestamp, variables.station_type)
        set_station(ibufr, variables.station_type, variables.wmo_id)
        set_AWS_variables(ibufr, variables)

        # Encode keys in data section
        codes_set(ibufr, "pack", 1)

        # Write bufr message to bufr file
        codes_write(ibufr, file)

    except CodesInternalError as ec:
        logger.exception(f"CodesInternalError in getBUFR", exc_info=ec)
        raise ec
    except Exception as e:
        logger.exception(f"ERROR in getBUFR", exc_info=e)
        raise e
    finally:
        codes_release(ibufr)


def set_template(ibufr, timestamp, station_type: str):
    """Set BUFR message template.

    Parameters
    ----------
    ibufr : bufr.msg
        Bufr message object
    timestamp : datetime.Datetime
        Timestamp of observation
    config_key : str
        Defines which config dict to use in wmo_config.ibufr_settings, 'mobile' or 'land'
    """
    template = BUFR_TEMPLATES[station_type]

    for k, v in template.items():
        if codes_is_defined(ibufr, k) == 1:
            codes_set(ibufr, k, v)
        else:
            logger.warning("-----> setTemplate Key not defined: {}".format(k))
            continue

    codes_set(ibufr, "typicalYear", timestamp.year)
    codes_set(ibufr, "typicalMonth", timestamp.month)
    codes_set(ibufr, "typicalDay", timestamp.day)
    codes_set(ibufr, "typicalHour", timestamp.hour)
    codes_set(ibufr, "typicalMinute", timestamp.minute)
    # codes_set(ibufr, 'typicalSecond', timestamp.second)


def set_station(ibufr, station_type: str, wmo_id: str):
    """Set station-specific info to bufr message.

    Parameters
    ----------
    ibufr : bufr.msg
        Bufr message object
    config_key : str
        Defines which config dict to use in wmo_config.ibufr_settings, 'mobile' or 'land'
    """
    if station_type == "mobile":
        station_config = dict(shipOrMobileLandStationIdentifier=wmo_id)
    elif station_type == "land":
        # StationNumber for land stations are integeres
        wmo_id_int = int(wmo_id)
        station_config = dict(stationNumber=wmo_id_int)
    else:
        raise Exception(f"Unsupported station station type {station_type}")
    station_config.update(STATION_CONFIGURATIONS[station_type])

    for key, value in station_config.items():
        codes_set(ibufr, key, value)


def set_AWS_variables(
    ibufr,
    variables: BUFRVariables,
    barometer_height_relative_to_gps: float = 0,
):
    """Set AWS measurements to bufr message.

    Parameters
    ----------
    ibufr s: bufr.msg
        Bufr message object
    variables
        Dict with AWS variable data
    timestamp : datetime.datetime
        timestamp for this row
    barometer_height_relative_to_gps : float
        Vertical distance from gps to barometer
    """
    # Set timestamp fields
    timestamp = variables.timestamp
    set_bufr_value(ibufr, "year", timestamp.year)
    set_bufr_value(ibufr, "month", timestamp.month)
    set_bufr_value(ibufr, "day", timestamp.day)
    set_bufr_value(ibufr, "hour", timestamp.hour)
    set_bufr_value(ibufr, "minute", timestamp.minute)

    set_bufr_value(ibufr, "relativeHumidity", variables.relativeHumidity)
    set_bufr_value(ibufr, "airTemperature", variables.airTemperature)
    set_bufr_value(ibufr, "pressure", variables.pressure)
    set_bufr_value(ibufr, "windDirection", variables.windDirection)
    set_bufr_value(ibufr, "windSpeed", variables.windSpeed)

    set_bufr_value(ibufr, "latitude", variables.latitude)

    # Set position metadata
    set_bufr_value(ibufr, "latitude", variables.latitude)
    set_bufr_value(ibufr, "longitude", variables.longitude)
    set_bufr_value(
        ibufr,
        "heightOfStationGroundAboveMeanSeaLevel",
        variables.heightOfStationGroundAboveMeanSeaLevel,
    )  # also height and heightOfStation?

    # The ## in the codes_set() indicate the position in the BUFR for the parameter.
    # e.g. #10#timePeriod will assign to the 10th occurence of "timePeriod", which corresponds
    # to the wind speed section. Note that both the "synopMobil" and "synopLand" templates
    # appear to have the same positions for all parameters that are set here.
    # View the output BUFR to see section keys with 'bufr_dump filename.bufr'.
    if math.isnan(variables.windSpeed) is False:
        # Set time significance (2=temporally averaged)
        codes_set(ibufr, "#1#timeSignificance", 2)
        # Set monitoring time period (-10=10 minutes)
        codes_set(ibufr, "#10#timePeriod", -10)

    # Set measurement heights
    # TODO: the bufr parameters ware previsously set using codes_set. setBUFR skips in case of nan. Is this preferable?
    set_bufr_value(
        ibufr,
        "#1#heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform",
        variables.heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH,
    )
    set_bufr_value(
        ibufr,
        "#7#heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform",
        variables.heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD,
    )
    set_bufr_value(
        ibufr,
        "heightOfBarometerAboveMeanSeaLevel",
        variables.heightOfBarometerAboveMeanSeaLevel,
    )  # For pressure


def set_bufr_value(ibufr, b_name, value):
    """Set variable in BUFR message
    Called in setAWSvariables() to make sure we aren't passing NaNs

    Parameters
    ----------
    ibufr : bufr.msg
        Active BUFR message
    b_name : str
        BUFR message variable name
    value : int/float
        Value to be assigned to variable
    """
    if math.isnan(value) is False:
        try:
            codes_set(ibufr, b_name, value)
        except CodesInternalError as ec:
            print(f"{ec}: {b_name}")
            print("-----> CodesInternalError in setBUFRvalue!")
            raise  # throw error back to getBUFR where it is handled
    else:
        print("----> {} {}".format(b_name, value))


def read_bufr_message(fp: BinaryIO) -> Optional[BUFRVariables]:
    """
    Read and parse BUFR message from binary IO stream.

    Extract AWS variables similar to the input to bufr_utilities.write_bufr_message.
    Note: stid is not written to the BUFR file hence it will be set to None in the output.

    Parameters
    ----------
    fp
        Readable binary io stream

    Returns
    -------
    BUFRVariables
        AWS variables or None if there are no messages in stream
    """
    ibufr = codes_bufr_new_from_file(fp)
    if ibufr is None:
        return None
    codes_set(ibufr, "unpack", 1)

    year = codes_get(
        ibufr,
        "year",
    )
    month = codes_get(
        ibufr,
        "month",
    )
    day = codes_get(
        ibufr,
        "day",
    )
    hour = codes_get(
        ibufr,
        "hour",
    )
    minute = codes_get(
        ibufr,
        "minute",
    )
    timestamp = datetime.datetime(
        year=year, month=month, day=day, hour=hour, minute=minute
    )

    # Determine template
    unexpanded_descriptors = codes_get(ibufr, "unexpandedDescriptors")
    if unexpanded_descriptors == 307090:
        # "synopMobil"
        station_type = "mobile"
        wmo_id = codes_get(ibufr, "shipOrMobileLandStationIdentifier")
    elif unexpanded_descriptors == 307080:
        # "synopLand"
        station_type = "land"
        # Note: stationNumber is an integer
        station_number = codes_get(ibufr, "stationNumber")
        wmo_id = str(station_number)
    else:
        raise ValueError(
            f"Unknown BUFR template unexpandedDescriptors: {unexpanded_descriptors}"
        )

    variables = BUFRVariables(
        timestamp=timestamp,
        relativeHumidity=codes_get(ibufr, "relativeHumidity"),
        airTemperature=codes_get(ibufr, "airTemperature"),
        pressure=codes_get(ibufr, "pressure"),
        windDirection=codes_get(ibufr, "windDirection"),
        windSpeed=codes_get(ibufr, "windSpeed"),
        latitude=codes_get(ibufr, "latitude"),
        longitude=codes_get(ibufr, "longitude"),
        heightOfStationGroundAboveMeanSeaLevel=codes_get(
            ibufr, "heightOfStationGroundAboveMeanSeaLevel"
        ),
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformTempRH=codes_get(
            ibufr, "#1#heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform"
        ),
        heightOfSensorAboveLocalGroundOrDeckOfMarinePlatformWSPD=codes_get(
            ibufr, "#7#heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform"
        ),
        heightOfBarometerAboveMeanSeaLevel=codes_get(
            ibufr, "heightOfBarometerAboveMeanSeaLevel"
        ),
        wmo_id=wmo_id,
        station_type=station_type,
    )
    codes_release(ibufr)

    return variables


def read_bufr_file(path: PathLike) -> pd.DataFrame:
    """
    Read aws data from all messages in a bufr file.

    Parameters
    ----------
    path : PathLike
        Path to bufr file

    Returns
    -------
    pd.DataFrame

    """
    path = Path(path)
    lines = []
    with path.open("rb") as fp:
        while True:
            message_vars = read_bufr_message(fp)
            if message_vars is None:
                break
            lines.append(message_vars)
    return pd.DataFrame(lines).rename_axis("message_index")
