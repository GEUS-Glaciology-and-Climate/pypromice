import random
import uuid

from pypromice.postprocess.bufr_utilities import BUFR_TEMPLATES
from pypromice.station_configuration import StationConfiguration

STATION_TYPE_STRINGS = tuple(BUFR_TEMPLATES.keys())


def get_station_configuration(**kwargs) -> StationConfiguration:
    """
    Create a StationConfiguration object with random values.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to providie explicit values for the StationConfiguration object.
    Returns
    -------
    """
    stid = kwargs.get("stid", str(uuid.uuid4()))
    station_site = kwargs.get("station_site", str(uuid.uuid4()))
    project = kwargs.get("project", str(uuid.uuid4()))
    station_type = kwargs.get("station_type", random.choice(STATION_TYPE_STRINGS))
    # WMO Station number <1024 for land stations
    # https://vocabulary-manager.eumetsat.int/vocabularies/BUFR/WMO/32/TABLE_B/001002
    wmo_id = kwargs.get("wmo_id", "{:05}".format(random.randint(0, 1023)))
    barometer_from_gps = kwargs.get("barometer_from_gps", random.random() * 3)
    anemometer_from_sonic_ranger = kwargs.get(
        "anemometer_from_sonic_ranger", random.random() * 3
    )
    temperature_from_sonic_ranger = kwargs.get(
        "temperature_from_sonic_ranger", random.random() * 3
    )
    height_of_gps_from_station_ground = kwargs.get(
        "height_of_gps_from_station_ground", random.random() * 3
    )
    sonic_ranger_from_gps = kwargs.get("sonic_ranger_from_gps", random.random() * 3)
    export_bufr = kwargs.get("export_bufr", random.random() > 0.5)
    skipped_variables = kwargs.get("skipped_variables", [])
    positions_update_timestamp_only = kwargs.get(
        "positions_update_timestamp_only", random.random() > 0.5
    )
    station_relocation = kwargs.get("station_relocation", [])

    return StationConfiguration(
        stid=stid,
        station_site=station_site,
        project=project,
        station_type=station_type,
        wmo_id=wmo_id,
        barometer_from_gps=barometer_from_gps,
        anemometer_from_sonic_ranger=anemometer_from_sonic_ranger,
        temperature_from_sonic_ranger=temperature_from_sonic_ranger,
        height_of_gps_from_station_ground=height_of_gps_from_station_ground,
        sonic_ranger_from_gps=sonic_ranger_from_gps,
        export_bufr=export_bufr,
        skipped_variables=skipped_variables,
        positions_update_timestamp_only=positions_update_timestamp_only,
        station_relocation=station_relocation,
    )
