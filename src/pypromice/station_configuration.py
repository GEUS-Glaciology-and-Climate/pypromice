from pathlib import Path
from typing import Optional, Dict, Mapping, Sequence

import attrs
import toml


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
    static_height_of_gps_from_mean_sea_level: Optional[float] = None
    station_relocation: Sequence[str] = attrs.field(factory=list)

    # The station data will be exported to BUFR if True. Otherwise, it will only export latest position
    export_bufr: bool = False
    comment: Optional[str] = None

    # skip specific variables for stations
    # If a variable has known bad data, use this collection to skip the variable
    # Note that if a station is not reporting both air temp and pressure it will be skipped,
    # as currently implemented in csv2bufr.min_data_check().
    # ['p_i'], # EXAMPLE
    skipped_variables: Sequence[str] = attrs.field(factory=list)

    positions_update_timestamp_only: bool = False

    @classmethod
    def load_toml(cls, path):
        return cls(**toml.load(path))

    def dump_toml(self, path: Path):
        with path.open("w") as fp:
            toml.dump(self.as_dict(), fp)

    def as_dict(self) -> Dict:
        return attrs.asdict(self)


def load_station_configuration_mapping(
    configuration_root_dir: Path,
) -> Mapping[str, StationConfiguration]:
    """
    Load station configurations from toml files in configuration_root_dir

    Parameters
    ----------
    configuration_root_dir
        Root directory containing toml files

    Returns
    -------
    Mapping from stid to StationConfiguration

    """
    return {
        config_file.stem: StationConfiguration(**toml.load(config_file))
        for config_file in configuration_root_dir.glob("*.toml")
    }


def write_station_configuration_mapping(
    station_configurations: Mapping[str, StationConfiguration],
    configuration_root_dir: Path,
) -> None:
    """
    Write station configurations to toml files in configuration_root_dir

    Parameters
    ----------
    station_configurations
        Mapping from stid to StationConfiguration
    configuration_root_dir
        Output directory

    """
    configuration_root_dir.mkdir(parents=True, exist_ok=True)
    for stid, station_configuration in station_configurations.items():
        with (configuration_root_dir / f"{stid}.toml").open("w") as fp:
            toml.dump(station_configuration.as_dict(), fp)
