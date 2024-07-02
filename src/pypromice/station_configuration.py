from os import PathLike
from typing import TypedDict, List

import toml

__all__ = [
    "StationConfig",
    "load_station_config",
]


class StationConfig(TypedDict):
    station_relocation: List[str]


def load_station_config(path: PathLike) -> StationConfig:
    """

    Parameters
    ----------
    path
        Path to station config toml file. Like aws-l0/metadata/station_configurations/

    Returns
    -------

    """
    with open(path) as fp:
        station_config = toml.load(fp)

    return station_config
