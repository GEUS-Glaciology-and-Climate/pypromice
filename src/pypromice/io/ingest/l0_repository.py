"""
Module for managing Level 0 data repositories for station-based datasets.

This module provides an abstraction for interacting with Level 0 (L0) datasets through
a repository interface. Two implementations are detailed: the `L0Repository` protocol
defines the interface, and `L0RepositoryFS` implements the interface using a file system-based
repository structure. This is intended for managing both raw and transformed datasets, along
with their configurations, for multiple stations.

Classes:
    L0Repository: Protocol interface for accessing L0 datasets and metadata.
    L0RepositoryFS: File system-based implementation of the `L0Repository` protocol.

Functions and attributes exposed:
    - Methods to query and manage raw and transformed datasets.
    - Mechanisms to verify dataset presence and access configuration paths.

"""

import dataclasses
from pathlib import Path
from typing import List, Protocol, Iterable

import xarray as xr

__all__ = [
    "L0Repository",
    "L0RepositoryFS",
]

from .l0 import load_config, load_data_files


class L0Repository(Protocol):
    def get_tx(self, station_id: str) -> Iterable[xr.Dataset]: ...
    def get_raw(self, station_id: str) -> Iterable[xr.Dataset]: ...
    def get_available_stations(self) -> Iterable[str]: ...
    def contains_tx(self, station_id: str) -> bool: ...
    def contains_raw(self, station_id: str) -> bool: ...


@dataclasses.dataclass(slots=True)
class L0RepositoryFS:
    root: Path

    template_tx_config = "tx/config/{station_id}.toml"
    template_tx_data_root = "tx/"
    template_raw_config = "raw/config/{station_id}.toml"
    template_row_data_root = "raw/{station_id}/"

    def get_tx_config_path(self, station_id: str) -> Path:
        return self.root / self.template_tx_config.format(station_id=station_id)

    def get_tx_data_root(self, station_id: str) -> Path:
        return self.root / self.template_tx_data_root.format(station_id=station_id)

    def get_raw_config_path(self, station_id: str) -> Path:
        return self.root / self.template_raw_config.format(station_id=station_id)

    def get_raw_data_root(self, station_id: str) -> Path:
        return self.root / self.template_row_data_root.format(station_id=station_id)

    def contains_tx(self, station_id: str) -> bool:
        return self.get_tx_config_path(station_id).exists()

    def contains_raw(self, station_id: str) -> bool:
        return self.get_raw_config_path(station_id).exists()

    def get_tx(self, station_id: str) -> List[xr.Dataset]:
        return load_data_files(self.get_tx_config(station_id))

    def get_tx_config(self, station_id):
        return load_config(
            self.get_tx_config_path(station_id),
            self.get_tx_data_root(station_id),
        )

    def get_raw(self, station_id: str) -> List[xr.Dataset]:
        return load_data_files(self.get_raw_config(station_id))

    def get_raw_config(self, station_id):
        return load_config(
            self.get_raw_config_path(station_id),
            self.get_raw_data_root(station_id),
        )

    def get_available_stations(self) -> List[str]:
        """
        Iterate over all available station configuration files

        """
        tx_pattern = self.get_tx_config_path("*")
        raw_pattern = self.get_raw_config_path("*")

        station_ids = {
            p.stem
            for p in [
                *tx_pattern.parent.glob(tx_pattern.name),
                *raw_pattern.parent.glob(raw_pattern.name),
            ]
        }

        return sorted(station_ids)
