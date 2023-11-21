import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd


def load_station_status(path: Path) -> pd.DataFrame:
    return (
        pd.read_csv(path)
        .set_index(["stid", "step"])
        .assign(datetime=lambda df: pd.to_datetime(df.datetime, format="ISO8601"))
    )


def save_station_status(path: Path, station_status: pd.DataFrame):
    station_status.to_csv(path, index=True)


def instantiate_station_status() -> pd.DataFrame:
    return pd.DataFrame(
        index=pd.MultiIndex(
            levels=[(), ()],
            names=["stid", "step"],
            codes=[(), ()],
        ),
        columns=["datetime", "failed"],
    )


def read_l0_file_modified(
    l0_tx_path: Path,
    l0_raw_path: Path,
) -> pd.DataFrame:
    lines = []

    for file_path in l0_tx_path.glob("config/*.toml"):
        lines.append(
            dict(
                stid=file_path.stem,
                path=file_path,
                modified_datetime=get_modified_timestamp(file_path),
                step="l0_tx",
            )
        )
    for file_path in l0_tx_path.glob("*.txt"):
        lines.append(
            dict(
                **parse_l0_tx_filename(file_path.name),
                path=file_path,
                modified_datetime=get_modified_timestamp(file_path),
                step="l0_tx",
            )
        )
    for file_path in l0_raw_path.glob("config/*.toml"):
        lines.append(
            dict(
                stid=file_path.stem,
                path=file_path,
                modified_datetime=get_modified_timestamp(file_path),
                step="l0_raw",
            )
        )
    for file_path in l0_raw_path.glob("*/*.txt"):
        lines.append(
            dict(
                stid=file_path.parent.stem,
                path=file_path,
                modified_datetime=get_modified_timestamp(file_path),
                step="l0_raw",
            )
        )
    return (
        pd.DataFrame(lines)
        .groupby(["stid", "step"])
        .agg(datetime=("modified_datetime", "max"))
    )


def get_modified_timestamp(path: Path) -> datetime:
    return datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    )


def parse_l0_tx_filename(filename: str) -> Dict:
    match = re.search(r"(\w+)_(\d*)_(\d)(-\w)?", filename)
    data = dict()
    if match is not None:
        data["stid"] = match.group(1)
        data["imei"] = match.group(2)
        data["imei_index"] = match.group(3)
        data["tx_flag"] = match.group(4)
    return data
