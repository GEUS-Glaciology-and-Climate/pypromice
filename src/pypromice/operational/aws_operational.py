import datetime
import logging
import multiprocessing
import sys
from pathlib import Path
from typing import List, Optional, Mapping

import attrs
import pandas as pd

from pypromice.operational.fileshare import AWSFileshare
from pypromice.operational.git_repository_utils import (
    prepare_git_repository,
    execute_git_command,
    DataRepositoryError,
)
from pypromice.operational.multiprocessing_utils import KwargsMap, Task, TaskAware
from pypromice.operational.station_status import (
    load_station_status,
    save_station_status,
    instantiate_station_status,
    get_modified_timestamp,
    parse_l0_tx_filename,
)
from pypromice.postprocess.bufr_upload import (
    concat_bufr_files,
    upload_bufr,
)
from pypromice.postprocess.get_bufr import get_bufr
from pypromice.process.get_l3 import get_l3
from pypromice.process.join_l3 import join_l3
from pypromice.tx.get_l0tx import get_l0tx
from pypromice.utilities.config_io import load_toml_files

# %%
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DEFAULT_L0_REPOSITORY_BRANCH = "main"
DEFAULT_PROMICE_DATA_ISSUES_BRANCH = "master"


@attrs.define(kw_only=True)
class AWSOperational:
    l0_repository_path: Path = attrs.field(converter=Path)
    l3_repository_path: Path = attrs.field(converter=Path)
    promice_data_issues: Path = attrs.field(converter=Path)
    bufr_output_path: Path = attrs.field(converter=Path)
    dmi_config: Optional[Mapping] = attrs.field(repr=False)
    gmail_config: Optional[Mapping] = attrs.field(repr=False)
    l0_tx_uid_path: Path = attrs.field(
        converter=Path,
    )
    aws_fileshare: AWSFileshare

    status_path: Path = attrs.field(converter=Path)

    l0_repository_branch: str = DEFAULT_L0_REPOSITORY_BRANCH
    promice_data_issues_branch: str = DEFAULT_PROMICE_DATA_ISSUES_BRANCH
    max_processes: Optional[int] = None

    _station_status: pd.DataFrame = attrs.field(factory=instantiate_station_status)

    def __attrs_post_init__(self):
        if self.status_path.exists():
            self._station_status = load_station_status(self.status_path)

    @classmethod
    def from_config_file(cls, *config_files: Path) -> "AWSOperational":
        config = load_toml_files(*config_files)
        aws_operational = config["aws_operational"]
        gmail_config = config.get("aws")
        dmi_config = config.get("dmi")

        aws_fileshare = AWSFileshare(root=aws_operational.pop("aws_fileshare"))

        return cls(
            **aws_operational,
            aws_fileshare=aws_fileshare,
            gmail_config=gmail_config,
            dmi_config=dmi_config,
        )

    def get_station_status(self) -> pd.DataFrame:
        return (
            self._station_status.groupby(level=[0, 1])
            .max()
            .reset_index()
            .pivot(
                index="stid",
                columns="step",
                values="datetime",
            )
        )

    def get_stids_to_be_updated(
        self,
        dependency_steps: List[str],
        output_step: str,
        max_stations: Optional[int] = None,
    ) -> List[str]:
        station_status = self.get_station_status()
        if any(set(dependency_steps) - set(station_status.columns)):
            # There are no AWSs with all dependencies
            return []
        has_dependencies = ~station_status[dependency_steps].isna().max(1)
        if output_step not in station_status:
            # There are no AWSs with output data
            return list(station_status.index[has_dependencies])

        # Note the date time values might be nan.
        # This expression therefore means notnan and dependecies < output
        is_up_to_date = (
            station_status[dependency_steps].max(1) < station_status[output_step]
        )
        to_update = has_dependencies & ~is_up_to_date
        return list(station_status.index[to_update])[:max_stations]

    def read_l0_file_modified(self):
        lines = []

        for file_path in self.l0_tx_path.glob("config/*.toml"):
            lines.append(
                dict(
                    stid=file_path.stem,
                    path=file_path,
                    modified_datetime=get_modified_timestamp(file_path),
                    step="l0_tx",
                )
            )
        for file_path in self.l0_tx_path.glob("*.txt"):
            lines.append(
                dict(
                    **parse_l0_tx_filename(file_path.name),
                    path=file_path,
                    modified_datetime=get_modified_timestamp(file_path),
                    step="l0_tx",
                )
            )
        for file_path in self.l0_raw_path.glob("config/*.toml"):
            lines.append(
                dict(
                    stid=file_path.stem,
                    path=file_path,
                    modified_datetime=get_modified_timestamp(file_path),
                    step="l0_raw",
                )
            )
        for file_path in self.l0_raw_path.glob("*/*.txt"):
            lines.append(
                dict(
                    stid=file_path.parent.stem,
                    path=file_path,
                    modified_datetime=get_modified_timestamp(file_path),
                    step="l0_raw",
                )
            )
        file_system_station_status = (
            pd.DataFrame(lines)
            .groupby(["stid", "step"])
            .agg(datetime=("modified_datetime", "max"))
        )
        self._station_status = file_system_station_status.combine_first(
            self._station_status
        )
        self.save_status()

    def save_status(self):
        save_station_status(self.status_path, self._station_status)

    def _add_result(self, stid: str, step: str, failed: bool):
        self._station_status.loc[(stid, step), ["datetime", "failed"]] = (
            datetime.datetime.now().astimezone(datetime.timezone.utc),
            failed,
        )
        self.save_status()

    @property
    def l0_raw_path(self) -> Path:
        return self.l0_repository_path.joinpath("raw")

    @property
    def l0_tx_path(self) -> Path:
        return self.l0_repository_path.joinpath("tx")

    @property
    def l3_raw_path(self) -> Path:
        return self.l3_repository_path.joinpath("raw")

    @property
    def l3_tx_path(self) -> Path:
        return self.l3_repository_path.joinpath("tx")

    @property
    def l3_level3_path(self) -> Path:
        return self.l3_repository_path.joinpath("level_3")

    def pull_l0_repository(self):
        prepare_git_repository(self.l0_repository_path, self.l0_repository_branch)
        self.read_l0_file_modified()

    def pull_issues_repository(self):
        prepare_git_repository(
            self.promice_data_issues, self.promice_data_issues_branch
        )

    def get_l0tx(self):
        get_l0tx(
            config_dir=self.l0_tx_path.joinpath("config").as_posix(),
            uid_file=self.l0_tx_uid_path.as_posix(),
            out_dir=self.l0_tx_path.as_posix(),
        )

    def commit_l0_repository(self):
        date_string = (
            datetime.datetime.now()
            .astimezone(datetime.timezone.utc)
            .strftime("%a %b %d %H:%M:%S UTC %Y")
        )
        commit_message = f"L0 update {date_string}"
        try:
            execute_git_command(self.l0_repository_path, ["add", "*"])
            execute_git_command(
                self.l0_repository_path, ["commit", "-m", f'"{commit_message}"']
            )
        except DataRepositoryError as e:
            logger.warning(e)

    def push_l0_repository(self):
        logger.info("pushing l0 commit")
        try:
            execute_git_command(self.l0_repository_path, ["push"])
        except DataRepositoryError as e:
            logger.warning(e)

    def process_tx(self, max_stations: Optional[int] = None):
        updated_stids = self.get_stids_to_be_updated(
            dependency_steps=["l0_tx"], output_step="l3_tx", max_stations=max_stations
        )

        logger.info(f"Processing tx L3 on {','.join(updated_stids)}")
        self.l3_tx_path.mkdir(parents=True, exist_ok=True)

        inputs: List[Task] = [
            Task(
                stid=stid,
                value=dict(
                    config_file=self.l0_tx_path.joinpath(f"config/{stid}.toml"),
                    inpath=self.l0_tx_path,
                    outpath=self.l3_tx_path,
                ),
            )
            for stid in updated_stids
        ]
        with multiprocessing.Pool(processes=self.max_processes) as pool:
            for result in pool.imap(TaskAware(KwargsMap(get_l3)), inputs):
                logger.info(
                    f"Processed l3 tx: {result.stid}, Success: {not result.failed}"
                )
                self._add_result(stid=result.stid, step="l3_tx", failed=result.failed)
        self.save_status()
        logger.info("Finished AWS L0 tx >> L3 processing")

    def process_raw(self, max_stations: Optional[int] = None):
        updated_stids = self.get_stids_to_be_updated(
            dependency_steps=["l0_raw"], output_step="l3_raw", max_stations=max_stations
        )

        logger.info(f"Processing raw L3 on {','.join(updated_stids)}")
        self.l3_raw_path.mkdir(parents=True, exist_ok=True)
        inputs: List[Task] = [
            Task(
                stid=stid,
                value=dict(
                    config_file=self.l0_raw_path.joinpath(f"config/{stid}.toml"),
                    inpath=self.l0_raw_path,  # TODO: Shouldn't there be an stid sub dir
                    outpath=self.l3_raw_path,
                ),
            )
            for stid in updated_stids
        ]
        with multiprocessing.Pool(processes=self.max_processes) as pool:
            for result in pool.imap(TaskAware(KwargsMap(get_l3)), inputs):
                logger.info(
                    f"Processed l3 raw: {result.stid}, Success: {not result.failed}"
                )
                logger.info(result)
                self._add_result(stid=result.stid, step="l3_raw", failed=result.failed)
        self.save_status()
        logger.info("Finished AWS L0 raw >> L3 processing")

    def export_bufr(self):
        logger.info("Running BUFR file export for DMI/WMO...")

        bufr_aws_path = self.bufr_output_path.joinpath("aws")
        bufr_aws_path.mkdir(parents=True, exist_ok=True)
        bufr_concat_path = self.bufr_output_path.joinpath("concat")
        bufr_concat_path.mkdir(parents=True, exist_ok=True)

        latest_locations_path = self.l3_repository_path.joinpath(
            "AWS_latest_locations.csv"
        )
        l3_file_path = self.l3_tx_path.joinpath("*/*_hour.csv")
        timestamps_pickle_filepath = self.bufr_output_path.joinpath(
            "latest_timestamps.pickle"
        )
        logger.info(f"Delete old bufr files for {bufr_aws_path}")
        for p in bufr_aws_path.glob("*.bufr"):
            p.unlink()
        get_bufr(
            store_positions=False,
            bufr_out=bufr_aws_path.as_posix(),
            positions_filepath=latest_locations_path.as_posix(),
            l3_filepath=l3_file_path.as_posix(),
            dev=True,
            timestamps_pickle_filepath=timestamps_pickle_filepath.as_posix(),
        )

        time_sting = datetime.datetime.now().strftime("%Y%m%dT%H%M")
        concat_filename = f"geus_{time_sting}.bufr"
        concat_output_path = bufr_concat_path.joinpath(concat_filename)
        logger.info(f"Concatenate BUFR file. {concat_output_path}")
        concat_bufr_files(self.bufr_output_path, concat_output_path)
        logger.info(
            f"Size of concatenated file: {concat_output_path.stat().st_size} bytes"
        )

        if False and isinstance(self.dmi_config, Mapping):
            logger.info("Upload to DMI")
            upload_bufr(concat_output_path, **self.dmi_config)
            logger.info("Done")
        else:
            logger.info("Skipping BUFR upload")

    def process_level3(self, max_stations: Optional[int] = None):
        updated_stids = self.get_stids_to_be_updated(
            dependency_steps=["l3_raw", "l3_tx"],
            output_step="level3",
            max_stations=max_stations,
        )
        logger.info(f"Running AWS L3 RAW and TX joiner on {','.join(updated_stids)}")
        self.l3_level3_path.mkdir(parents=True, exist_ok=True)
        inputs: List[Task] = [
            Task(
                stid=stid,
                value=dict(
                    file1_path=self.l3_raw_path.joinpath(
                        stid, f"{stid}_10min.csv"
                    ).as_posix(),
                    file2_path=self.l3_tx_path.joinpath(
                        stid, f"{stid}_hour.csv"
                    ).as_posix(),
                    output_path=self.l3_level3_path.as_posix(),
                    data_type="raw",
                ),
            )
            for stid in updated_stids
        ]
        with multiprocessing.Pool(processes=self.max_processes) as pool:
            for result in pool.imap(TaskAware(KwargsMap(join_l3)), inputs):
                logger.info(
                    f"Processed level3: {result.stid}, Success: {not result.failed}"
                )
                self._add_result(stid=result.stid, step="level3", failed=result.failed)
        self.save_status()

    def sync_l3(self):
        self.aws_fileshare.sync(self.l3_repository_path)


# %% Production instance


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=logging.INFO,
    )

    aws_operational = AWSOperational.from_config_file(
        Path("/Users/maclu/work/pypromice/configurations/maclu_laptop.toml"),
        Path("/Users/maclu/work/pypromice/credentials/credentials.toml"),
    )
