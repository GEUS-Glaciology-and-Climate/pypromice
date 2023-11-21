import logging
import shutil
from pathlib import Path

import attrs

from pypromice.operational.station_status import get_modified_timestamp
import pypromice.process

logger = logging.getLogger(__name__)


@attrs.define
class AWSFileshare:
    root: Path = attrs.field(converter=Path)
    exclude = ("XXX", "CEN")

    @property
    def l3_flat_path(self) -> Path:
        return self.root.joinpath("aws-l3-flat")

    def l3_export(self, l3_repository: Path):
        logger.info("Exporting l3 data")
        for directory in ["level_3", "tx", "raw"]:
            output_dir = self.root.joinpath("aws-l3", directory)
            input_dir = l3_repository.joinpath(directory)

            for src_path in input_dir.glob("*/*"):
                if src_path.stem in self.exclude:
                    continue
                dst_path = output_dir.joinpath(src_path.relative_to(input_dir))

                if self.is_new(src_path, dst_path):
                    logger.debug(f"Exporting {src_path} -> {dst_path}")
                    dst_path.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copy(src_path, dst_path)

        logger.info("Exporting AWS_latest_locations")
        latest_location_path = l3_repository.joinpath("AWS_latest_locations.csv")
        if latest_location_path.exists():
            shutil.copy(
                latest_location_path, self.root.joinpath("AWS_latest_locations.csv")
            )

        logger.info("Exporting variables.csv")
        variables_path = Path(pypromice.process.__file__).parent.joinpath(
            "variables.csv"
        )
        shutil.copy(variables_path, self.root.joinpath("AWS_variables.csv"))

    @staticmethod
    def is_new(src_path: Path, dst_path: Path) -> bool:
        if not dst_path.exists():
            return True
        if get_modified_timestamp(src_path) > get_modified_timestamp(dst_path):
            return True
        return False

    def l3_flat_export(self, l3_repository: Path):
        logger.info("Export l3 flat")
        self.l3_flat_export_level3(l3_repository)
        self.l3_flat_export_rawtx(l3_repository)

    def l3_flat_export_rawtx(self, l3_repository: Path):
        for dir_name in ["raw", "tx"]:
            output_dir = self.l3_flat_path.joinpath(dir_name)
            output_dir.mkdir(parents=True, exist_ok=True)
            for src_path in l3_repository.joinpath(dir_name).glob("*/*"):
                dst_path = output_dir.joinpath(src_path.name)
                if self.is_new(src_path, dst_path):
                    logger.debug(f"l3_flat_export, {dir_name}: {src_path} -> {dst_path}")
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src_path, dst_path)

    def l3_flat_export_level3(self, l3_repository: Path):
        output_dir = self.l3_flat_path.joinpath("level_3")
        for src_path in l3_repository.joinpath('level_3').glob("*/*"):
            period = src_path.stem.rsplit("_")[-1]
            dst_path = output_dir.joinpath(period, src_path.name)
            if self.is_new(src_path, dst_path):
                logger.debug(f"l3_flat_export, level3 {src_path} -> {dst_path}")
                dst_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(src_path, dst_path)

    def sync(self, l3_repository: Path):
        self.l3_export(l3_repository)
        self.l3_flat_export(l3_repository)
