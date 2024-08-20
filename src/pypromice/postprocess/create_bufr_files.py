import logging
from pathlib import Path
from typing import Sequence, List

import pandas as pd
from pypromice.station_configuration import load_station_configuration_mapping

from pypromice.postprocess.get_bufr import (
    get_bufr,
    DEFAULT_LIN_REG_TIME_LIMIT,
    DEFAULT_POSITION_SEED_PATH,
)

main_logger = logging.getLogger(__name__)


def create_bufr_files(
    input_files: Sequence[Path],
    station_configuration_root: Path,
    period_start: str,
    period_end: str,
    output_root: Path,
    override: bool,
    break_on_error: bool = False,
    output_filename_suffix: str = "geus_",
):
    """
    Generate hourly bufr files from the for all input files

    Parameters
    ----------
    input_files
        Paths to csv l3 hourly data files
    station_configuration_root
        Root directory containing station configuration toml files
    period_start
        Datetime string for period start. Eg '2024-01-01T00:00' or '20240101
    period_end
        Datetime string for period end
    output_root
        Output dir for both bufr files for individual stations and compiled. Organized in two sub directories.
    override
        If False: Skip a period if the compiled output file exists.
    break_on_error
        If True: Stop processing if an error occurs
    output_filename_suffix
        Suffix for the compiled output file

    """
    periods = pd.date_range(period_start, period_end, freq="h")
    output_individual_root = output_root / "individual"
    output_compiled_root = output_root / "compiled"
    output_individual_root.mkdir(parents=True, exist_ok=True)
    output_compiled_root.mkdir(parents=True, exist_ok=True)

    station_configuration_mapping = load_station_configuration_mapping(
        station_configuration_root,
        skip_unexpected_fields=True,
    )

    for period in periods:
        period: pd.Timestamp
        date_str = period.strftime("%Y%m%dT%H%M")
        main_logger.info(f"Processing {date_str}")
        output_dir_path = output_individual_root / f"{date_str}"
        output_file_path = (
            output_compiled_root / f"{output_filename_suffix}{date_str}.bufr"
        )

        main_logger.info(f"{period}, {date_str}")
        if override or not output_file_path.exists():
            get_bufr(
                bufr_out=output_dir_path,
                input_files=input_files,
                store_positions=False,
                positions_filepath=None,
                linear_regression_time_limit=DEFAULT_LIN_REG_TIME_LIMIT,
                timestamps_pickle_filepath=None,
                target_timestamp=period,
                station_configuration_mapping=station_configuration_mapping,
                positions_seed_path=DEFAULT_POSITION_SEED_PATH,
                break_on_error=break_on_error,
            )

            with output_file_path.open("wb") as fp_dst:
                for src_path in output_dir_path.glob("*.bufr"):
                    with src_path.open("rb") as fp_src:
                        fp_dst.write(fp_src.read())
        else:
            main_logger.info(f"Output file exists. Skipping {output_file_path}")


# %%


def main():
    import argparse
    import glob
    import sys

    logger_format_string = "%(asctime)s; %(levelname)s; %(name)s; %(message)s"
    logging.basicConfig(
        level=logging.ERROR,
        stream=sys.stdout,
        format=logger_format_string,
    )

    main_handler = logging.StreamHandler(sys.stdout)
    main_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(logger_format_string)
    main_handler.setFormatter(formatter)
    main_logger.addHandler(main_handler)
    main_logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser("Create BUFR files from L3 tx .csv files.")
    parser.add_argument(
        "--input_files",
        "--l3-filepath",
        "-i",
        type=Path,
        nargs="+",
        required=True,
        help="Path to L3 tx .csv files. Can be direct paths or glob patterns",
    )
    parser.add_argument(
        "--period_start",
        "-s",
        required=True,
        help="Datetime string for period start. Eg '2024-01-01T00:00' or '20240101",
    )
    parser.add_argument(
        "--period_end", "-e", required=True, help="Datetime string for period end"
    )
    parser.add_argument(
        "--output_root",
        "-o",
        required=True,
        type=Path,
        help="Output dir for both bufr files for individual stations and compiled. Organized in two sub directories.",
    )
    parser.add_argument(
        "--station_configuration_root",
        "-c",
        required=True,
        type=Path,
        help="Root directory containing station configuration toml files",
    )
    parser.add_argument(
        "--override",
        "-f",
        default=False,
        action="store_true",
        help="Recreate and overide existing output files",
    )
    args = parser.parse_args()

    # Interpret all input file paths as glob patterns if they don't exist
    input_files: List[Path] = list()
    for path in args.input_files:
        if path.exists():
            input_files.append(path)
        else:
            # The input path might be a glob pattern
            input_files += map(Path, glob.glob(path.as_posix()))

    main_logger.info(f"Processing {len(input_files)} input files")
    create_bufr_files(
        input_files=input_files,
        period_start=args.period_start,
        period_end=args.period_end,
        output_root=args.output_root,
        override=args.override,
        station_configuration_root=args.station_configuration_root,
    )


if __name__ == "__main__":
    main()
