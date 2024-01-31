#!/usr/bin/env python
import logging, os, sys, unittest
from argparse import ArgumentParser
from pypromice.process.aws import AWS
from pathlib import Path
from typing import Optional

def parse_arguments_l3() -> ArgumentParser:
    parser = ArgumentParser(description="AWS L3 processor")

    parser.add_argument(
        "-c",
        "--config_file",
        type=Path,
        required=True,
        help="Path to config (TOML) file",
    )
    parser.add_argument(
        "-i",
        "--inpath",
        default="data",
        type=Path,
        required=True,
        help="Path to input data",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        default=None,
        type=Path,
        required=False,
        help="Path where to write output",
    )
    parser.add_argument(
        "-v",
        "--variables",
        default=None,
        type=Path,
        required=False,
        help="File path to variables look-up table",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        default=None,
        type=Path,
        required=False,
        help="File path to metadata",
    )
    return parser


def get_l3(
    config_file: Path,
    inpath: Path,
    outpath: Optional[Path] = None,
    metadata: Optional[Path] = None,
    variables: Optional[Path] = None,
):
    station_path = inpath.joinpath(config_file.stem)

    if station_path.exists():
        aws = AWS(config_file.as_posix(), station_path.as_posix(), variables.as_posix() if variables else None, metadata.as_posix() if metadata else None)
    else:
        aws = AWS(config_file.as_posix(), inpath.as_posix(), variables.as_posix() if variables else None, metadata.as_posix() if metadata else None)

    aws.process()

    if outpath is not None:
        aws.write(outpath.as_posix())

if __name__ == "__main__":
    parser = parse_arguments_l3()

    logging.basicConfig(
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    args = parser.parse_args()

    get_l3(
        config_file=args.config_file,
        inpath=args.inpath,
        outpath=args.outpath,
        metadata=args.metadata,
        variables=args.variables,
    )
