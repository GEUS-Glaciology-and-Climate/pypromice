import logging
import sys
from pathlib import Path

from pypromice.operational.aws_operational import AWSOperational


def production(aws_operational: AWSOperational):
    """
    Assume the modified time stamps in the input folder are correct.
    This is not the case if it is a git repository after pull
    """
    aws_operational.pull_l0_repository()
    aws_operational.pull_issues_repository()
    aws_operational.get_l0tx()
    aws_operational.commit_l0_repository()
    aws_operational.process_tx()
    aws_operational.export_bufr()
    aws_operational.process_raw()
    aws_operational.process_level3()
    aws_operational.sync_l3()
    aws_operational.push_l0_repository()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--configuration_file', '-c', nargs='+', type=Path,
    )
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=logging.INFO,
    )

    aws_operational = AWSOperational.from_config_file(args.configuration_file)
    production(aws_operational)
