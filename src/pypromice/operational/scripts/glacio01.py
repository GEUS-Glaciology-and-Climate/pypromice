import logging
import sys
from pathlib import Path

from pypromice.operational.aws_operational import AWSOperational

def glacio01(aws_operational: AWSOperational):
    aws_operational.pull_l0_repository()
    aws_operational.pull_issues_repository()
    aws_operational.read_l0_file_modified()
    aws_operational.process_tx()
    aws_operational.export_bufr()
    aws_operational.process_raw()
    aws_operational.process_level3()
    aws_operational.commit_l0_repository()


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
    glacio01(aws_operational)
