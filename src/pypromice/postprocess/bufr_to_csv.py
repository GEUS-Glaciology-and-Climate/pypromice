import argparse
from pathlib import Path

from pypromice.postprocess.bufr_utilities import read_bufr_file


def main():
    parser = argparse.ArgumentParser("BUFR to CSV converter")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    print(read_bufr_file(args.path).to_csv())


if __name__ == "__main__":
    main()
