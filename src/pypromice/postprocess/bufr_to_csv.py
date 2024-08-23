import argparse
from pathlib import Path

import pandas as pd

from pypromice.postprocess.bufr_utilities import read_bufr_file


def main():
    parser = argparse.ArgumentParser("BUFR to CSV converter")
    parser.add_argument("path", type=Path, nargs='+')
    args = parser.parse_args()

    paths = []
    for path in args.path:
        paths += list(path.parent.glob(path.name))

    df = pd.concat([read_bufr_file(path) for path in paths])
    print(df.to_csv())


if __name__ == "__main__":
    main()
