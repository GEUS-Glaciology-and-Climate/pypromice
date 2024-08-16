import csv
from pathlib import Path
from typing import Dict, Union

import pandas as pd

DEFAULT_METADATA_PATH = (Path(__file__).parent / "file_attributes.csv").absolute()
DEFAULT_VARIABLES_PATH = (Path(__file__).parent / "variables.csv").absolute()
DEFAULT_VARIABLES_ALIASES_GCNET_PATH = (Path(__file__).parent / "variable_aliases_GC-Net.csv").absolute()

def load_metadata(path: Union[None, str, Path] = None) -> Dict[str, str]:
    """
    Load metadata table from csv file
    """
    if path is None:
        path = DEFAULT_METADATA_PATH
    with open(path, "r") as f:
        csv_reader = csv.reader(f)
        return {row[0]: row[1] for row in csv_reader}


def load_variables(path: Union[None, str, Path] = None) -> pd.DataFrame:
    """
    Load variables table from csv file
    """
    if path is None:
        path = DEFAULT_VARIABLES_PATH
    return pd.read_csv(path, index_col=0, comment="#")
