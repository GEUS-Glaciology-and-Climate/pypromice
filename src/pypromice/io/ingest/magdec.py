"""
Module for handling magnetic declination coefficients loading and parsing.

Functions
---------
- load_magdec_config: Read and proces data file to a configuration dictionary.
- magdec_config_to_array: Parse magnetic declination config dictionary to array.
"""

import logging
from pathlib import Path
import pandas as pd
import toml
import xarray as xr
import re
import numpy as np

__all__ = [
    "load_magdec_config",
    "magdec_config_to_array",
]

logger = logging.getLogger(__name__)


def load_magdec_config(path):
    """
    Load magnetic declination TOML config while tolerating
    non-standard `nan` literals.
    """

    with open(path, "r") as f:
        text = f.read()

    # Replace bare nan with quoted "nan"
    # but avoids touching words like "banana"
    text = re.sub(r'(?<![A-Za-z0-9_])nan(?![A-Za-z0-9_])', '"nan"', text)

    data = toml.loads(text)

    # Convert "nan" strings back to np.nan
    for station, entries in data.items():
        for entry in entries:
            for key, value in entry.items():
                if value == "nan":
                    entry[key] = np.nan

    return data

def magdec_config_to_array(
    magdec_config: dict,
    station_name: str
):
    """Extract magnetic declination coefficients from
    one station into an xarray.DataArray

    Parameters
    ----------
    magdec_config: dict
        Magnetic declination coefficients dictionary
    station_name: str
        Station name to be extracted

    Returns
    -------
    magdec : dict
        Magnetic Declination coefficients dictionary
    """
    # Extract entries by station name
    entries = magdec_config[station_name]

    # Format dates and coefficient values
    dates = pd.to_datetime([e["date"] for e in entries])
    values = [e["magnetic_declination"] for e in entries]

    # Construct DataArray object
    da = xr.DataArray(
        values,
        coords={"time": dates},
        dims=["time"],
        name="magnetic_declination",
        attrs={
            "latitude": entries[0]["latitude"],
            "longitude": entries[0]["longitude"],
            "altitude": entries[0]["altitude"],
            "model_name": entries[0]["model_name"],
            "units": "degrees",
        },
    )

    return da