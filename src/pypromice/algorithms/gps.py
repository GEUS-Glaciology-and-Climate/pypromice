import re
import numpy as np
import pandas as pd

def decode(gps):
    '''Decode GPS information based on names of GPS attributes. This should be
    applied if gps information does not consist of float values

    Parameters
    ----------
    ds : xr.Dataset
        Data set
    gps_names : list
        Variable names for GPS information, such as "gps_lat", "gps_lon" and
        "gps_alt"

    Returns
    -------
    ds : xr.Dataset
        Data set with decoded GPS information
    '''
    a = gps.attrs
    str2nums = [re.findall(r"[-+]?\d*\.\d+|\d+", _) if isinstance(_, str) else [np.nan] for _ in gps.values]
    gps[:] = pd.DataFrame(str2nums).astype(float).T.values[0]
    gps = gps.astype(float)
    gps.attrs = a
    return gps

def reformat(pos_arr, attrs):
    '''Correct latitude and longitude from native format to decimal degrees.

    v2 stations should send  "NH6429.01544","WH04932.86061" (NUK_L 2022)
    v3 stations should send coordinates as "6628.93936","04617.59187" (DY2) or 6430,4916 (NUK_Uv3)
    gps.decode should have decoded these strings to floats in ddmm.mmmm format
    v1 stations however only saved decimal minutes (mm.mmmmm) as float<=60. '
    In this case, we use the integer part of the latitude given in the config
    file and append the gps value after it.

    Parameters
    ----------
    pos_arr : xr.Dataarray
        Array of latitude or longitude measured by the GPS
    attrs : dict
        The global attribute 'latitude' or 'longitude' associated with the
        file being processed. It is the standard latitude/longitude given in the
        config file for that station.

    Returns
    -------
    pos_arr : xr.Dataarray
        Formatted GPS position array in decimal degree
    '''
    if np.any((pos_arr <= 90) & (pos_arr > 0)):

        # then pos_arr is in decimal minutes, so we add to it the integer
        # part of the latitude given in the config file x100
        # so that it reads ddmm.mmmmmm like for v2 and v3 files
        # Note that np.sign and np.attrs handles negative longitudes.
        pos_arr = np.sign(attrs) * (pos_arr + 100 * np.floor(np.abs(attrs)))

    a = pos_arr.attrs
    pos_arr = np.floor(pos_arr / 100) + (pos_arr / 100 - np.floor(pos_arr / 100)) * 100 / 60
    pos_arr.attrs = a
    return pos_arr
