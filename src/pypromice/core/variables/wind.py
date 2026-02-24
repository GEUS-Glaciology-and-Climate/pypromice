__all__=['correct_wind_speed', 'filter_wind_direction', 'calculate_directional_wind_speed']

import numpy as np
import xarray as xr
from pypromice.core.qc.common import set_flag

DEG2RAD=np.pi/180

def correct_wind_speed(wspd: xr.DataArray, coefficient) -> xr.DataArray:
    """Correct wind speed with a linear correction coefficient. For example, the conversion from a standard
    Young anemometer to an Arctic Young anemometer is 1.7.

    Parameters
    ----------
    wspd : xr.DataArray
        Wind speed
    coefficient : float
        Correction coefficient

    Returns
    -------
    xr.DataArray
        Corrected wind speed
    """
    return wspd * coefficient


def filter_wind_direction(wdir: xr.DataArray,
                          wspd: xr.DataArray,
                          wdir_qc: xr.DataArray=None
) -> xr.DataArray:
    """Flag wind direction samples where wind speed is zero.

    Wind direction is physically undefined when wind speed equals zero.
    This function identifies such cases and assigns a QC flag to the
    corresponding wind direction variable using `set_flag`.

    Parameters
    ----------
    wdir : xr.DataArray
        Wind direction
    wspd: xr.DataArray
        Wind speed
    wdir_qc : xr.DataArray
        Quality control for wind direction

    Returns
    -------
    wdir_qc : xr.DataArray
        Updated quality control flags for wind direction
    """
    mask = wspd == 0
    wdir_qc = set_flag(wdir,
                       "ZERO_WSPD",
                       mask=mask,
                       qc=wdir_qc)
    return wdir_qc


def calculate_directional_wind_speed(wspd: xr.DataArray,
                                     wdir: xr.DataArray
) -> xr.DataArray:
    """Calculate directional wind speed from wind speed and direction

    Parameters
    ----------
    wspd : xr.DataArray
        Wind speed data array
    wdir : xr.DataArray
        Wind direction data array
    deg2rad : float
        Degree to radians coefficient. The default is np.pi/180

    Returns
    -------
    wspd_x : xr.DataArray
        Wind speed in X direction
    wspd_y : xr.DatArray
        Wind speed in Y direction
    """
    wspd_x = wspd * np.sin(wdir * DEG2RAD)
    wspd_y = wspd * np.cos(wdir * DEG2RAD)
    return wspd_x, wspd_y