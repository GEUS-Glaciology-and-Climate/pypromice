__all__ = ["adjust", "adjust_and_include_uncorrected_values"]

import numpy as np
import xarray as xr

T_0=273.15                  # degrees Celsius to Kelvin conversion

def adjust(z_boom: xr.DataArray,
           air_temperature: xr.DataArray
) -> xr.DataArray:
    """Adjust sonic ranger readings for sensitivity to air temperature

    Parameters
    ----------
    z_boom : xr.DataArray
        Station boom height from sonic ranger
    air_temperature : xr.DataArray
        Air temperature

    Returns
    -------
    xr.DataArray
        Adjusted station boom height
    """
    return z_boom * ((air_temperature + T_0)/T_0)**0.5


def adjust_and_include_uncorrected_values(z_boom: xr.DataArray,
           air_temperature: xr.DataArray
) -> xr.DataArray:
    """Adjust sonic ranger readings for sensitivity to air temperature,
    and retain uncorrected values where air temperature measurements
    are not available.

    Parameters
    ----------
    z_boom : xr.DataArray
        Station boom height from sonic ranger
    air_temperature : xr.DataArray
        Air temperature

    Returns
    -------
    xr.DataArray
        Adjusted station boom height
    """
    return xr.where(air_temperature.notnull(),
                    z_boom * ((air_temperature + T_0)/T_0)**0.5,
                    z_boom)