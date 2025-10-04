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


def include_uncorrected_values(
    z_boom: xr.DataArray,
    z_boom_cor: xr.DataArray,
    air_temperature_1: xr.DataArray,
    air_temperature_2: xr.DataArray = None,
    t_rad: xr.DataArray = None,
    T_0: float = 273.15,
) -> xr.DataArray:
    """
    Adjust sonic ranger readings for sensitivity to air temperature and
    retain uncorrected values where temperature measurements are unavailable.

    Parameters
    ----------
    z_boom : xr.DataArray
        Uncorrected station boom height from sonic ranger
    z_boom_cor : xr.DataArray
        Boom height corrected with air_temperature_1
    air_temperature_1 : xr.DataArray
        Primary air temperature
    air_temperature_2 : xr.DataArray, optional
        Secondary air temperature
    t_rad : xr.DataArray, optional
        Radiative temperature
    T_0 : float, optional
        Reference temperature in Kelvin (default 273.15)

    Returns
    -------
    xr.DataArray
        Corrected boom height with fallback where needed
    """
    if air_temperature_2 is None:
        air_temperature_2 = xr.full_like(z_boom, np.nan)
    if t_rad is None:
        t_rad = xr.full_like(z_boom, np.nan)
    else:
        t_rad = t_rad.clip(max=0)

    z_boom_cor_w_t_rad = z_boom * ((t_rad + T_0) / T_0) ** 0.5
    z_boom_cor_w_ta2 = z_boom * ((air_temperature_2 + T_0) / T_0) ** 0.5
    z_boom_ta2_t_rad = xr.where(air_temperature_2.notnull(), z_boom_cor_w_ta2, z_boom_cor_w_t_rad)

    return xr.where(air_temperature_1.notnull(), z_boom_cor, z_boom_ta2_t_rad)
