__all__=["clip_and_interpolate", "get_cloud_coefficients"]

import pandas as pd
import xarray as xr

T_0=273.15                  # degrees Celsius to Kelvin conversion
eps_overcast = 1.0          # Clouds overcast default coefficient
eps_clear = 9.36508e-6      # Clouds clear default coefficient

def clip_and_interpolate(temp : xr.DataArray,
                         lo : float,
                         hi : float,
                         max_interp : pd.Timedelta = pd.Timedelta(12,'h')
) -> xr.DataArray:
    """Clip and interpolate temperature dataset for use in
    corrections

    Parameters
    ----------
    temp : `xr.DataArray`
        Array of temperature data
    lo : float
        Minimum threshold value for clipping
    hi : float
        Maximum threshold value for clipping
    max_interp : `pd.Timedelta`
        Maximum time steps to interpolate across.
        The default is 12 hours.

    Returns
    -------
    temp_interp : `xr.DataArray`
        Array of interpolated temperature data
    """
    # Clip values to high and low threshold values
    temp = temp.where((temp >= lo) & (temp <= hi))

    # Drop duplicates and interpolate across NaN values
    temp_interp = temp.interpolate_na(dim='time',
                                      max_gap=max_interp)

    return temp_interp


def get_cloud_coefficients(temp: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Get overcast and clear cloud longwave coefficients using
    air temperature, based on assumptions from Swinbank (1963)

    Parameters
    ----------
    temp : xr.DataArray
        Air temperature

    Returns
    -------
    LR_overcast : xr.DataArray
        Overcast cloud coefficients, using overcast cloud assumption from Swinbank (1963)
    LR_clear : xr.DataArray
        Clear cloud coefficients, using clear cloud assumption, from Swinbank (1963)
    """
    LR_overcast = eps_overcast * 5.67e-8 * (temp + T_0) ** 4
    LR_clear = eps_clear * 5.67e-8 * (temp + T_0) ** 6
    return LR_overcast, LR_clear