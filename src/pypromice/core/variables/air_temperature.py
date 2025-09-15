__all__=["clip_and_interpolate"]

import pandas as pd
import xarray as xr

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