__all__ = ["correct_rainfall_undercatch", "get_rainfall_per_timestep", "filter_lufft_errors"]

import numpy as np
import xarray as xr


def filter_lufft_errors(
    precip: xr.DataArray, t: xr.DataArray, p: xr.DataArray, rh: xr.DataArray
) -> xr.DataArray:
    """Filter precipitation measurements where air temperature, pressure, or
    relative humidity measurements are null values. This assumes that
    air temperature, air pressure, relative humidity and precipitation
    measurements are measured using the same instrument, e.g. a
    lufft instrument.

    Parameters
    ----------
    precip : xr.DataArray
        Cumulative precipitation measurements
    t : xr.DataArray
        Air temperature measurements
    p : xr.DataArray
        Air pressure measurements
    rh : xr.DataArray
        Relative humidity measurements

    Returns
    -------
    xr.DataArray
        Filtered precipitation values
    """
    mask = (t.isnull() | p.isnull() | rh.isnull()) & (precip == 0)
    return precip.where(~mask)


def correct_rainfall_undercatch(
    rainfall_per_timestep: xr.DataArray, wspd: xr.DataArray
) -> xr.DataArray:
    """Corrects rainfall amount per timestep for undercatch as in
    Yang et al. (1999) and Box et al. (2022), based on Goodison et al. (1998).

    Yang, D., Ishida, S., Goodison, B. E., and Gunther, T.: Bias correction of
    daily precipitation measurements for Greenland,
    https://doi.org/10.1029/1998jd200110, 1999.

    Box, J., Wehrle, A., van As, D., Fausto, R., Kjeldsen, K., Dachauer, A.,
    Ahlstrom, A. P., and Picard, G.: Greenland Ice Sheet rainfall, heat and
    albedo feedback imapacts from the Mid-August 2021 atmospheric river,
    Geophys. Res. Lett. 49 (11), e2021GL097356,
    https://doi.org/10.1029/2021GL097356, 2022.

    Goodison, B. E., Louie, P. Y. T., and Yang, D.: Solid Precipitation
    Measurement Intercomparison, WMO, 1998

    Parameters
    ----------
    rainfall : xr.DataArray
        Uncorrected rainfall per timestep
    wspd : xr.DataArray
        Wind speed measurements

    Returns
    -------
    rainfall_cor : xr.DataArray
        Corrected rainfall per timestep
    """

    # Calculate undercatch correction factor
    corr = 100 / (100.00 - 4.37 * wspd + 0.35 * wspd * wspd)

    # Fix all values below 1.02 to 1.02
    corr = corr.where(corr > 1.02, other=1.02)

    # Apply correction to rate
    rainfall_per_timestep_cor = rainfall_per_timestep * corr

    return rainfall_per_timestep_cor

def get_rainfall_per_timestep(
    precip: xr.DataArray,
    t: xr.DataArray
) -> xr.DataArray:
    """
    Derive rainfall per timestep from cumulative precipitation data.

    Parameters
    ----------
    precip : xr.DataArray
        Cumulative precipitation measurements.
    t : xr.DataArray
        Air temperature measurements.

    Returns
    -------
    xr.DataArray
        Rainfall per timestep with negative values removed and
        cold-season precipitation (T < -2 °C) filtered out.
    """
    rainfall_per_timestep = precip.diff("time").reindex_like(precip)

    # Removing all negative precipitation, both corrected and uncorrected
    rainfall_per_timestep = rainfall_per_timestep.where(rainfall_per_timestep >= 0)

    # Filtering cold season precipitation, both corrected and uncorrected
    rain_in_cold = (rainfall_per_timestep > 0) & (t < -2)
    rainfall_per_timestep = rainfall_per_timestep.where(~rain_in_cold)

    return rainfall_per_timestep
