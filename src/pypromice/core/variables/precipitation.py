__all__ = ["convert_to_rate_and_correct_undercatch", "filter_lufft_errors", "get_rate"]

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


def convert_to_rate_and_correct_undercatch(
    precip: xr.DataArray, wspd: xr.DataArray, t: xr.DataArray
) -> xr.DataArray:
    """Correct precipitation with the undercatch correction method used in
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
    precip : xr.DataArray
        Cumulative precipitation measurements
    wspd : xr.DataArray
        Wind speed measurements
    t : xr.DataArray
        Air temperature measurements

    Returns
    -------
    precip_rate : xr.DataArray
        Corrected precipitation rate
    """
    precip_rate = get_rate(precip)

    # Calculate undercatch correction factor
    corr = 100 / (100.00 - 4.37 * wspd + 0.35 * wspd * wspd)

    # Fix all values below 1.02 to 1.02
    corr = corr.where(corr > 1.02, other=1.02)

    # Apply correction to rate
    precip_rate = precip_rate * corr

    # Removing all negative precipitation rates
    precip_rate = precip_rate.where(precip_rate > 0)

    # Filtering cold season precipitation measurements
    rain_in_cold = (precip_rate > 0) & (t < -2)
    precip_rate = precip_rate.where(~rain_in_cold)

    return precip_rate


def get_rate(
    precip: xr.DataArray,
) -> xr.DataArray:
    """
    Calculates the precipitation rate by differentiating the precipitation values
    with respect to time while considering the time step size to ensure the rate
    is in the units of mm/hr. It also handles missing values by forward-filling
    and removes results at locations where calculation depends on interpolated
    values.

    Parameters
    ----------
    precip : xarray.DataArray
        The precipitation data with a time dimension. It includes timestamped
        precipitation values.

    Returns
    -------
    xarray.DataArray
        The calculated precipitation rate with the same dimensions as the input
        data. Missing values are ignored in the rate calculation and replaced
        appropriately.
    """
    nan_mask = precip.isnull()

    # Calculate precipitation rate and makes sure it remains of the same size
    # and taking into account the time step size to ensure mm/hr
    dt_hours = precip["time"].diff("time").reindex_like(precip) / np.timedelta64(1, "h")
    precip_rate = precip.diff("time").reindex_like(precip) / dt_hours
    # Removing timestamps where precipitation rates have been calculated over
    # interpolated values
    precip_rate = precip_rate.where(~nan_mask)
    return precip_rate
