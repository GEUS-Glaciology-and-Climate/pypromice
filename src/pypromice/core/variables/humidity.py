__all__ = ["adjust", "convert", "calculate_specific_humidity"]

import xarray as xr
import numpy as np

# Define constants
T_0 = 273.15                # Ice point temperature (Kelvins)
T_100 = T_0+100             # Steam point temperature (Kelvins)
ews = 1013.246              # Saturation vapour pressure at steam point temperature (normal atmosphere) (hPa)
ei0 = 6.1071                # Saturation vapour pressure at ice melting point temperature (normal atmosphere) (hPa)
eps=0.622                   # Ratio of molar masses of vapor and dry air

def adjust(rh: xr.DataArray,
           t: xr.DataArray
) -> xr.DataArray:
    """Correct relative humidity so that values are given with respect to
    saturation over ice in subfreezing conditions, and with respect to
    saturation over water (as given by the instrument) above the melting
    point temperature. Saturation water vapors are calculated after
    Groff & Gratch method.

    Parameters
    ----------
    rh : xr.DataArray
        Relative humidity
    t : xr.DataArray
        Air temperature

    Returns
    -------
    rh_wrt_ice_or_water : xr.DataArray
        CAdjusted relative humidity
    """
    # Convert to hPa (Groff & Gratch)
    e_s_wtr = 10**(-7.90298 * (T_100 / (t + T_0) - 1)
                   + 5.02808 * np.log10(T_100 / (t + T_0))
                   - 1.3816E-7 * (10**(11.344 * (1 - (t + T_0) / T_100)) - 1)
                   + 8.1328E-3 * (10**(-3.49149 * (T_100/(t + T_0) - 1)) -1)
                   + np.log10(ews))
    e_s_ice = 10**(-9.09718 * (T_0 / (t + T_0) - 1)
                   - 3.56654 * np.log10(T_0 / (t + T_0))
                   + 0.876793 * (1 - (t + T_0) / T_0)
                   + np.log10(ei0))

    # Define freezing point. Why > -100?
    nan_mask = t.notnull()
    freezing = (t < 0) & (t > -100) & nan_mask

    # Set to Groff & Gratch values when freezing, otherwise just rh
    rh_wrt_ice_or_water = rh.where(~freezing & nan_mask,
                                   other=rh*(e_s_wtr/e_s_ice))
    return rh_wrt_ice_or_water


def calculate_specific_humidity(t, p, rh_wrt_ice_or_water):
    """Calculate specific humidity.

    Parameters
    ----------
    t : xr.DataArray
        Air temperature
    p : xr.DataArray
        Air pressure
    rh_wrt_ice_or_water : xr.DataArray
        Adjusted relative humidity

    Returns
    -------
    xr.DataArray
        Specific humidity (kg/kg)
    """
    # Saturation vapour pressure above 0 C (hPa)
    es_wtr = 10**(-7.90298 * (T_100 / (t + T_0) - 1) + 5.02808 * np.log10(T_100 / (t + T_0))
                  - 1.3816E-7 * (10**(11.344 * (1 - (t + T_0) / T_100)) - 1)
                  + 8.1328E-3 * (10**(-3.49149 * (T_100 / (t + T_0) -1)) - 1) + np.log10(ews))

    # Saturation vapour pressure below 0 C (hPa)
    es_ice = 10**(-9.09718 * (T_0 / (t + T_0) - 1) - 3.56654
                  * np.log10(T_0 / (t + T_0)) + 0.876793
                  * (1 - (t + T_0) / T_0)
                  + np.log10(ei0))

    # Specific humidity at saturation (incorrect below melting point)
    q_sat = eps * es_wtr / (p - (1 - eps) * es_wtr)

    # Replace saturation specific humidity values below melting point
    freezing = t < 0
    q_sat[freezing] = eps * es_ice[freezing] / (p[freezing] - (1 - eps) * es_ice[freezing])

    # Mask where temperature or pressure are null values
    q_nan = np.isnan(t) | np.isnan(p)
    q_sat[q_nan] = np.nan

    # Convert to kg/kg
    return rh_wrt_ice_or_water * q_sat / 100

def convert(qh: xr.DataArray
) -> xr.DataArray:
    """Convert specific humidity from kg/kg to g/kg units

    Parameters
    ----------
    qh : xr.DataArray
        Specific humidity (kg/kg)

    Returns
    -------
    xr.DataArray
        Specific humidity (g/kg)
    """
    return 1000 * qh
