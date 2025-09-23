__all__ = ["convert_sr", "convert_lr", "filter_lr", "filter_sr",
           "correct_sr", "calculate_albedo", "calculate_surface_temperature",
           "calculate_cloud_coverage", "calculate_TOA"]

import xarray as xr
import numpy as np

# Define coefficients for radiometer adjustments
T_0=273.15                  # degrees Celsius to Kelvin conversion
deg2rad = np.pi / 180       # Degrees to radians conversion
emissivity=0.97

def convert_sr(sr: xr.DataArray,
               sr_eng_coef: float) -> xr.DataArray:
    """Convert shortwave radiation measurements from engineering to
    physical units, using a calibration coefficient (defined by
    manufacturers usually)

    Parameters
    ----------
    sr : xr.DataArray
        Shortwave radiation (upwelling or downwelling) measurements
    sr_eng_coef : float
        Shortwave engineering calibration coefficient

    Returns
    -------
    xr.DataArray
        Converted shortwave measurements
    """
    return (sr * 10) / sr_eng_coef

def convert_lr(lr: xr.DataArray,
               t_rad: xr.DataArray,
               lr_eng_coef: float) -> xr.DataArray:
    """Convert longwave radiation measurements from engineering to
    physical units, using the reported radiometer temperature and
    a calibration coefficient (defined by manufacturers usually)

    Parameters
    ----------
    lr : xr.DataArray
        Longwave radiation (upwelling or downwelling) measurements
    t_rad : xr.DataArray
        Radiometer temperature
    lr_eng_coef : float
        Longwave engineering calibration coefficient

    Returns
    -------
    xr.DataArray
        Converted shortwave measurements
    """
    return ((lr * 10) / lr_eng_coef) + 5.67e-8 * (t_rad + T_0) **4

def filter_lr(lr: xr.DataArray,
           t_rad: xr.DataArray) -> xr.DataArray:
    """Remove longwave radiation measurements that are missing
    simultaneous radiometer temperature measurements

    Parameters
    ----------
    lr : xr.DataArray
        Longwave radiation measurements (upwelling or downwelling)
    t_rad : xr.DataArray
        Radiometer temperature

    Returns
    -------
    xr.DataArray
        Filtered radiation measurements
    """
    return lr.where(t_rad.notnull())

def filter_sr(dsr: xr.DataArray,
              usr: xr.DataArray,
              cc : xr.DataArray,
              ZenithAngle_rad: xr.DataArray,
              ZenithAngle_deg: xr.DataArray,
              AngleDif_deg: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, tuple]:
    """Filter shortwave radiation data for tilt, station pose relative to sun position, and top-of-atmosphere (TOA)
    irradiance

    Parameters
    ----------
    dsr : xr.DataArray
        Downwelling shortwave radiation
    usr : xr.DataArray
        Upwelling shortwave radiation
    cc : xr.DataArray
        Cloud cover
    ZenithAngle_deg : xr.DataArray
        Zenith angle in degrees
    ZenithAngle_rad : xr.DataArray
        Zenith angle in radians
     AngleDif_deg : xr.DataArray
        Angle between sun and sensor in degrees

    Returns
    -------
    dsr_filtered : xr.DataArray
        Filtered downwelling shortwave radiation
    usr_filtered : xr. DataArray
        Filtered upwelling shortwave radiation
    tuple
        Filter flags for 1) sun position below horizon; 2) sun on lower dome; 3) downwelling sr measurements greater
        than TOA; and 4) upwelling sr measurements greater than TOA
    """
    dsr_filtered = dsr.copy()
    usr_filtered = usr.copy()

    # Setting to zero when sun below the horizon.
    bad = ZenithAngle_deg > 95
    dsr_filtered[bad & dsr_filtered.notnull()] = 0
    usr_filtered[bad & usr_filtered.notnull()] = 0

    # Setting to zero when values are negative
    dsr_filtered = dsr_filtered.clip(min=0)
    usr_filtered = usr_filtered.clip(min=0)

    # Filtering usr and dsr for sun on lower dome
    # in theory, this is not a problem in cloudy conditions, but the cloud cover
    # index is too uncertain at this point to be used
    sunonlowerdome = (AngleDif_deg >= 90) & (ZenithAngle_deg <= 90)

    # Relaxing the filter for cases where sensor tilt is unknown
    mask = ~sunonlowerdome | AngleDif_deg.isnull()

    # Perform filter
    dsr_filtered = dsr_filtered.where(mask)
    usr_filtered = usr_filtered.where(mask)

    # Calculate TOA shortwave radiation
    isr_toa = calculate_TOA(ZenithAngle_deg, ZenithAngle_rad)

    # Filter dsr values that are greater than top of the atmosphere irradiance
    # Case where no tilt is available. If it is, then the same filter is used
    # after tilt correction.
    tilt_correction_possible = AngleDif_deg.notnull() & cc.notnull()
    TOA_crit_nopass_dsr = ~tilt_correction_possible & (dsr_filtered > (1.2 * isr_toa + 150))
    dsr_filtered[TOA_crit_nopass_dsr] = np.nan

    # The upward flux should not be higher than the TOA downward flux
    TOA_crit_nopass_usr = (usr_filtered > 0.8 * (1.2 * isr_toa + 150))
    usr_filtered[TOA_crit_nopass_usr] = np.nan

    return dsr_filtered, usr_filtered, (bad, sunonlowerdome, TOA_crit_nopass_dsr, TOA_crit_nopass_usr)

def correct_sr(dsr_filtered: xr.DataArray,
               usr_filtered: xr.DataArray,
               cc: xr.DataArray,
               phi_sensor_rad : xr.DataArray,
               theta_sensor_rad : xr.DataArray,
               lat: float,
               Declination_rad : xr.DataArray,
               HourAngle_rad : xr.DataArray,
               ZenithAngle_rad : xr.DataArray,
               ZenithAngle_deg : xr.DataArray,
               AngleDif_deg: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, tuple]:
    """Correct shortwave radiation data for station tilt and top-of-atmosphere (TOA) irradiance

    Parameters
    ----------
    dsr_filtered : xr.DataArray
        Downwelling shortwave radiation (filtered for tilt)
    usr_filtered : xr.DataArray
        Upwelling shortwave radiation (filtered for tilt)
    cc : xr.DataArray
        Cloud cover
    phi_sensor_rad : xr.DataArray
        Spherical tilt coordinates
    theta_sensor_rad : xr.DataArray
        Total tilt of sensor, where 0 is horizontal
    lat : float
        Station latitude
    Declination_rad : xr.DataArray
        Sun declination
    HourAngle_rad : xr.DataArray
        Hour angle of sun
    ZenithAngle_rad : xr.DataArray
        Zenith angle in radians
    ZenithAngle_deg : xr.DataArray
        Zenith angle in degrees
    AngleDif_deg : xr.DataArray
        Angle between sun and sensor in degree

    Returns
    -------
    dsr_cor : xr.DataArray
        Corrected downwelling shortwave radiation
    usr_cor : xr.DataArray
        Corrected upwelling shortwave radiation
    TOA_crit_nopass_cor : xr.DataArray
        Correction flags for invalid TOA values
    """
    # Diffuse to direct irradiance fraction
    DifFrac = 0.2 + 0.8 * cc
    CorFac_all = calculate_correction_factor(phi_sensor_rad,
                                             theta_sensor_rad,
                                             Declination_rad,
                                             HourAngle_rad,
                                             ZenithAngle_rad,
                                             ZenithAngle_deg,
                                             lat,
                                             DifFrac)

    tilt_correction_possible = AngleDif_deg.notnull() & cc.notnull()
    CorFac_all = CorFac_all.where(tilt_correction_possible)

    # Apply correction to downwelling shortwave radiation and then mask upwelling values
    dsr_cor = dsr_filtered * CorFac_all
    usr_cor = usr_filtered.where(dsr_cor.notnull())

    # Calculate TOA shortwave radiation
    isr_toa = calculate_TOA(ZenithAngle_deg, ZenithAngle_rad)

    # Remove data where TOA shortwave radiation invalid
    TOA_crit_nopass_cor = dsr_cor > (1.2 * isr_toa + 150)
    dsr_cor[TOA_crit_nopass_cor] = np.nan
    usr_cor[TOA_crit_nopass_cor] = np.nan

    return dsr_cor, usr_cor, TOA_crit_nopass_cor


def calculate_albedo(dsr_filtered: xr.DataArray,
                     usr_filtered: xr.DataArray,
                     dsr_cor: xr.DataArray,
                     cc: xr.DataArray,
                     ZenithAngle_deg: xr.DataArray,
                     AngleDif_deg: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate surface albedo based on upwelling and downwelling shortwave
    flux, the angle between the sun and sensor, and the sun zenith angle.

    Parameters
    ----------
    dsr_filtered : xr.DataArray
        Downwelling shortwave radiation
    usr_filtered : xr.DataArray
        Upwelling shortwave radiation
    dsr_cor : xr.DataArray
        Corrected downwelling shortwave radiation
    cc : xr.DataArray
        Cloud cover
    ZenithAngle_deg : xr.DataArray
        Sun zenith angle in degrees.
    AngleDif_deg : xr.DataArray
        Angle between the sun and the sensor in degrees

    Returns
    -------
    albedo : xr.DataArray
        Calculated albedo
    OKalbedos : xr.DataArray
        Boolean mask indicating valid albedo values
    """
    tilt_correction_possible = AngleDif_deg.notnull() & cc.notnull()

    albedo = xr.where(tilt_correction_possible,
                      usr_filtered / dsr_cor,
                      usr_filtered / dsr_filtered)

    OOL = (albedo  >= 1) | (albedo  <= 0)
    good_zenith_angle = ZenithAngle_deg < 70
    good_relative_zenith_angle = (AngleDif_deg < 70) | (AngleDif_deg.isnull())
    OKalbedos = good_relative_zenith_angle & good_zenith_angle & ~OOL
    albedo = albedo.where(OKalbedos)
    return albedo, OKalbedos


def calculate_TOA(ZenithAngle_deg: xr.DataArray,
                  ZenithAngle_rad: xr.DataArray
) -> xr.DataArray:
    """Calculate incoming shortwave radiation at the top of the atmosphere,
    accounting for sunset periods

    Parameters
    ----------
    ZenithAngle_deg : xr.DataArray
        Zenith angle in degrees
    ZenithAngle_rad : xr.DataArray
        Zenith angle in radians

    Returns
    -------
    isr_toa : float
        Incoming shortwave radiation at the top of the atmosphere
    """
    sundown = ZenithAngle_deg >= 90

    # Incoming shortware radiation at the top of the atmosphere
    isr_toa = 1372 * np.cos(ZenithAngle_rad)
    isr_toa[sundown] = 0
    return isr_toa


def calculate_correction_factor(phi_sensor_rad: xr.DataArray,
                                theta_sensor_rad: xr.DataArray,
                                Declination_rad: xr.DataArray,
                                HourAngle_rad: xr.DataArray,
                                ZenithAngle_rad: xr.DataArray,
                                ZenithAngle_deg: xr.DataArray,
                                lat: float,
                                DifFrac: xr.DataArray
) -> xr.DataArray:
    """Calculate radiometer correction factor for direct beam radiation, as described
    here: http://solardat.uoregon.edu/SolarRadiationBasics.html

    Offset correction (where solar zenith angles are larger than 110 degrees) not
    implemented as it should not improve the accuracy of well-calibrated
    instruments.

    It would go something like this:
    ds['dsr'] = ds['dsr'] - ds['dwr_offset']
    SRout = SRout - SRout_offset

    Parameters
    ----------
    Declination_rad : float
        Declination in radians
    phi_sensor_rad : xr.DataArray
        Spherical tilt coordinates
    theta_sensor_rad : xr.DataArray
        Total tilt of sensor, where 0 is horizontal
    HourAngle_rad : float
        Sun hour angle in radians
    ZenithAngle_rad : float
        Zenith angle in radians
    ZenithAngle_deg : float
        Zenith Angle in degrees
    lat :  float
        Latitude
    DifFrac : xr.DataArray
        Fractional cloud cover

    Returns
    -------
    CorFac_all : xr.DataArray
        Correction factor
    """
    CorFac = np.sin(Declination_rad) * np.sin(lat * deg2rad) \
        * np.cos(theta_sensor_rad) \
        - np.sin(Declination_rad) \
        * np.cos(lat * deg2rad) \
        * np.sin(theta_sensor_rad) \
        * np.cos(phi_sensor_rad + np.pi) \
        + np.cos(Declination_rad) \
        * np.cos(lat * deg2rad) \
        * np.cos(theta_sensor_rad) \
        * np.cos(HourAngle_rad) \
        + np.cos(Declination_rad) \
        * np.sin(lat * deg2rad) \
        * np.sin(theta_sensor_rad) \
        * np.cos(phi_sensor_rad + np.pi) \
        * np.cos(HourAngle_rad) \
        + np.cos(Declination_rad) \
        * np.sin(theta_sensor_rad) \
        * np.sin(phi_sensor_rad + np.pi) \
        * np.sin(HourAngle_rad) \

    CorFac = np.cos(ZenithAngle_rad) / CorFac

    # Sun out of field of view upper sensor
    CorFac[(CorFac < 0) | (ZenithAngle_deg > 90)] = 1

    # Calculating ds['dsr'] over a horizontal surface corrected for station/sensor tilt
    CorFac_all = CorFac / (1 - DifFrac + CorFac * DifFrac)

    return CorFac_all.where(theta_sensor_rad.notnull())


def calculate_cloud_coverage(dlr: xr.DataArray,
                             LR_overcast: xr.DataArray,
                             LR_clear: xr.DataArray
) -> xr.DataArray:
    """Calculate cloud cover using downwelling longwave radiation and the
    overcast and clear cloud assumptions from Swinbank (1963) which are
    derived from air temperature.

    Parameters
    ----------
    dlr : xr.DataArray
        Downwelling longwave radiation, with array of same length as T and T_0
    LR_overcast : xr.DataArray
        Cloud overcast assumption, from Swinbank (1963)
    LR_clear : xr.DataArray
        Cloud clear assumption, from Swinbank (1963)

    Returns
    -------
    cc : xr.DataArray
        Cloud cover data array
    """
    cc = (dlr - LR_clear) / (LR_overcast - LR_clear)
    cc[cc > 1] = 1
    cc[cc < 0] = 0
    return cc


def calculate_surface_temperature(dlr: xr.DataArray,
                                  ulr: xr.DataArray
) -> xr.DataArray:
    """Calculate surface temperature from downwelling and
    upwelling longwave radiation.

    Parameters
    ----------
    dlr : xr.DataArray
        Downwelling longwave radiation
    ulr : xr.DataArray
        Upwelling longwave radiation

    Returns
    -------
    xr.DataArray
        Calculated surface temperature
    """
    t_surf = ((ulr - (1 - emissivity) * dlr) / emissivity / 5.67e-8)**0.25 - T_0
    return t_surf