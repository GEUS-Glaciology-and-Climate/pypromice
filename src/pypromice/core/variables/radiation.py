
import xarray as xr
import numpy as np
from pypromice.core.variables import station_pose

# Define air temperature for radiometer adjustments
T_0=273.15                  # radiometer air temperature adjustment
deg2rad = np.pi / 180       # Degrees to radians conversion
rad2deg = 1 / deg2rad       # Radians to degrees conversion

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
    T_0 : float

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

def correct_sr(ds):
    """
    Processes shortwave radiation data from a dataset by applying tilt and sun
    angle corrections.

    Parameters:
        ds (xarray.Dataset): Dataset containing variables such as time, tilt_x,
                tilt_y, dsr (downwelling SW radiation), usr (upwelling SW radiation),
                cloud cover (cc), gps_lat, gps_lon, and optional attributes
                 latitude and longitude.

    Returns:
        ds (xarray.Dataset): Updated dataset with corrected downwelling ('dsr_cor')
                and upwelling ('usr_cor') SW radiation, and derived surface albedo ('albedo').
        tuple: A tuple containing masks and calculated TOA radiation:
               (OKalbedos, sunonlowerdome, bad, isr_toa, TOA_crit_nopass)
    """
    # Determine station position relative to sun
    doy = ds['time'].to_dataframe().index.dayofyear.values                     # Gather variables to calculate sun pos
    hour = ds['time'].to_dataframe().index.hour.values
    minute = ds['time'].to_dataframe().index.minute.values

    if hasattr(ds, 'latitude') and hasattr(ds, 'longitude'):
        lat = ds.attrs['latitude']                                             # TODO Why is mean GPS lat lon not preferred for calcs?
        lon = ds.attrs['longitude']
    else:
        lat = ds['gps_lat'].mean()
        lon = ds['gps_lon'].mean()

    phi_sensor_rad, theta_sensor_rad = station_pose.calculate_tilt(ds['tilt_x'], ds['tilt_y'])

    Declination_rad = station_pose.calculate_declination(doy, hour, minute)

    HourAngle_rad = station_pose.calculate_hour_angle(hour, minute, lon)

    ZenithAngle_rad, ZenithAngle_deg = station_pose.calculate_zenith(lat,
                                                        Declination_rad,
                                                        HourAngle_rad)

    # Setting to zero when sun below the horizon.
    bad = ZenithAngle_deg > 95
    ds['dsr'][bad & ds['dsr'].notnull()] = 0
    ds['usr'][bad & ds['usr'].notnull()] = 0

    # Setting to zero when values are negative
    ds['dsr'] = ds['dsr'].clip(min=0)
    ds['usr'] = ds['usr'].clip(min=0)

    # Calculate angle between sun and sensor
    AngleDif_deg = station_pose.calculate_angle_difference(ZenithAngle_rad, HourAngle_rad,
                                 phi_sensor_rad, theta_sensor_rad)
    tilt_correction_possible = AngleDif_deg.notnull() & ds['cc'].notnull()

    # Filtering usr and dsr for sun on lower dome
    # in theory, this is not a problem in cloudy conditions, but the cloud cover
    # index is too uncertain at this point to be used
    sunonlowerdome = (AngleDif_deg >= 90) & (ZenithAngle_deg <= 90)
    mask = ~sunonlowerdome | AngleDif_deg.isnull()                             # relaxing the filter for cases where sensor tilt is unknown
    ds['dsr'] = ds['dsr'].where(mask)
    ds['usr'] = ds['usr'].where(mask)

    # Filter dsr values that are greater than top of the atmosphere irradiance
    # Case where no tilt is available. If it is, then the same filter is used
    # after tilt correction.
    isr_toa = calculate_TOA(ZenithAngle_deg, ZenithAngle_rad)                        # Calculate TOA shortwave radiation
    TOA_crit_nopass = ~tilt_correction_possible & (ds['dsr'] > (1.2 * isr_toa + 150))
    ds['dsr'][TOA_crit_nopass] = np.nan

    # the upward flux should not be higher than the TOA downard flux
    TOA_crit_nopass_usr = (ds['usr'] > 0.8*(1.2 * isr_toa + 150))
    ds['usr'][TOA_crit_nopass_usr] = np.nan

    # Diffuse to direct irradiance fraction
    DifFrac = 0.2 + 0.8 * ds['cc']
    CorFac_all = calculate_correction_factor(Declination_rad,
                                             phi_sensor_rad,
                                             theta_sensor_rad,
                                             HourAngle_rad,
                                             ZenithAngle_rad,
                                             ZenithAngle_deg,
                                             lat,
                                             DifFrac)
    CorFac_all = CorFac_all.where(tilt_correction_possible)

    # Correct Downwelling shortwave radiation
    ds['dsr_cor'] = ds['dsr'].copy() * CorFac_all
    ds['usr_cor'] = ds['usr'].copy().where(ds['dsr_cor'].notnull())

    # Remove data where TOA shortwave radiation invalid
    # this can only be done after correcting for tilt
    TOA_crit_nopass_cor = ds['dsr_cor'] > (1.2 * isr_toa + 150)
    ds['dsr_cor'][TOA_crit_nopass_cor] = np.nan
    ds['usr_cor'][TOA_crit_nopass_cor] = np.nan

    ds, OKalbedos = calculate_albedo(ds['usr'], ds['dsr'], ds['dsr_cor'], ds['cc'], AngleDif_deg, ZenithAngle_deg)

    return ds, (OKalbedos, sunonlowerdome, bad, isr_toa, TOA_crit_nopass_cor, TOA_crit_nopass, TOA_crit_nopass_usr)


def calculate_albedo(usr: xr.DataArray,
                     dsr: xr.DataArray,
                     dsr_cor: xr.DataArray,
                     cc: xr.DataArray,
                     AngleDif_deg: xr.DataArray,
                     ZenithAngle_deg: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    '''
    Calculate surface albedo based on upwelling and downwelling shortwave
    flux, the angle between the sun and sensor, and the sun zenith angle.

    Parameters
    ----------
    usr : xr.DataArray
        Upwelling shortwave radiation
    dsr : xr.DataArray
        Downwelling shortwave radiation
    dsr_cor : xr.DataArray
        Corrected downwelling shortwave radiation
    AngleDif_deg : xr.DataArray
        Angle between the sun and the sensor in degrees.
    ZenithAngle_deg : xr.DataArray
        Sun zenith angle in degrees.

    Returns
    -------
    albedo : xr.DataArray
        Calculated albedo
    OKalbedos : xr.DataArray
        Boolean mask indicating valid albedo values.
    '''
    tilt_correction_possible = AngleDif_deg.notnull() & cc.notnull()

    albedo = xr.where(tilt_correction_possible,
                            usr / dsr_cor,
                            usr / dsr)

    OOL = (albedo  >= 1) | (albedo  <= 0)
    good_zenith_angle = ZenithAngle_deg < 70
    good_relative_zenith_angle = (AngleDif_deg < 70) | (AngleDif_deg.isnull())
    OKalbedos = good_relative_zenith_angle & good_zenith_angle & ~OOL
    albedo = albedo.where(OKalbedos)
    return albedo, OKalbedos


def calculate_TOA(ZenithAngle_deg: float,
                  ZenithAngle_rad: float
) -> float:
    '''Calculate incoming shortwave radiation at the top of the atmosphere,
    accounting for sunset periods

    Parameters
    ----------
    ZenithAngle_deg : float
        Zenith angle in degrees
    ZenithAngle_rad : float
        Zenith angle in radians

    Returns
    -------
    isr_toa : float
        Incoming shortwave radiation at the top of the atmosphere
    '''
    sundown = ZenithAngle_deg >= 90

    # Incoming shortware radiation at the top of the atmosphere
    isr_toa = 1372 * np.cos(ZenithAngle_rad)
    isr_toa[sundown] = 0
    return isr_toa


def calculate_correction_factor(Declination_rad: float,
                                phi_sensor_rad: xr.DataArray,
                                theta_sensor_rad: xr.DataArray,
                                HourAngle_rad: float,
                                ZenithAngle_rad: float,
                                ZenithAngle_deg: float,
                                lat: float,
                                DifFrac: float
) -> xr.DataArray:
    '''Calculate radiometer correction factor for direct beam radiation, as described
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
    DifFrac : float
        Fractional cloud cover

    Returns
    -------
    CorFac_all : xr.DataArray
        Correction factor
    '''
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