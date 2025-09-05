
import xarray as xr
import numpy as np

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

    phi_sensor_rad, theta_sensor_rad = calculate_tilt(ds['tilt_x'],
                                                      ds['tilt_y'],    # Calculate station tilt
                                                      deg2rad)

    Declination_rad = calculate_declination(doy,
                                            hour,
                                            minute)
    HourAngle_rad = calculate_hour_angle(hour,
                                         minute,
                                         lon)
    ZenithAngle_rad, ZenithAngle_deg = calculate_zenith(lat,
                                                        Declination_rad,
                                                        HourAngle_rad,
                                                        deg2rad,
                                                        rad2deg)

    # Setting to zero when sun below the horizon.
    bad = ZenithAngle_deg > 95
    ds['dsr'][bad & ds['dsr'].notnull()] = 0
    ds['usr'][bad & ds['usr'].notnull()] = 0

    # Setting to zero when values are negative
    ds['dsr'] = ds['dsr'].clip(min=0)
    ds['usr'] = ds['usr'].clip(min=0)

    # Calculate angle between sun and sensor
    AngleDif_deg = calculate_angle_difference(ZenithAngle_rad, HourAngle_rad,
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
                                             DifFrac,
                                             deg2rad)
    CorFac_all = CorFac_all.where(tilt_correction_possible)

    # Correct Downwelling shortwave radiation
    ds['dsr_cor'] = ds['dsr'].copy() * CorFac_all
    ds['usr_cor'] = ds['usr'].copy().where(ds['dsr_cor'].notnull())

    # Remove data where TOA shortwave radiation invalid
    # this can only be done after correcting for tilt
    TOA_crit_nopass_cor = ds['dsr_cor'] > (1.2 * isr_toa + 150)
    ds['dsr_cor'][TOA_crit_nopass_cor] = np.nan
    ds['usr_cor'][TOA_crit_nopass_cor] = np.nan

    ds, OKalbedos = calculate_albedo(ds, AngleDif_deg, ZenithAngle_deg)

    return ds, (OKalbedos, sunonlowerdome, bad, isr_toa, TOA_crit_nopass_cor, TOA_crit_nopass, TOA_crit_nopass_usr)

def calculate_tilt(tilt_x, tilt_y, deg2rad):
    '''Calculate station tilt

    Parameters
    ----------
    tilt_x : xarray.DataArray
        X tilt inclinometer measurements
    tilt_y : xarray.DataArray
        Y tilt inclinometer measurements
    deg2rad : float
        Degrees to radians conversion

    Returns
    -------
    phi_sensor_rad : xarray.DataArray
        Spherical tilt coordinates
    theta_sensor_rad : xarray.DataArray
        Total tilt of sensor, where 0 is horizontal
    '''
    # Tilt as radians
    tx = tilt_x * deg2rad
    ty = tilt_y * deg2rad

    # Calculate cartesian coordinates
    X = np.sin(tx) * np.cos(tx) * np.sin(ty)**2 + np.sin(tx) * np.cos(ty)**2
    Y = np.sin(ty) * np.cos(ty) * np.sin(tx)**2 + np.sin(ty) * np.cos(tx)**2
    Z = np.cos(tx) * np.cos(ty) + np.sin(tx)**2 * np.sin(ty)**2

    # Calculate spherical coordinates
    phi_sensor_rad = -np.pi /2 - np.arctan(Y/X)
    phi_sensor_rad[X > 0] += np.pi
    phi_sensor_rad[(X == 0) & (Y < 0)] = np.pi
    phi_sensor_rad[(X == 0) & (Y == 0)] = 0
    phi_sensor_rad[phi_sensor_rad < 0] += 2*np.pi

    # Total tilt of the sensor, i.e. 0 when horizontal
    theta_sensor_rad = np.arccos(Z / (X**2 + Y**2 + Z**2)**0.5)
    return phi_sensor_rad, theta_sensor_rad

def calculate_declination(doy, hour, minute):
    '''Calculate sun declination based on time

    Parameters
    ----------
    doy : int
        Day of year
    hour : int
        Hour of day
    minute : int
        Minute of hour

    Returns
    -------
    float
        Sun declination
    '''
    d0_rad = 2 * np.pi * (doy + (hour + minute / 60) / 24 -1) / 365
    return np.arcsin(0.006918 - 0.399912
                     * np.cos(d0_rad) + 0.070257
                     * np.sin(d0_rad) - 0.006758
                     * np.cos(2 * d0_rad) + 0.000907
                     * np.sin(2 * d0_rad) - 0.002697
                     * np.cos(3 * d0_rad) + 0.00148
                     * np.sin(3 * d0_rad))

def calculate_hour_angle(hour, minute, lon):
    '''Calculate hour angle of sun based on time and longitude. Make sure that
    time is set to UTC and longitude is positive when west. Hour angle should
    be 0 at noon

    Parameters
    ----------
    hour : int
        Hour of day
    minute : int
        Minute of hour
    lon : float
        Longitude

    Returns
    -------
    float
        Hour angle of sun
    '''
    return 2 * np.pi * (((hour + minute / 60) / 24 - 0.5) - lon/360)
     # ; - 15.*timezone/360.)


def calculate_sun_direction_degrees(HourAngle_rad):                                          #TODO remove if not plan to use this
    '''Calculate sun direction as degrees. This is an alternative to
    _calcHourAngle that is currently not implemented into the offical L0>>L3
    workflow. Here, 180 degrees is at noon (NH), as opposed to HourAngle

    Parameters
    ----------
    HourAngle_rad : float
        Sun hour angle in radians

    Returns
    -------
    DirectionSun_deg
        Sun direction in degrees
    '''
    DirectionSun_deg = HourAngle_rad * 180/np.pi - 180
    DirectionSun_deg[DirectionSun_deg < 0] += 360
    DirectionSun_deg[DirectionSun_deg < 0] += 360
    return DirectionSun_deg

def calculate_zenith(lat, Declination_rad, HourAngle_rad, deg2rad, rad2deg):
    '''Calculate sun zenith in radians and degrees

    Parameters
    ----------
    lat : float
        Latitude
    Declination_Rad : float
        Sun declination in radians
    HourAngle_rad : float
        Sun hour angle in radians
    deg2rad : float
        Degrees to radians conversion
    rad2deg : float
        Radians to degrees conversion

    Returns
    -------
    ZenithAngle_rad : float
        Zenith angle in radians
    ZenithAngle_deg : float
        Zenith angle in degrees
    '''
    ZenithAngle_rad = np.arccos(np.cos(lat * deg2rad)
                                * np.cos(Declination_rad)
                                * np.cos(HourAngle_rad)
                                + np.sin(lat * deg2rad)
                                * np.sin(Declination_rad))

    ZenithAngle_deg = ZenithAngle_rad * rad2deg
    return ZenithAngle_rad, ZenithAngle_deg


def calculate_angle_difference(ZenithAngle_rad, HourAngle_rad, phi_sensor_rad,
                  theta_sensor_rad):
    '''Calculate angle between sun and upper sensor (to determine when sun is
    in sight of upper sensor

    Parameters
    ----------
    ZenithAngle_rad : float
        Zenith angle in radians
    HourAngle_rad : float
        Sun hour angle in radians
    phi_sensor_rad : xarray.DataArray
        Spherical tilt coordinates
    theta_sensor_rad : xarray.DataArray
        Total tilt of sensor, where 0 is horizontal

    Returns
    -------
    float
        Angle between sun and sensor
    '''
    return 180 / np.pi * np.arccos(np.sin(ZenithAngle_rad)
                                   * np.cos(HourAngle_rad + np.pi)
                                   * np.sin(theta_sensor_rad)
                                   * np.cos(phi_sensor_rad)
                                   + np.sin(ZenithAngle_rad)
                                   * np.sin(HourAngle_rad + np.pi)
                                   * np.sin(theta_sensor_rad)
                                   * np.sin(phi_sensor_rad)
                                   + np.cos(ZenithAngle_rad)
                                   * np.cos(theta_sensor_rad))


def calculate_albedo(ds, AngleDif_deg, ZenithAngle_deg):
    '''
    Calculate surface albedo based on upwelling and downwelling shortwave
    flux, the angle between the sun and sensor, and the sun zenith angle.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'usr' (upwelling shortwave), 'dsr_cor' (corrected downwelling shortwave),
        and optionally 'dsr' (uncorrected downwelling shortwave) and 'cc' (cloud cover).
    AngleDif_deg : xarray.DataArray
        Angle between the sun and the sensor in degrees.
    ZenithAngle_deg : xarray.DataArray
        Sun zenith angle in degrees.

    Returns
    -------
    ds : xarray.Dataset
        Input dataset with a new 'albedo' variable added.
    OKalbedos : xarray.DataArray
        Boolean mask indicating valid albedo values.
    '''
    tilt_correction_possible = AngleDif_deg.notnull() & ds['cc'].notnull()

    ds['albedo'] = xr.where(tilt_correction_possible,
                            ds['usr'] / ds['dsr_cor'],
                            ds['usr'] / ds['dsr'])

    OOL = (ds['albedo']  >= 1) | (ds['albedo']  <= 0)
    good_zenith_angle = ZenithAngle_deg < 70
    good_relative_zenith_angle = (AngleDif_deg < 70) | (AngleDif_deg.isnull())
    OKalbedos = good_relative_zenith_angle & good_zenith_angle & ~OOL
    ds['albedo'] = ds['albedo'].where(OKalbedos)
    return ds, OKalbedos

def calculate_TOA(ZenithAngle_deg, ZenithAngle_rad):
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

def calculate_correction_factor(Declination_rad, phi_sensor_rad, theta_sensor_rad,
                          HourAngle_rad, ZenithAngle_rad, ZenithAngle_deg,
                          lat, DifFrac, deg2rad):
    '''Calculate correction factor for direct beam radiation, as described
    here: http://solardat.uoregon.edu/SolarRadiationBasics.html

    Offset correction (where solar zenith angles larger than 110 degrees) not
    implemented as it should not improve the accuracy of well-calibrated
    instruments. It would go something like this:
    ds['dsr'] = ds['dsr'] - ds['dwr_offset']
    SRout = SRout - SRout_offset

    Parameters
    ----------
    Declination_rad : float
        Declination in radians
    phi_sensor_rad : xarray.DataArray
        Spherical tilt coordinates
    theta_sensor_rad : xarray.DataArray
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
    deg2rad : float
        Degrees to radians conversion

    Returns
    -------
    CorFac_all : xarray.DataArray
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
    # sun out of field of view upper sensor
    CorFac[(CorFac < 0) | (ZenithAngle_deg > 90)] = 1

    # Calculating ds['dsr'] over a horizontal surface corrected for station/sensor tilt
    CorFac_all = CorFac / (1 - DifFrac + CorFac * DifFrac)

    return CorFac_all.where(theta_sensor_rad.notnull())