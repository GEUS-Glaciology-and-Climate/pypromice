__all__ = ["calculate_spherical_tilt", "calculate_declination", "calculate_hour_angle",
           "calculate_sun_direction_degrees", "calculate_zenith", "calculate_angle_difference"]
import numpy as np
import xarray as xr

deg2rad = np.pi / 180       # Degrees to radians conversion
rad2deg = 1 / deg2rad       # Radians to degrees conversion

def calculate_spherical_tilt(
    tilt_x: xr.DataArray,
    tilt_y: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate station tilt

    Parameters
    ----------
    tilt_x : xr.DataArray
        X tilt inclinometer measurements
    tilt_y : xr.DataArray
        Y tilt inclinometer measurements

    Returns
    -------
    phi_sensor_rad : xr.DataArray
        Spherical tilt coordinates
    theta_sensor_rad : xr.DataArray
        Total tilt of sensor, where 0 is horizontal
    """
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

def calculate_declination(doy: xr.DataArray,
                          hour: xr.DataArray,
                          minute: xr.DataArray
) -> xr.DataArray:
    """Calculate sun declination based on time

    Parameters
    ----------
    doy : xr.DataArray
        Day of year
    hour : xr.DataArray
        Hour of day
    minute : xr.DataArray
        Minute of hour

    Returns
    -------
    xr.DataArray
        Sun declination in radians
    """
    d0_rad = 2 * np.pi * (doy + (hour + minute / 60) / 24 -1) / 365
    return np.arcsin(0.006918 - 0.399912
                     * np.cos(d0_rad) + 0.070257
                     * np.sin(d0_rad) - 0.006758
                     * np.cos(2 * d0_rad) + 0.000907
                     * np.sin(2 * d0_rad) - 0.002697
                     * np.cos(3 * d0_rad) + 0.00148
                     * np.sin(3 * d0_rad))

def calculate_hour_angle(hour: xr.DataArray,
                         minute: xr.DataArray,
                         lon: float
) -> xr.DataArray:
    """Calculate hour angle of sun based on time and longitude. Make sure that
    time is set to UTC and longitude is positive when west. Hour angle should
    be 0 at noon

    Parameters
    ----------
    hour : xr.DataArray
        Hour of day
    minute : xr.DataArray
        Minute of hour
    lon : float
        Longitude

    Returns
    -------
    xr.DataArray
        Hour angle of sun
    """
    return 2 * np.pi * (((hour + minute / 60) / 24 - 0.5) - lon/360)
     # ; - 15.*timezone/360.)

def calculate_sun_direction_degrees(HourAngle_rad: xr.DataArray
) -> xr.DataArray:
    """Calculate sun direction as degrees. This is an alternative to
    calculate_hour_angle that is currently not implemented into the official
    L0>>L3 workflow. Here, 180 degrees is at noon (NH), as opposed to
    HourAngle

    Parameters
    ----------
    HourAngle_rad : xr.DataArray
        Sun hour angle in radians

    Returns
    -------
    DirectionSun_deg : xr.DataArray
        Sun direction in degrees
    """
    DirectionSun_deg = HourAngle_rad * 180/np.pi - 180
    DirectionSun_deg[DirectionSun_deg < 0] += 360
    DirectionSun_deg[DirectionSun_deg < 0] += 360
    return DirectionSun_deg

def calculate_zenith(lat: float,
                     Declination_rad: xr.DataArray,
                     HourAngle_rad: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate sun zenith in radians and degrees

    Parameters
    ----------
    lat : float
        Latitude
    Declination_rad : xr.DataArray
        Sun declination in radians
    HourAngle_rad : xr.DataArray
        Sun hour angle in radians

    Returns
    -------
    ZenithAngle_rad : xr.DataArray
        Zenith angle in radians
    ZenithAngle_deg : xr.DataArray
        Zenith angle in degrees
"""
    ZenithAngle_rad = np.arccos(np.cos(lat * deg2rad)
                                * np.cos(Declination_rad)
                                * np.cos(HourAngle_rad)
                                + np.sin(lat * deg2rad)
                                * np.sin(Declination_rad))

    ZenithAngle_deg = ZenithAngle_rad * rad2deg
    return ZenithAngle_rad, ZenithAngle_deg

def calculate_angle_difference(ZenithAngle_rad: xr.DataArray,
                               HourAngle_rad: xr.DataArray,
                               phi_sensor_rad: xr.DataArray,
                               theta_sensor_rad: xr.DataArray
) -> xr.DataArray:
    """Calculate angle between sun and upper sensor (to determine when sun is
    in sight of upper radiometer sensor)

    Parameters
    ----------
    ZenithAngle_rad : xr.DataArray
        Zenith angle in radians
    HourAngle_rad : xr.DataArray
        Sun hour angle in radians
    phi_sensor_rad : xarray.DataArray
        Spherical tilt coordinates
    theta_sensor_rad : xarray.DataArray
        Total tilt of sensor, where 0 is horizontal

    Returns
    -------
    xr.DataArray
        Angle between sun and sensor
    """
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