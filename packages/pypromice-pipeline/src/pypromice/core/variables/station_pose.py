__all__ = ["convert_and_filter_tilt", "apply_tilt_factor",
           "smooth_tilt_with_moving_window", "interpolate_tilt",
           "interpolate_rotation", "calculate_spherical_tilt",
           "calculate_declination", "calculate_hour_angle",
           "calculate_sun_direction_degrees", "calculate_zenith",
           "calculate_angle_difference"]

import numpy as np
import xarray as xr
import pandas as pd

tilt_smoothing_win_size = 7 # Station tilt smoothing window size
tilt_stddev_threshold = 0.2 # Station tilt interpolation standard deviation threshold
rot_stddev_threshold = 4    # Station rotation interpolation standard deviation threshold
tilt_threshold = -100       # Station tilt threshold
deg2rad = np.pi / 180       # Degrees to radians conversion
rad2deg = 1 / deg2rad       # Radians to degrees conversion


def apply_tilt_factor(tilt: xr.DataArray,
                      tilt_correction_factor: float
) -> xr.DataArray:
    """Apply tilt correction factor to station tilt values

    Parameters
    ----------
    tilt : xr.DataArray
        Tilt array (either 'tilt_x' or 'tilt_y')
    tilt_correction_factor : float
        Correction factor to apply to tilt measurements

    Returns
    -------
    xr.DataArray
        Corrected tilt measurements
    """
    return tilt * tilt_correction_factor


def convert_and_filter_tilt(tilt: xr.DataArray
) -> xr.DataArray:
    """Convert station tilt from voltage to degrees,
    and filter tilt with given threshold. Voltage-to-degrees
    conversion is based on the equation in 3.2.9 in Fausto
    et al. (2021) https://doi.org/10.5194/essd-13-3819-2021

    Parameters
    ----------
    tilt : xr.DataArray
        Array (either 'tilt_x' or 'tilt_y'), tilt values (voltage)

    Returns
    -------
    xr.DataArray
        Array (either 'tilt_x' or 'tilt_y'), tilt values (degrees)
    """
    # IDL version:
    # notOKtiltX = where(tiltX lt -100, complement=OKtiltX) & notOKtiltY = where(tiltY lt -100, complement=OKtiltY)
    # tiltX = tiltX/10.
    # tiltnonzero = where(tiltX ne 0 and tiltX gt -40 and tiltX lt 40)
    # if n_elements(tiltnonzero) ne 1 then tiltX[tiltnonzero] = tiltX[tiltnonzero]/abs(tiltX[tiltnonzero])*(-0.49*(abs(tiltX[tiltnonzero]))^4 + 3.6*(abs(tiltX[tiltnonzero]))^3 - 10.4*(abs(tiltX[tiltnonzero]))^2 +21.1*(abs(tiltX[tiltnonzero])))
    # tiltY = tiltY/10.
    # tiltnonzero = where(tiltY ne 0 and tiltY gt -40 and tiltY lt 40)
    # if n_elements(tiltnonzero) ne 1 then tiltY[tiltnonzero] = tiltY[tiltnonzero]/abs(tiltY[tiltnonzero])*(-0.49*(abs(tiltY[tiltnonzero]))^4 + 3.6*(abs(tiltY[tiltnonzero]))^3 - 10.4*(abs(tiltY[tiltnonzero]))^2 +21.1*(abs(tiltY[tiltnonzero])))

    # if n_elements(OKtiltX) gt 1 then tiltX[notOKtiltX] = interpol(tiltX[OKtiltX],OKtiltX,notOKtiltX) ; Interpolate over gaps for radiation correction; set to -999 again below.

    # Define valid tilt values and create mask
    notOKtilt = (tilt < tilt_threshold)
    OKtilt = (tilt >= tilt_threshold)

    # Convert tilt values
    dst = tilt / 10
    nz = (dst != 0) & (np.abs(dst) < 40)
    dst = dst.where(~nz, other = dst / np.abs(dst)
                      * (-0.49
                         * (np.abs(dst))**4 + 3.6
                         * (np.abs(dst))**3 - 10.4
                         * (np.abs(dst))**2 + 21.1
                         * (np.abs(dst))))

    # Apply filtering mask
    dst = dst.where(~notOKtilt)

    # TODO: Filling w/o considering time gaps to re-create IDL/GDL outputs.
    #  Should fill with coordinate not False. Also consider 'max_gap' option?
    return dst.interpolate_na(dim='time', use_coordinate=False)


def smooth_tilt_with_moving_window(tilt: xr.DataArray
) -> tuple[str, np.ndarray]:
    """Smooth tilt values using the pandas 'rolling' window method
    e.g. a value of 7 spans 70 minutes using 10-minute data.
    This is translated from the previous IDL/GDL smoothing algorithm:
    tiltX = smooth(tiltX,7,/EDGE_MIRROR,MISSING=-999) & tiltY = smooth(tiltY,7,/EDGE_MIRROR, MISSING=-999)
    endif
    In Python, this should be
    dstxy = dstxy.rolling(time=7, win_type='boxcar', center=True).mean()
    But the EDGE_MIRROR makes it a bit more complicated

    Parameters
    ----------
    tilt : xr.DataArray
        Array (either 'tilt_x' or 'tilt_y'), tilt values (can be in degrees or voltage)

    Returns
    -------
    tdf_rolling : tuple, as: (str, numpy.ndarray)
        The numpy array is the tilt values, smoothed with a rolling mean
    """
    s = int(tilt_smoothing_win_size/2)
    tdf = tilt.to_dataframe()
    mirror_start = tdf.iloc[:s][::-1]
    mirror_end = tdf.iloc[-s:][::-1]
    mirrored_tdf = pd.concat([mirror_start, tdf, mirror_end])

    tdf_rolling = (
        ("time"),
        mirrored_tdf.rolling(
            tilt_smoothing_win_size, win_type="boxcar", min_periods=1, center=True
            ).mean()[s:-s].values.flatten()
        )
    return tdf_rolling


def interpolate_tilt(tilt: xr.DataArray,
) -> xr.DataArray:
    """Interpolate (fill data gaps) and smooth station tilt
    using a moving standard deviation over a 3-day sliding
    window.

    Parameters
    ----------
    tilt : xr.DataArray
        Either X or Y tilt inclinometer measurements

    Returns
    -------
    xr.DataArray
        Either X or Y smoothed tilt inclinometer measurements
    """
    # We calculate the moving standard deviation over a 3-day sliding window
    # hourly resampling is necessary to make sure the same threshold can be used
    # for 10 min and hourly data
    moving_std_gap_filled = tilt.to_series().resample("h").median().rolling(
                    3*24, center=True, min_periods=2
                    ).std().reindex(tilt.time, method="bfill").values

    # We select the good timestamps and gapfill assuming that
    # - when tilt goes missing the last available value is used
    # - when tilt is not available for the very first time steps, the first
    #   good value is used for backfill
    return tilt.where(
                moving_std_gap_filled < tilt_stddev_threshold
                ).ffill(dim="time").bfill(dim="time")


def interpolate_rotation(rot: xr.DataArray,
                         threshold=4):
    """Interpolate and smooth station rotation

    Parameters
    ----------
    rot : xr.DataArray
        Rotation measurements from inclinometer
    threshold : float
        threshold used in a standard deviation based filter

    Returns
    -------
    xr.DataArray
        smoothed rotation measurements from inclinometer
    """
    moving_std_gap_filled = rot.to_series().resample("h").median().rolling(
                    3*24, center=True, min_periods=2
                    ).std().reindex(rot.time, method="bfill").values

    # Same as for interpolate_tilt() with, in addition:
    #     - a resampling to daily values
    #     - a two-week median smoothing
    #     - a resampling from these daily values to the original temporal resolution
    return ("time", (rot.where(moving_std_gap_filled < rot_stddev_threshold)
            .ffill(dim="time")
            .to_series().resample("D").median()
            .rolling(7*2,center=True,min_periods=2).median()
            .reindex(rot.time, method="bfill").values
            ))


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