#!/usr/bin/env python
"""
AWS Level 1 (L1) to Level 2 (L2) data processing
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.qc.github_data_issues import flagNAN, adjustTime, adjustData
from pypromice.core.qc.percentiles.outlier_detector import ThresholdBasedOutlierDetector
from pypromice.core.qc.persistence import persistence_qc
from pypromice.core.qc.value_clipping import clip_values
from pypromice.core.variables import wind, precipitation

__all__ = [
    "toL2",
]

logger = logging.getLogger(__name__)


def toL2(
    L1: xr.Dataset,
    vars_df: pd.DataFrame,
    data_flags_dir: Path,
    data_adjustments_dir: Path,
    T_0=273.15,
    ews=1013.246,
    ei0=6.1071,
    emissivity=0.97,
) -> xr.Dataset:
    '''Process one Level 1 (L1) product to Level 2.
    In this step we do:
        - manual flagging and adjustments
        - automated QC: persistence, percentile
        - custom filter: gps_alt filter, NaN t_rad removed from dlr & ulr
        - smoothing of tilt and rot
        - calculation of rh with regards to ice in subfreezin conditions
        - calculation of cloud coverage
        - correction of dsr and usr for tilt
        - filtering of dsr based on a theoritical TOA irradiance and grazing light
        - calculation of albedo
        - calculation of directional wind speed

    Parameters
    ----------
    L1 : xarray.Dataset
        Level 1 dataset
    vars_df : pd.DataFrame
        Metadata dataframe
    T_0 : float
        Ice point temperature in K. The default is 273.15.
    ews : float
        Saturation pressure (normal atmosphere) at steam point temperature.
        The default is 1013.246.
    ei0 : float
        Saturation pressure (normal atmosphere) at ice-point temperature. The
        default is 6.1071.
    eps_overcast : int
        Cloud overcast. The default is 1..
    eps_clear : float
        Cloud clear. The default is 9.36508e-6.
    emissivity : float
        Emissivity. The default is 0.97.

    Returns
    -------
    ds : xarray.Dataset
        Level 2 dataset
    '''
    ds = L1.copy()                                                    # Reassign dataset
    ds.attrs['level'] = 'L2'
    try:
        ds = adjustTime(ds, adj_dir=data_adjustments_dir.as_posix())       # Adjust time after a user-defined csv files
        ds = flagNAN(ds, flag_dir=data_flags_dir.as_posix())             # Flag NaNs after a user-defined csv files
        ds = adjustData(ds, adj_dir=data_adjustments_dir.as_posix())       # Adjust data after a user-defined csv files
    except Exception:
        logger.exception('Flagging and fixing failed:')

    ds = persistence_qc(ds)                                               # Flag and remove persistence outliers
    # if ds.attrs['format'] == 'TX':
    #     # TODO: The configuration should be provided explicitly
    #     outlier_detector = ThresholdBasedOutlierDetector.default()
    #     ds = outlier_detector.filter_data(ds)                                 # Flag and remove percentile outliers

    # filtering gps_lat, gps_lon and gps_alt based on the difference to a baseline elevation
    # right now baseline elevation is gapfilled monthly median elevation
    baseline_elevation = (ds.gps_alt.to_series().resample('MS').median()
                          .reindex(ds.time.to_series().index, method='nearest')
                          .ffill().bfill())
    mask = (np.abs(ds.gps_alt - baseline_elevation) < 100) | ds.gps_alt.isnull()
    ds[['gps_alt','gps_lon', 'gps_lat']] = ds[['gps_alt','gps_lon', 'gps_lat']].where(mask)

    # removing dlr and ulr that are missing t_rad
    # this is done now becasue t_rad can be filtered either manually or with persistence
    ds['dlr'] = ds.dlr.where(ds.t_rad.notnull())
    ds['ulr'] = ds.ulr.where(ds.t_rad.notnull())

    # calculating realtive humidity with regard to ice
    T_100 = _getTempK(T_0)
    ds['rh_u_wrt_ice_or_water'] = adjustHumidity(ds['rh_u'], ds['t_u'],
                                     T_0, T_100, ews, ei0)

    if ds.attrs['number_of_booms']==2:
        ds['rh_l_wrt_ice_or_water'] = adjustHumidity(ds['rh_l'], ds['t_l'],
                                         T_0, T_100, ews, ei0)

    if hasattr(ds,'t_i'):
        if ~ds['t_i'].isnull().all():
            ds['rh_i_wrt_ice_or_water'] = adjustHumidity(ds['rh_i'], ds['t_i'],
                                             T_0, T_100, ews, ei0)

    # Determine surface temperature
    ds['t_surf'] = calcSurfaceTemperature(T_0, ds['ulr'], ds['dlr'],
                                          emissivity)
    is_bedrock = ds.attrs['bedrock']
    if not is_bedrock:
        ds['t_surf'] = ds['t_surf'].clip(max=0)

    # smoothing tilt and rot
    ds['tilt_x'] = smoothTilt(ds['tilt_x'])
    ds['tilt_y'] = smoothTilt(ds['tilt_y'])
    ds['rot'] = smoothRot(ds['rot'])

    # Determiune cloud cover for on-ice stations
    if not is_bedrock:
        ds['cc'] = calcCloudCoverage(ds['t_u'], ds['dlr'], ds.attrs['station_id'], T_0)
    else:
        ds['cc'] = ds['t_u'].copy() * np.nan

    # Filtering and correcting shortwave radiation
    ds, _ = process_sw_radiation(ds)

    # Correct precipitation
    if hasattr(ds, "correct_precip"):
        precip_flag = ds.attrs["correct_precip"]
    else:
        precip_flag=True
    if ~ds["precip_u"].isnull().all() and precip_flag:
        ds["precip_u"] = precipitation.filter_lufft_errors(ds["precip_u"], ds["t_u"], ds["p_u"], ds["rh_u"])
        ds["rainfall_u"] = precipitation.get_rainfall_per_timestep(ds["precip_u"])
        ds["rainfall_cor_u"] = precipitation.correct_rainfall_undercatch(ds["rainfall_u"], ds["wspd_u"], ds["t_u"])

    if ds.attrs["number_of_booms"]==2:
        if ~ds["precip_l"].isnull().all() and precip_flag:
            ds["precip_l"] = precipitation.filter_lufft_errors(ds["precip_l"], ds["t_l"], ds["p_l"], ds["rh_l"])
            ds["rainfall_l"] = precipitation.get_rainfall_per_timestep(ds["precip_l"])
            ds["rainfall_cor_l"] = precipitation.correct_rainfall_undercatch(ds["rainfall_l"], ds["wspd_l"], ds["t_l"])

    # Calculate directional wind speed for upper boom
    ds['wdir_u'] = wind.filter_wind_direction(ds['wdir_u'],
                                              ds['wspd_u'])
    ds['wspd_x_u'], ds['wspd_y_u'] = wind.calculate_directional_wind_speed(ds['wspd_u'],
                                                                           ds['wdir_u'])

    # Calculate directional wind speed for lower boom
    if ds.attrs['number_of_booms'] == 2:
        ds['wdir_l'] = wind.filter_wind_direction(ds['wdir_l'],
                                                  ds['wspd_l'])
        ds['wspd_x_l'], ds['wspd_y_l'] = wind.calculate_directional_wind_speed(ds['wspd_l'],

                                                                               ds['wdir_l'])
    # Calculate directional wind speed for instantaneous measurements
    if hasattr(ds, 'wdir_i'):
        if ~ds['wdir_i'].isnull().all() and ~ds['wspd_i'].isnull().all():
            ds['wdir_i'] = wind.filter_wind_direction(ds['wdir_i'],
                                                      ds['wspd_i'])
            ds['wspd_x_i'], ds['wspd_y_i'] = wind.calculate_directional_wind_speed(ds['wspd_i'],
                                                                                   ds['wdir_i'])
            # Get directional wind speed

    ds = clip_values(ds, vars_df)
    return ds

def process_sw_radiation(ds):
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

    deg2rad, rad2deg = _getRotation()                                          # Get degree-radian conversions
    phi_sensor_rad, theta_sensor_rad = calcTilt(ds['tilt_x'], ds['tilt_y'],    # Calculate station tilt
                                                deg2rad)

    Declination_rad = calcDeclination(doy, hour, minute)                       # Calculate declination
    HourAngle_rad = calcHourAngle(hour, minute, lon)                           # Calculate hour angle
    ZenithAngle_rad, ZenithAngle_deg = calcZenith(lat, Declination_rad,        # Calculate zenith
                                                  HourAngle_rad, deg2rad,
                                                  rad2deg)

    # Setting to zero when sun below the horizon.
    bad = ZenithAngle_deg > 95
    ds['dsr'][bad & ds['dsr'].notnull()] = 0
    ds['usr'][bad & ds['usr'].notnull()] = 0

    # Setting to zero when values are negative
    ds['dsr'] = ds['dsr'].clip(min=0)
    ds['usr'] = ds['usr'].clip(min=0)

    # Calculate angle between sun and sensor
    AngleDif_deg = calcAngleDiff(ZenithAngle_rad, HourAngle_rad,
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
    isr_toa = calcTOA(ZenithAngle_deg, ZenithAngle_rad)                        # Calculate TOA shortwave radiation
    TOA_crit_nopass = ~tilt_correction_possible & (ds['dsr'] > (1.2 * isr_toa + 150))
    ds['dsr'][TOA_crit_nopass] = np.nan

    # the upward flux should not be higher than the TOA downard flux
    TOA_crit_nopass_usr = (ds['usr'] > 0.8*(1.2 * isr_toa + 150))
    ds['usr'][TOA_crit_nopass_usr] = np.nan

    # Diffuse to direct irradiance fraction
    DifFrac = 0.2 + 0.8 * ds['cc']
    CorFac_all = calcCorrectionFactor(Declination_rad, phi_sensor_rad,         # Calculate correction
                                      theta_sensor_rad, HourAngle_rad,
                                      ZenithAngle_rad, ZenithAngle_deg,
                                      lat, DifFrac, deg2rad)
    CorFac_all = CorFac_all.where(tilt_correction_possible)

    # Correct Downwelling shortwave radiation
    ds['dsr_cor'] = ds['dsr'].copy() * CorFac_all
    ds['usr_cor'] = ds['usr'].copy().where(ds['dsr_cor'].notnull())

    # Remove data where TOA shortwave radiation invalid
    # this can only be done after correcting for tilt
    TOA_crit_nopass_cor = ds['dsr_cor'] > (1.2 * isr_toa + 150)
    ds['dsr_cor'][TOA_crit_nopass_cor] = np.nan
    ds['usr_cor'][TOA_crit_nopass_cor] = np.nan

    ds, OKalbedos = calcAlbedo(ds, AngleDif_deg, ZenithAngle_deg)

    return ds, (OKalbedos, sunonlowerdome, bad, isr_toa, TOA_crit_nopass_cor, TOA_crit_nopass, TOA_crit_nopass_usr)


def calcCloudCoverage(T, dlr, station_id,T_0, eps_overcast=1.0,
    eps_clear=9.36508e-6):
    '''Calculate cloud cover from T and T_0

    Parameters
    ----------
    T : xarray.DataArray
        Air temperature 1
    T_0 : xarray.DataArray
        Air temperature 0
    eps_overcast : int
        Cloud overcast assumption, from Swinbank (1963)
    eps_clear : int
        Cloud clear assumption, from Swinbank (1963)
    dlr : xarray.DataArray
        Downwelling longwave radiation, with array of same length as T and T_0
    station_id : str
        Station ID string, for special cases at selected stations where cloud
        overcast and cloud clear assumptions are pre-defined. Currently
        KAN_M and KAN_U are special cases, but this will need to be done for
        all stations eventually

    Returns
    -------
    cc : xarray.DataArray
        Cloud cover data array
    '''
    if station_id == 'KAN_M':
       LR_overcast = 315 + 4*T
       LR_clear = 30 + 4.6e-13 * (T + T_0)**6
    elif station_id == 'KAN_U':
       LR_overcast = 305 + 4*T
       LR_clear = 220 + 3.5*T
    else:
       LR_overcast = eps_overcast * 5.67e-8 *(T + T_0)**4
       LR_clear = eps_clear * 5.67e-8 * (T + T_0)**6
    cc = (dlr - LR_clear) / (LR_overcast - LR_clear)
    cc[cc > 1] = 1
    cc[cc < 0] = 0

    return cc


def calcSurfaceTemperature(T_0, ulr, dlr, emissivity):
    '''Calculate surface temperature from air temperature, upwelling and
    downwelling radiation and emissivity

    Parameters
    ----------
    T_0 : xarray.DataArray
        Air temperature
    ulr : xarray.DataArray
        Upwelling longwave radiation
    dlr : xarray.DataArray
        Downwelling longwave radiation
    emissivity : int
        Assumed emissivity

    Returns
    -------
    xarray.DataArray
        Calculated surface temperature
    '''
    t_surf = ((ulr - (1 - emissivity) * dlr) / emissivity / 5.67e-8)**0.25 - T_0
    return t_surf


def smoothTilt(da: xr.DataArray, threshold=0.2):
    '''Smooth the station tilt

    Parameters
    ----------
    da : xarray.DataArray
        either X or Y tilt inclinometer measurements
    threshold : float
        threshold used in a standrad.-deviation based filter

    Returns
    -------
    xarray.DataArray
        either X or Y smoothed tilt inclinometer measurements
    '''
    # we calculate the moving standard deviation over a 3-day sliding window
    # hourly resampling is necessary to make sure the same threshold can be used
    # for 10 min and hourly data
    moving_std_gap_filled = da.to_series().resample('h').median().rolling(
                    3*24, center=True, min_periods=2
                    ).std().reindex(da.time, method='bfill').values
    # we select the good timestamps and gapfill assuming that
    # - when tilt goes missing the last available value is used
    # - when tilt is not available for the very first time steps, the first
    #   good value is used for backfill
    return da.where(
                moving_std_gap_filled < threshold
                ).ffill(dim='time').bfill(dim='time')


def smoothRot(da: xr.DataArray, threshold=4):
    '''Smooth the station rotation

    Parameters
    ----------
    da : xarray.DataArray
        rotation measurements from inclinometer
    threshold : float
        threshold used in a standrad-deviation based filter

    Returns
    -------
    xarray.DataArray
        smoothed rotation measurements from inclinometer
    '''
    moving_std_gap_filled = da.to_series().resample('h').median().rolling(
                    3*24, center=True, min_periods=2
                    ).std().reindex(da.time, method='bfill').values
    # same as for tilt with, in addition:
    #     - a resampling to daily values
    #     - a two week median smoothing
    #     - a resampling from these daily values to the original temporal resolution
    return ('time', (da.where(moving_std_gap_filled <4).ffill(dim='time')
            .to_series().resample('D').median()
            .rolling(7*2,center=True,min_periods=2).median()
            .reindex(da.time, method='bfill').values
            ))


def calcTilt(tilt_x, tilt_y, deg2rad):
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


def adjustHumidity(rh, T, T_0, T_100, ews, ei0):
    '''Adjust relative humidity so that values are given with respect to
    saturation over ice in subfreezing conditions, and with respect to
    saturation over water (as given by the instrument) above the melting
    point temperature. Saturation water vapors are calculated after
    Groff & Gratch method.

    Parameters
    ----------
    rh : xarray.DataArray
        Relative humidity
    T : xarray.DataArray
        Air temperature
    T_0 : float
        Ice point temperature in K
    T_100 : float
        Steam point temperature in K
    ews : float
        Saturation pressure (normal atmosphere) at steam point temperature
    ei0 : float
        Saturation pressure (normal atmosphere) at ice-point temperature

    Returns
    -------
    rh_wrt_ice_or_water : xarray.DataArray
        Corrected relative humidity
    '''
    # Convert to hPa (Groff & Gratch)
    e_s_wtr = 10**(-7.90298 * (T_100 / (T + T_0) - 1)
                   + 5.02808 * np.log10(T_100 / (T + T_0))
                   - 1.3816E-7 * (10**(11.344 * (1 - (T + T_0) / T_100)) - 1)
                   + 8.1328E-3 * (10**(-3.49149 * (T_100/(T + T_0) - 1)) -1)
                   + np.log10(ews))
    e_s_ice = 10**(-9.09718 * (T_0 / (T + T_0) - 1)
                   - 3.56654 * np.log10(T_0 / (T + T_0))
                   + 0.876793 * (1 - (T + T_0) / T_0)
                   + np.log10(ei0))

    # Define freezing point. Why > -100?
    freezing = (T < 0) & (T > -100).values

    # Set to Groff & Gratch values when freezing, otherwise just rh
    rh_wrt_ice_or_water = rh.where(~freezing, other = rh*(e_s_wtr / e_s_ice))
    return rh_wrt_ice_or_water


def calcDeclination(doy, hour, minute):
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

def calcHourAngle(hour, minute, lon):
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


def calcDirectionDeg(HourAngle_rad):                                          #TODO remove if not plan to use this
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

def calcZenith(lat, Declination_rad, HourAngle_rad, deg2rad, rad2deg):
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


def calcAngleDiff(ZenithAngle_rad, HourAngle_rad, phi_sensor_rad,
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


def calcAlbedo(ds, AngleDif_deg, ZenithAngle_deg):
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

def calcTOA(ZenithAngle_deg, ZenithAngle_rad):
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

def calcCorrectionFactor(Declination_rad, phi_sensor_rad, theta_sensor_rad,
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


def _getTempK(T_0):                                                            #TODO same as L2toL3._getTempK()
    '''Return steam point temperature in Kelvins

    Parameters
    ----------
    T_0 : float
        Ice point temperature in K

    Returns
    -------
    float
        Steam point temperature in K'''
    return T_0+100


def _getRotation():                                                            #TODO same as L2toL3._getRotation()
    '''Return degrees-to-radians and radians-to-degrees'''
    deg2rad = np.pi / 180
    rad2deg = 1 / deg2rad
    return deg2rad, rad2deg


if __name__ == "__main__":
    # unittest.main()
    pass
