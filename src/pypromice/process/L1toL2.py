#!/usr/bin/env python
"""
AWS Level 1 (L1) to Level 2 (L2) data processing
"""
import logging

import numpy as np
import pandas as pd
import xarray as xr

from pypromice.qc.github_data_issues import flagNAN, adjustTime, adjustData
from pypromice.qc.percentiles.outlier_detector import ThresholdBasedOutlierDetector
from pypromice.qc.persistence import persistence_qc
from pypromice.process.value_clipping import clip_values

__all__ = [
    "toL2",
]

logger = logging.getLogger(__name__)


def toL2(
    L1: xr.Dataset,
    vars_df: pd.DataFrame,
    T_0=273.15,
    ews=1013.246,
    ei0=6.1071,
    eps_overcast=1.0,
    eps_clear=9.36508e-6,
    emissivity=0.97,
) -> xr.Dataset:
    '''Process one Level 1 (L1) product to Level 2

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
    ds = L1.copy(deep=True)                                                    # Reassign dataset
    try:
        ds = adjustTime(ds)                                                    # Adjust time after a user-defined csv files
        ds = flagNAN(ds)                                                       # Flag NaNs after a user-defined csv files
        ds = adjustData(ds)                                                    # Adjust data after a user-defined csv files
    except Exception:
        logger.exception('Flagging and fixing failed:')

    ds = persistence_qc(ds)                                               # Flag and remove persistence outliers
    # if ds.attrs['format'] == 'TX':
    #     # TODO: The configuration should be provided explicitly
    #     outlier_detector = ThresholdBasedOutlierDetector.default()
    #     ds = outlier_detector.filter_data(ds)                                 # Flag and remove percentile outliers

    # filtering gps_lat, gps_lon and gps_alt based on the difference to a baseline elevation
    # right now baseline elevation is gapfilled monthly median elevation
    baseline_elevation = (ds.gps_alt.to_series().resample('M').median()
                          .reindex(ds.time.to_series().index, method='nearest')
                          .ffill().bfill())
    mask = (np.abs(ds.gps_alt - baseline_elevation) < 100) & ds.gps_alt.notnull()
    ds[['gps_alt','gps_lon', 'gps_lat']] = ds[['gps_alt','gps_lon', 'gps_lat']].where(mask)
    
    # removing dlr and ulr that are missing t_rad
    # this is done now becasue t_rad can be filtered either manually or with persistence
    ds['dlr'] = ds.dlr.where(ds.t_rad.notnull())
    ds['ulr'] = ds.ulr.where(ds.t_rad.notnull())

    T_100 = _getTempK(T_0)
    ds['rh_u_cor'] = correctHumidity(ds['rh_u'], ds['t_u'],
                                     T_0, T_100, ews, ei0)

    # Determiune cloud cover for on-ice stations
    if not ds.attrs['bedrock']:
        cc = calcCloudCoverage(ds['t_u'], T_0, eps_overcast, eps_clear,        # Calculate cloud coverage
                               ds['dlr'], ds.attrs['station_id'])
        ds['cc'] = (('time'), cc.data)
    else:
        # Default cloud cover for bedrock station for which tilt should be 0 anyway.
        cc = 0.8

    # Determine surface temperature
    ds['t_surf'] = calcSurfaceTemperature(T_0, ds['ulr'], ds['dlr'],           # Calculate surface temperature
                                          emissivity)
    if not ds.attrs['bedrock']:
        ds['t_surf'] = xr.where(ds['t_surf'] > 0, 0, ds['t_surf'])

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

    # Correct Downwelling shortwave radiation
    DifFrac = 0.2 + 0.8 * cc
    CorFac_all = calcCorrectionFactor(Declination_rad, phi_sensor_rad,         # Calculate correction
                                      theta_sensor_rad, HourAngle_rad,
                                      ZenithAngle_rad, ZenithAngle_deg,
                                      lat, DifFrac, deg2rad)
    ds['dsr_cor'] = ds['dsr'].copy(deep=True) * CorFac_all                     # Apply correction

    AngleDif_deg = calcAngleDiff(ZenithAngle_rad, HourAngle_rad,               # Calculate angle between sun and sensor
                                 phi_sensor_rad, theta_sensor_rad)

    ds['albedo'], OKalbedos = calcAlbedo(ds['usr'], ds['dsr_cor'],             # Determine albedo
                              AngleDif_deg, ZenithAngle_deg)

    # Correct upwelling and downwelling shortwave radiation
    sunonlowerdome =(AngleDif_deg >= 90) & (ZenithAngle_deg <= 90)             # Determine when sun is in FOV of lower sensor, assuming sensor measures only diffuse radiation
    ds['dsr_cor'] = ds['dsr_cor'].where(~sunonlowerdome,
                                        other=ds['dsr'] / DifFrac)             # Apply to downwelling
    ds['usr_cor'] = ds['usr'].copy(deep=True)
    ds['usr_cor'] = ds['usr_cor'].where(~sunonlowerdome,
                                        other=ds['albedo'] * ds['dsr'] / DifFrac) # Apply to upwelling
    bad = (ZenithAngle_deg > 95) | (ds['dsr_cor'] <= 0) | (ds['usr_cor'] <= 0) # Set to zero for solar zenith angles larger than 95 deg or either values are (less than) zero
    ds['dsr_cor'][bad] = 0
    ds['usr_cor'][bad] = 0
    ds['dsr_cor'] = ds['usr_cor'].copy(deep=True) / ds['albedo']               # Correct DWR using more reliable USWR when sun not in sight of upper sensor
    ds['albedo'] = ds['albedo'].where(OKalbedos)                               #TODO remove?

    # Remove data where TOA shortwave radiation invalid
    isr_toa = calcTOA(ZenithAngle_deg, ZenithAngle_rad)                        # Calculate TOA shortwave radiation
    TOA_crit_nopass = (ds['dsr_cor'] > (0.9 * isr_toa + 10))                   # Determine filter
    ds['dsr_cor'][TOA_crit_nopass] = np.nan                                    # Apply filter and interpolate
    ds['usr_cor'][TOA_crit_nopass] = np.nan
    ds['dsr_cor'] = ds['dsr_cor'].interpolate_na(dim='time', use_coordinate=False)
    ds['usr_cor'] = ds['usr_cor'].interpolate_na(dim='time', use_coordinate=False)

    # # Check sun position
    # sundown = ZenithAngle_deg >= 90
    # _checkSunPos(ds, OKalbedos, sundown, sunonlowerdome, TOA_crit_nopass)

    if hasattr(ds, 'correct_precip'):                                          # Correct precipitation
        precip_flag=ds.attrs['correct_precip']
    else:
        precip_flag=True
    if ~ds['precip_u'].isnull().all() and precip_flag:
        ds['precip_u_cor'], ds['precip_u_rate'] = correctPrecip(ds['precip_u'],
                                                                ds['wspd_u'])
    if ds.attrs['number_of_booms']==2:
        ds['rh_l_cor'] = correctHumidity(ds['rh_l'], ds['t_l'],           # Correct relative humidity
                                         T_0, T_100, ews, ei0)

        if ~ds['precip_l'].isnull().all() and precip_flag:                     # Correct precipitation
            ds['precip_l_cor'], ds['precip_l_rate']= correctPrecip(ds['precip_l'],
                                                                   ds['wspd_l'])

    if hasattr(ds,'t_i'):
        if ~ds['t_i'].isnull().all():                                          # Instantaneous msg processing
            ds['rh_i_cor'] = correctHumidity(ds['rh_i'], ds['t_i'],       # Correct relative humidity
                                             T_0, T_100, ews, ei0)

    ds = clip_values(ds, vars_df)
    return ds


def calcCloudCoverage(T, T_0, eps_overcast, eps_clear, dlr, station_id):
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
    # phi_sensor_deg = phi_sensor_rad * rad2deg                                #TODO take these out if not needed
    # theta_sensor_deg = theta_sensor_rad * rad2deg
    return phi_sensor_rad, theta_sensor_rad


def correctHumidity(rh, T, T_0, T_100, ews, ei0):                        #TODO figure out if T replicate is needed
    '''Correct relative humidity using Groff & Gratch method, where values are
    set when freezing and remain the original values when not freezing

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
    rh_cor : xarray.DataArray
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

    # Set to Groff & Gratch values when freezing, otherwise just rh
    rh_cor = xr.where(T>=0, rh, rh*(e_s_wtr / e_s_ice))
    rh_cor = rh_cor.where(T.notnull())
    return rh_cor


def correctPrecip(precip, wspd):
    '''Correct precipitation with the undercatch correction method used in
    Yang et al. (1999) and Box et al. (2022), based on Goodison et al. (1998)

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
    precip : xarray.DataArray
        Cumulative precipitation measurements
    wspd : xarray.DataArray
        Wind speed measurements

    Returns
    -------
    precip_cor : xarray.DataArray
        Cumulative precipitation corrected
    precip_rate : xarray.DataArray
        Precipitation rate corrected
    '''
    # Calculate undercatch correction factor
    corr=100/(100.00-4.37*wspd+0.35*wspd*wspd)

    # Fix all values below 1.02 to 1.02
    corr = corr.where(corr>1.02, other=1.02)

    # Fill nan values in precip with preceding value
    precip = precip.ffill(dim='time')

    # Calculate precipitation rate
    precip_rate = precip.diff(dim='time', n=1)

    # Apply correction to rate
    precip_rate = precip_rate*corr

    # Flag rain bucket reset
    precip_rate = precip_rate.where(precip_rate>-0.01, other=np.nan)
    b = precip_rate.to_dataframe('precip_flag').notna().to_xarray()

    # Get corrected cumulative precipitation, reset if rain bucket flag
    precip_cor = precip_rate.cumsum()-precip_rate.cumsum().where(~b['precip_flag']).ffill(dim='time').fillna(0).astype(float)

    return precip_cor, precip_rate


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

def calcAlbedo(usr, dsr_cor, AngleDif_deg, ZenithAngle_deg):
    '''Calculate surface albedo based on upwelling and downwelling shorwave
    flux, the angle between the sun and sensor, and the sun zenith

    Parameters
    ----------
    usr : xarray.DataArray
        Upwelling shortwave radiation
    dsr_cor : xarray.DataArray
        Downwelling shortwave radiation corrected
    AngleDif_def : float
        Angle between sun and sensor in degrees
    ZenithAngle_deg: float
        Zenith angle in degrees

    Returns
    -------
    albedo : xarray.DataArray
        Derived albedo
    OKalbedos : xarray.DataArray
        Valid albedo measurements
    '''
    albedo = usr / dsr_cor

    # NaN bad data
    OKalbedos = (AngleDif_deg < 70) & (ZenithAngle_deg < 70) & (albedo < 1) & (albedo > 0)
    albedo[~OKalbedos] = np.nan

    # Interpolate all. Note "use_coordinate=False" is used here to force
    # comparison against the GDL code when that is run with *only* a TX file.
    # Should eventually set to default (True) and interpolate based on time,
    # not index.
    albedo = albedo.interpolate_na(dim='time', use_coordinate=False)
    albedo = albedo.ffill(dim='time').bfill(dim='time')                        #TODO remove this line and one above?
    return albedo, OKalbedos


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

    return CorFac_all


def _checkSunPos(ds, OKalbedos, sundown, sunonlowerdome, TOA_crit_nopass):
    '''Check sun position

    Parameters
    ----------
    ds : xarray.Dataset
        Data set
    OKalbedos : xarray.DataArray
        Valid measurements flag
    sundown : xarray.DataArray
        Sun below horizon flag
    sunonlowerdome : xarray.DataArray
        Sun in view of lower sensor flag
    TOA_crit_nopass : xarray.DataArray
        Top-of-Atmosphere flag
    '''
    valid = (~(ds['dsr_cor'].isnull())).sum()
    print('Sun in view of upper sensor / workable albedos:', OKalbedos.sum().values,
          (100*OKalbedos.sum()/valid).round().values, "%")
    print('Sun below horizon:', sundown.sum(),
          (100*sundown.sum()/valid).round().values, "%")
    print('Sun in view of lower sensor:', sunonlowerdome.sum().values,
          (100*sunonlowerdome.sum()/valid).round().values, "%")
    print('Spikes removed using TOA criteria:', TOA_crit_nopass.sum().values,
          (100*TOA_crit_nopass.sum()/valid).round().values, "%")
    print('Mean net SR change by corrections:',
          (ds['dsr_cor']-ds['usr_cor']-ds['dsr']+ds['usr']).sum().values/valid.values,
          "W/m2")

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
