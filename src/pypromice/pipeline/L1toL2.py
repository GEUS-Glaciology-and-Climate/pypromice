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
from pypromice.core.variables import wind, precipitation, radiation, station_pose

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

    # Removing dlr and ulr that are missing t_rad
    # this is done now because t_rad can be filtered either manually or with persistence
    ds["dlr"] = radiation.filter_lr(ds["dlr"], ds["t_rad"])
    ds["ulr"] = radiation.filter_lr(ds["ulr"], ds["t_rad"])

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
    ds['t_surf'] = radiation.calculate_surface_temperature(ds['dlr'],
                                                           ds['ulr'])
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

    # Determine station pose relative to sun position
    if hasattr(ds, 'latitude') and hasattr(ds, 'longitude'):
        lat = ds.attrs['latitude']                                             # TODO Why is mean GPS lat lon not preferred for calcs?
        lon = ds.attrs['longitude']
    else:
        lat = ds['gps_lat'].mean()
        lon = ds['gps_lon'].mean()

        # Determine station position relative to sun
    #    doy = ds['time'].to_dataframe().index.dayofyear.values
    #    hour = ds['time'].to_dataframe().index.hour.values
    #    minute = ds['time'].to_dataframe().index.minute.values

    # Determine station position relative to sun
    doy = ds['time'].dt.dayofyear
    hour = ds['time'].dt.hour
    minute = ds['time'].dt.minute
    phi_sensor_rad, theta_sensor_rad = station_pose.calculate_spherical_tilt(ds['tilt_x'], ds['tilt_y'])
    Declination_rad = station_pose.calculate_declination(doy, hour, minute)
    HourAngle_rad = station_pose.calculate_hour_angle(hour, minute, lon)
    ZenithAngle_rad, ZenithAngle_deg = station_pose.calculate_zenith(lat,
                                                                     Declination_rad,
                                                                     HourAngle_rad)
    AngleDif_deg = station_pose.calculate_angle_difference(ZenithAngle_rad,
                                                           HourAngle_rad,
                                                           phi_sensor_rad,
                                                           theta_sensor_rad)

    # Filter shortwave radiation
    ds["dsr_filtered"], ds["usr_filtered"], _ = radiation.filter_sr(ds["dsr"],
                                                                    ds["usr"],
                                                                    ds["cc"],
                                                                    ZenithAngle_rad,
                                                                    ZenithAngle_deg,
                                                                    AngleDif_deg)

    # Correct shortwave radiation
    ds["dsr_cor"], ds["usr_cor"], _ = radiation.correct_sr(ds["dsr_filtered"],
                                                           ds["usr_filtered"],
                                                           ds["cc"],
                                                           phi_sensor_rad,
                                                           theta_sensor_rad,
                                                           lat,
                                                           Declination_rad,
                                                           HourAngle_rad,
                                                           ZenithAngle_rad,
                                                           ZenithAngle_deg,
                                                           AngleDif_deg)

    ds['albedo'], _ = radiation.calculate_albedo(ds["usr_filtered"],
                                                 ds["dsr_filtered"],
                                                 ds["dsr_cor"],
                                                 ds["cc"],
                                                 ZenithAngle_deg,
                                                 AngleDif_deg)

    # Correct precipitation
    if hasattr(ds, "correct_precip"):
        precip_flag = ds.attrs["correct_precip"]
    else:
        precip_flag=True
    if ~ds["precip_u"].isnull().all() and precip_flag:
        ds["precip_u"] = precipitation.filter(ds["precip_u"], ds["t_u"], ds["p_u"], ds["rh_u"])
        ds["precip_rate_u"] = precipitation.convert_to_rate(ds["precip_u"], ds["wspd_u"], ds["t_u"])

    if ds.attrs["number_of_booms"]==2:
        if ~ds["precip_l"].isnull().all() and precip_flag:
                ds["precip_l"] = precipitation.filter(ds["precip_l"], ds["t_l"], ds["p_l"], ds["rh_l"])
                ds["precip_rate_l"] = precipitation.convert_to_rate(ds["precip_l"], ds["wspd_l"], ds["t_l"])

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

if __name__ == "__main__":
    # unittest.main()
    pass
