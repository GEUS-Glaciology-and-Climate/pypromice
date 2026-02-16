#!/usr/bin/env python
"""
AWS Level 1 (L1) to Level 2 (L2) data processing
"""
__all__ = ["toL2"]

import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from pypromice.core.qc.github_data_issues import flagNAN, adjustTime, adjustData
from pypromice.core.qc.percentiles.outlier_detector import ThresholdBasedOutlierDetector
from pypromice.core.qc.persistence import persistence_qc
from pypromice.core.qc.value_clipping import clip_values
from pypromice.core.variables import (wind,
                                      gps,
                                      precipitation,
                                      humidity,
                                      radiation,
                                      station_pose,
                                      air_temperature)
from pypromice.core.qc.rate_of_change_filter import rate_of_change_filter
from pypromice.core.qc.common import remove_flagged_data, add_qc_variables

def toL2(L1: xr.Dataset,
         vars_df: pd.DataFrame,
         data_flags_dir: Path,
         data_adjustments_dir: Path,
         keep_flagged_data: str = False,
) -> xr.Dataset:
    """Process one Level 1 (L1) product to Level 2.
    In this step we do:
        - manual flagging and adjustments
        - automated QC: persistence, percentile
        - custom filter: gps_alt filter, NaN t_rad removed from dlr & ulr
        - smoothing of tilt and rot
        - calculation of rh with regard to ice in subfreezing conditions
        - calculation of cloud coverage
        - correction of dsr and usr for tilt
        - filtering of dsr based on a theoretical TOA irradiance and grazing light
        - calculation of albedo
        - calculation of directional wind speed

    Parameters
    ----------
    L1 : xr.Dataset
        Level 1 dataset
    vars_df : pd.DataFrame
        Metadata dataframe
    data_flags_dir : pathlib.Path
        Directory path to data flags file
    data_adjustments_dir : pathlib.Path
        Directory path to data adjustments file
    keep_flagged_data: str
        If False replace all data not flagged as "OK" by NaN

    Returns
    -------
    ds : xr.Dataset
        Level 2 dataset
    """
    # Copy input dataset and add quality flag variables
    ds = L1.copy(deep=True)

    # Flag and remove persistence outliers
    ds = persistence_qc(ds)

    # Flag high-rate-of-change outliers
    ds = rate_of_change_filter(ds)

    try:
        # Adjust time after a user-defined csv files
        ds = adjustTime(ds, adj_dir=data_adjustments_dir.as_posix())

        # Flag NaNs after a user-defined csv files
        ds = flagNAN(ds, flag_dir=data_flags_dir.as_posix())

        # Adjust data after a user-defined csv files
        ds = adjustData(ds, adj_dir=data_adjustments_dir.as_posix())

    except Exception:
        logger.exception("Flagging and fixing failed:")

    # if ds.attrs['format'] == 'TX':
    #     # TODO: The configuration should be provided explicitly
    #     outlier_detector = ThresholdBasedOutlierDetector.default()
    #     ds = outlier_detector.filter_data(ds)

    # Filter GPS values based on baseline elevation
    ds["gps_lat_qc"], ds["gps_lon_qc"], ds["gps_alt_qc"] = gps.filter(ds["gps_lat"],
                                                                      ds["gps_lon"],
                                                                      ds["gps_alt"],
                                                                      ds["gps_lat_qc"],
                                                                      ds["gps_lon_qc"],
                                                                      ds["gps_alt_qc"])

    # Calculate relative humidity with regard to ice
    ds["rh_u_wrt_ice_or_water"] = humidity.adjust(ds["rh_u"], ds["t_u"])

    if ds.attrs["number_of_booms"]==2:
        ds["rh_l_wrt_ice_or_water"] = humidity.adjust(ds["rh_l"], ds["t_l"])

    if hasattr(ds,"t_i"):
        if ~ds["t_i"].isnull().all():
            ds["rh_i_wrt_ice_or_water"] = humidity.adjust(ds["rh_i"], ds["t_i"])

    # Determine surface temperature
    ds["t_surf"] = radiation.calculate_surface_temperature(ds["dlr"], ds["ulr"])
    is_bedrock = ds.attrs["bedrock"]
    if not is_bedrock:
        ds["t_surf"] = ds["t_surf"].clip(max=0)

    # Interpolate and smooth station tilt and rotation
    # TODO tilt smoothing is performed here and at L0toL1 also (and they are different functions). Is this needed? PHO
    ds['tilt_x'] = station_pose.interpolate_tilt(ds['tilt_x'])
    ds['tilt_y'] = station_pose.interpolate_tilt(ds['tilt_y'])
    ds['rot'] = station_pose.interpolate_rotation(ds['rot'])

    # Determine cloud cover for on-ice stations
    if not is_bedrock:
        # Selected stations have pre-defined cloud assumption coefficients
        # TODO Ideally these will be pre-defined for all stations eventually
        if ds.attrs["station_id"] == "KAN_M":
            LR_overcast = 315 + 4 * ds["t_u"]
            LR_clear = 30 + 4.6e-13 * (ds["t_u"] + air_temperature.T_0) ** 6
        elif ds.attrs["station_id"] == "KAN_U":
            LR_overcast = 305 + 4 * ds["t_u"]
            LR_clear = 220 + 3.5 * ds["t_u"]

        # Else, calculate cloud assumption coefficients based on default values
        else:
            LR_overcast, LR_clear = air_temperature.get_cloud_coefficients(ds["t_u"])

        ds["cc"] = radiation.calculate_cloud_coverage(ds["dlr"], LR_overcast, LR_clear)

    # Set cloud cover to nans if station is not on ice
    else:
        ds["cc"] = xr.full_like(ds["dlr"], np.nan)

    # Determine station pose relative to sun position
    lat = lon = np.nan
    if ("latitude" in ds.attrs) and ("longitude" in ds.attrs):
        lat = float(ds.attrs["latitude"])
        lon = float(ds.attrs["longitude"])
    if ("gps_lat" in ds.data_vars) and ("gps_lon" in ds.data_vars):
        lat_ok = ds["gps_lat"].where(ds["gps_lat_qc"] == "OK")
        lon_ok = ds["gps_lon"].where(ds["gps_lon_qc"] == "OK")

        if lat_ok.notnull().any() and lon_ok.notnull().any():
            lat = lat_ok.mean().item()
            lon = lon_ok.mean().item()

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

    # Clip shortwave radiation
    ds["dsr"], ds["usr"] = radiation.clip_sr(ds["dsr"], ds["usr"])

    # Filter shortwave radiation
    ds["dsr_qc"], ds["usr_qc"] = radiation.filter_sr(ds["dsr"],
                                                     ds["usr"],
                                                     ds["cc"],
                                                     ZenithAngle_rad,
                                                     ZenithAngle_deg,
                                                     AngleDif_deg,
                                                     ds["dsr_qc"],
                                                     ds["usr_qc"])

    # Correct shortwave radiation
    ds["dsr_cor"], ds["usr_cor"], _ = radiation.correct_sr(ds["dsr"],
                                                           ds["usr"],
                                                           ds["cc"],
                                                           phi_sensor_rad,
                                                           theta_sensor_rad,
                                                           lat,
                                                           Declination_rad,
                                                           HourAngle_rad,
                                                           ZenithAngle_rad,
                                                           ZenithAngle_deg,
                                                           AngleDif_deg)

    ds['albedo'], _ = radiation.calculate_albedo(ds["dsr"],
                                                 ds["usr"],
                                                 ds["dsr_cor"],
                                                 ds["cc"],
                                                 ZenithAngle_deg,
                                                 AngleDif_deg)

    # Determine if precipitation filtering and rate needed
    if hasattr(ds, "correct_precip"):
        precip_flag = ds.attrs["correct_precip"]
    else:
        precip_flag=True

    if ~ds["precip_u"].isnull().all() and precip_flag:
        ds["precip_u"] = precipitation.filter_lufft_errors(ds["precip_u"],
                                                           ds["t_u"],
                                                           ds["p_u"],
                                                           ds["rh_u"])

        ds["rainfall_u"] = precipitation.get_rainfall_per_timestep(ds["precip_u"],
                                                                   ds["t_u"])

        ds["rainfall_cor_u"] = precipitation.correct_rainfall_undercatch(ds["rainfall_u"],
                                                                         ds["wspd_u"])

    if ds.attrs["number_of_booms"]==2:
        if ~ds["precip_l"].isnull().all() and precip_flag:
            ds["precip_l"] = precipitation.filter_lufft_errors(ds["precip_l"],
                                                               ds["t_l"],
                                                               ds["p_l"],
                                                               ds["rh_l"])

            ds["rainfall_l"] = precipitation.get_rainfall_per_timestep(ds["precip_l"],
                                                                       ds["t_l"])

            ds["rainfall_cor_l"] = precipitation.correct_rainfall_undercatch(ds["rainfall_l"],
                                                                             ds["wspd_l"])

    # Calculate directional wind speed for upper boom
    ds["wdir_u_qc"] = wind.filter_wind_direction(ds["wdir_u"],
                                                 ds["wspd_u"],
                                                 ds["wdir_u_qc"])

    ds['wspd_x_u'], ds['wspd_y_u'] = wind.calculate_directional_wind_speed(ds['wspd_u'],
                                                                           ds['wdir_u'])

    # Calculate directional wind speed for lower boom
    if ds.attrs['number_of_booms'] == 2:
        ds["wdir_l_qc"] = wind.filter_wind_direction(ds["wdir_l"],
                                                     ds["wspd_l"],
                                                     ds["wdir_l_qc"])

        ds['wspd_x_l'], ds['wspd_y_l'] = wind.calculate_directional_wind_speed(ds['wspd_l'],
                                                                               ds['wdir_l'])

    # Calculate directional wind speed for instantaneous measurements
    if hasattr(ds, 'wdir_i'):
        if ~ds['wdir_i'].isnull().all() and ~ds['wspd_i'].isnull().all():
            ds["wdir_i_qc"] = wind.filter_wind_direction(ds["wdir_i"],
                                                         ds["wspd_i"],
                                                         ds["wdir_i_qc"])

            ds['wspd_x_i'], ds['wspd_y_i'] = wind.calculate_directional_wind_speed(ds['wspd_i'],
                                                                                   ds['wdir_i'])

    # Clip values (i.e. threshold filtering)
    ds = clip_values(ds, vars_df)

    # Removing the non-OK data
    if not keep_flagged_data: ds = remove_flagged_data(ds)

    # Return L2 dataset
    ds.attrs['level'] = 'L2'
    return ds


if __name__ == "__main__":
    pass
