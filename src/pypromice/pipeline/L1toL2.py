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

from pypromice.core.qc.common import finalize_qc, clean_view
from pypromice.core.qc.github_data_issues import flagNAN, adjustTime, adjustData
from pypromice.core.qc.percentiles.outlier_detector import ThresholdBasedOutlierDetector
from pypromice.core.qc.persistence import persistence_qc
from pypromice.core.qc.rate_of_change_filter import rate_of_change_filter
from pypromice.core.qc.value_clipping import clip_values
from pypromice.core.variables import (wind,
                                      gps,
                                      precipitation,
                                      humidity,
                                      radiation,
                                      station_pose,
                                      air_temperature)


def toL2(L1: xr.Dataset,
         vars_df: pd.DataFrame,
         data_flags_dir: Path,
         data_adjustments_dir: Path,
         keep_flagged_data: bool = False,
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

    Every QC step along the way (persistence, rate-of-change, value
    clipping, manual GitHub-issue flags) flags rejected samples on a
    "<var>_qc" companion variable but never touches the data itself, so
    every variable's true, original reading stays available in the
    returned dataset. Every calculation here that consumes a raw sensor
    reading does so through a "clean" view (flagged samples as NaN)
    instead, so the science is never computed from a flagged value. By
    default the "<var>_qc" variables (and, along with them, every flagged
    sample) are dropped at the end, so the returned dataset is unchanged
    from a non-flag-based pipeline; pass ``keep_flagged_data=True`` to
    instead get every variable's true reading back, flagged or not, with
    its "<var>_qc" companion attached.

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
    keep_flagged_data : bool, optional
        If False (default), flagged samples are NaN and no "<var>_qc"
        variable is returned. If True, every variable keeps its true
        reading (flagged or not), with "<var>_qc" variables kept for
        diagnostics. Defaults to False.

    Returns
    -------
    ds : xr.Dataset
        Level 2 dataset
    """
    ds = L1.copy()

    # Flag persistence outliers (data itself is untouched)
    ds = persistence_qc(ds)

    # Flag high-rate-of-change outliers (data itself is untouched)
    ds = rate_of_change_filter(ds)

    try:
        # Adjust time after a user-defined csv files
        ds = adjustTime(ds, adj_dir=data_adjustments_dir.as_posix())

        # Flag NaNs after a user-defined csv files (data itself is untouched)
        ds = flagNAN(ds, flag_dir=data_flags_dir.as_posix())

        # Adjust data after a user-defined csv files
        ds = adjustData(ds, adj_dir=data_adjustments_dir.as_posix())

    except Exception:
        logger.exception("Flagging and fixing failed:")

    # if ds.attrs['format'] == 'TX':
    #     # TODO: The configuration should be provided explicitly
    #     outlier_detector = ThresholdBasedOutlierDetector.default()
    #     ds = outlier_detector.filter_data(ds)

    # From here on, every raw sensor reading is consumed through this clean
    # view (flagged samples as NaN) rather than `ds` directly, so derived
    # variables are never computed from a value some QC step has already
    # rejected -- while `ds` itself keeps every raw variable's true,
    # unflagged reading available for the whole rest of the pipeline.
    ds_clean = clean_view(ds)

    # Filter GPS values based on baseline elevation
    ds["gps_lat"], ds["gps_lon"], ds["gps_alt"] = gps.filter(ds_clean["gps_lat"],
                                                             ds_clean["gps_lon"],
                                                             ds_clean["gps_alt"])

    # Calculate relative humidity with regard to ice
    ds["rh_u_wrt_ice_or_water"] = humidity.adjust(ds_clean["rh_u"], ds_clean["t_u"])

    if ds.attrs["number_of_booms"]==2:
        ds["rh_l_wrt_ice_or_water"] = humidity.adjust(ds_clean["rh_l"], ds_clean["t_l"])

    if hasattr(ds,"t_i"):
        if ~ds["t_i"].isnull().all():
            ds["rh_i_wrt_ice_or_water"] = humidity.adjust(ds_clean["rh_i"], ds_clean["t_i"])

    # Determine surface temperature
    ds["t_surf"] = radiation.calculate_surface_temperature(ds_clean["dlr"],
                                                           ds_clean["ulr"])
    is_bedrock = ds.attrs["bedrock"]
    if not is_bedrock:
        ds["t_surf"] = ds["t_surf"].clip(max=0)

    # Interpolate and smooth station tilt
    # TODO tilt smoothing is performed here and at L0toL1 also (and they are different functions). Is this needed? PHO
    ds['tilt_x'] = station_pose.interpolate_tilt(ds_clean['tilt_x'])
    ds['tilt_y'] = station_pose.interpolate_tilt(ds_clean['tilt_y'])

    # Determine cloud cover for on-ice stations
    if not is_bedrock:

        # Selected stations have pre-defined cloud assumption coefficients
        # TODO Ideally these will be pre-defined for all stations eventually
        if ds.attrs["station_id"] == "KAN_M":
            LR_overcast = 315 + 4 * ds_clean["t_u"]
            LR_clear = 30 + 4.6e-13 * (ds_clean["t_u"] + air_temperature.T_0) ** 6
        elif ds.attrs["station_id"] == "KAN_U":
            LR_overcast = 305 + 4 * ds_clean["t_u"]
            LR_clear = 220 + 3.5 * ds_clean["t_u"]

        # Else, calculate cloud assumption coefficients based on default values
        else:
            LR_overcast, LR_clear = air_temperature.get_cloud_coefficients(ds_clean["t_u"])

        ds["cc"] = radiation.calculate_cloud_coverage(ds_clean["dlr"], LR_overcast, LR_clear)

    # Set cloud cover to nans if station is not on ice
    else:
        ds["cc"] = xr.full_like(ds_clean["dlr"], np.nan)

    # Determine station pose relative to sun position
    if hasattr(ds, 'latitude') and hasattr(ds, 'longitude'):
        lat = ds.attrs['latitude']
        lon = ds.attrs['longitude']
    else:
        lat = ds['gps_lat'].mean()
        lon = ds['gps_lon'].mean()

    # Calculate spherical tilt
    phi_sensor_rad, theta_sensor_rad = station_pose.calculate_spherical_tilt(ds['tilt_x'],
                                                                             ds['tilt_y'])

    # Determine station position relative to sun
    doy = ds['time'].dt.dayofyear
    hour = ds['time'].dt.hour
    minute = ds['time'].dt.minute
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
    ds["dsr"], ds["usr"], _ = radiation.filter_sr(ds_clean["dsr"],
                                                  ds_clean["usr"],
                                                  ds["cc"],
                                                  ZenithAngle_rad,
                                                  ZenithAngle_deg,
                                                  AngleDif_deg)

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

    if ~ds_clean["precip_u"].isnull().all() and precip_flag:
        ds["precip_u"] = precipitation.filter_lufft_errors(ds_clean["precip_u"], ds_clean["t_u"], ds_clean["p_u"], ds_clean["rh_u"])
        ds["rainfall_u"] = precipitation.get_rainfall_per_timestep(ds["precip_u"], ds_clean["t_u"])
        ds["rainfall_cor_u"] = precipitation.correct_rainfall_undercatch(ds["rainfall_u"], ds_clean["wspd_u"])

    if ds.attrs["number_of_booms"]==2:
        if ~ds_clean["precip_l"].isnull().all() and precip_flag:
            ds["precip_l"] = precipitation.filter_lufft_errors(ds_clean["precip_l"], ds_clean["t_l"], ds_clean["p_l"], ds_clean["rh_l"])
            ds["rainfall_l"] = precipitation.get_rainfall_per_timestep(ds["precip_l"], ds_clean["t_l"])
            ds["rainfall_cor_l"] = precipitation.correct_rainfall_undercatch(ds["rainfall_l"], ds_clean["wspd_l"])

    # Calculate directional wind speed for upper boom
    ds['wdir_u'] = wind.filter_wind_direction(ds_clean['wdir_u'], ds_clean['wspd_u'])
    ds['wspd_x_u'], ds['wspd_y_u'] = wind.calculate_directional_wind_speed(ds_clean['wspd_u'], ds['wdir_u'])

    # Calculate directional wind speed for lower boom
    if ds.attrs['number_of_booms'] == 2:
        ds['wdir_l'] = wind.filter_wind_direction(ds_clean['wdir_l'], ds_clean['wspd_l'])
        ds['wspd_x_l'], ds['wspd_y_l'] = wind.calculate_directional_wind_speed(ds_clean['wspd_l'], ds['wdir_l'])

    # Calculate directional wind speed for instantaneous measurements
    if hasattr(ds, 'wdir_i'):
        if ~ds_clean['wdir_i'].isnull().all() and ~ds_clean['wspd_i'].isnull().all():
            ds['wdir_i'] = wind.filter_wind_direction(ds_clean['wdir_i'], ds_clean['wspd_i'])
            ds['wspd_x_i'], ds['wspd_y_i'] = wind.calculate_directional_wind_speed(ds_clean['wspd_i'], ds['wdir_i'])

    # Clip values (i.e. threshold filtering)
    ds = clip_values(ds, vars_df)

    # Finalize QC: by default, NaN out flagged samples and drop the
    # "<var>_qc" variables; with keep_flagged_data=True, return every
    # variable's true reading (flagged or not) with its "<var>_qc" kept.
    ds = finalize_qc(ds, keep_flagged_data=keep_flagged_data)

    # Return L2 dataset
    ds.attrs['level'] = 'L2'
    return ds


if __name__ == "__main__":
    pass
