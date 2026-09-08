# __all__ = ["adjust", "include_uncorrected_values"]

import pandas as pd
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from pypromice.core.qc.github_data_issues import adjustData
import logging
logger = logging.getLogger(__name__)

def process_surface_height(ds, data_adjustments_dir, station_config={}):
    """
    Process surface height data for different site types and create
    surface height variables.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing various measurements and attributes including
        'site_type' which determines the type of site (e.g., 'ablation',
        'accumulation', 'bedrock') and other relevant data variables such as
        'z_boom_u', 'z_stake', 'z_pt_cor', etc.

    Returns
    -------
    xarray.Dataset
        The dataset with additional processed surface height variables:
        'z_surf_1', 'z_surf_2', 'z_ice_surf', 'z_surf_combined', 'snow_height',
        and possibly depth variables derived from temperature measurements.
    """
    # Initialize surface height variables with NaNs
    ds['z_surf_1'] = ('time', ds['z_boom_u'].data * np.nan)
    ds['z_surf_2'] = ('time', ds['z_boom_u'].data * np.nan)

    z_boom_best_u = ds["z_boom_cor_u"]
    # z_boom_best_u = station_boom_height.include_uncorrected_values(
    #                             ds["z_boom_u"],
    #                             ds["z_boom_cor_u"],
    #                             ds["t_l"] if "t_l" in ds.data_vars else None,
    #                             ds["t_rad"] if "t_rad" in ds.data_vars else None)



    if 'z_stake' in ds.data_vars and ds.z_stake.notnull().any():
        # Calculate stake boom height correction with uncorrected values where needed
        z_stake_best = ds["z_stake_cor"]
        # z_stake_best = station_boom_height.include_uncorrected_values(
        #                             ds["z_stake"],
        #                             ds["z_stake_cor"],
        #                             ds["t_l"] if "t_l" in ds.data_vars else None,
        #                             ds["t_rad"] if "t_rad" in ds.data_vars else None)

    if ds.attrs['site_type'] == 'ablation':
        # Calculate surface heights for ablation sites
        ds['z_surf_1'] = 2.6 - z_boom_best_u
        if ds.z_stake.notnull().any():
            first_valid_index = ds.time.where((z_stake_best + z_boom_best_u).notnull(), drop=True).data[0]
            ds['z_surf_2'] = ds.z_surf_1.sel(time=first_valid_index) + z_stake_best.sel(time=first_valid_index) - z_stake_best

        # Use corrected point data if available
        if 'z_pt_cor' in ds.data_vars:
            ds['z_ice_surf'] = ('time', ds['z_pt_cor'].data)

    else:
        # Calculate surface heights for other site types
        first_valid_index = ds.time.where(z_boom_best_u.notnull(), drop=True).data[0]
        ds['z_surf_1'] = z_boom_best_u.sel(time=first_valid_index) - z_boom_best_u

        if 'z_stake' in ds.data_vars and ds.z_stake.notnull().any():
            first_valid_index = ds.time.where(z_stake_best.notnull(), drop=True).data[0]
            ds['z_surf_2'] = z_stake_best.sel(time=first_valid_index) - z_stake_best

        if 'z_boom_l' in ds.data_vars:

            # Calculate lower boom height correction with uncorrected values where needed
            z_boom_best_l = ds["z_boom_cor_l"]
            # z_boom_best_l = station_boom_height.include_uncorrected_values(
            #                             ds["z_boom_l"],
            #                             ds["z_boom_cor_l"],
            #                             ds["t_u"] if "t_u" in ds.data_vars else None,
            #                             ds["t_rad"] if "t_rad" in ds.data_vars else None)

            # need a combine first because KAN_U switches from having a z_stake_best
            # to having a z_boom_best_l
            first_valid_index = ds.time.where(z_boom_best_l.notnull(), drop=True).data[0]
            ds['z_surf_2'] = ds['z_surf_2'].combine_first(
                z_boom_best_l.sel(time=first_valid_index) - z_boom_best_l)

    # Adjust data for the created surface height variables
    ds = adjustData(ds, data_adjustments_dir, var_list=['z_surf_1', 'z_surf_2', 'z_ice_surf'])

    # Convert to dataframe and combine surface height variables
    df_in = ds[[v for v in ['z_surf_1', 'z_surf_2', 'z_ice_surf'] if v in ds.data_vars]].to_dataframe()

    (ds['z_surf_combined'], ds['z_ice_surf'],
     ds['z_surf_1_adj'], ds['z_surf_2_adj']) = combine_surface_height(df_in, ds.attrs['site_type'])


    if ds.attrs['site_type'] == 'ablation':
        # post processing of ice surface height (rolling median smoothing + gap-fill)
        z_ice_surf = post_processing_z_ice_surf(
            ds['z_ice_surf'], ds['z_surf_combined'], ds['z_surf_2_adj']
        )
        ds['z_ice_surf'] = ('time', z_ice_surf.values)

        ds['z_surf_combined'] = np.maximum(ds['z_surf_combined'], ds['z_ice_surf'])
        ds['snow_height'] = np.maximum(0, ds['z_surf_combined'] - ds['z_ice_surf'])
        ds['z_ice_surf'] = ds['z_ice_surf'].where(ds.snow_height.notnull())
    elif ds.attrs['site_type'] in ['accumulation', 'bedrock']:
        # Handle accumulation and bedrock site types
        ds['z_ice_surf'] = ('time', ds['z_surf_1'].data * np.nan)
        ds['snow_height'] = ds['z_surf_combined']
    else:
        # Log info for other site types
        logger.info('other site type')

    return ds



def find_ablation_periods(df, threshold_ablation, min_period="2D",
                       max_gap="60D", shift_threshold=3,
                       smooth_window="14D", interp_limit=72):
    """Find ice-ablation periods from pressure-transducer surface height.

    Spurious step changes in ``z_ice_surf`` are removed before the time series
    is interpolated and smoothed. Ablation is initially detected from the
    derivative of the smoothed surface height and the beginning and end of each
    ablation season are refined using the curvature of the smoothed series.

    Args:
        df (pd.DataFrame): DataFrame containing ``z_ice_surf`` and a
            DatetimeIndex.
        threshold_ablation (float): Threshold applied to the hourly change in
            smoothed surface height to identify ablation.
        min_period (str, optional): Minimum duration of an ablation period.
            Defaults to ``"2D"``.
        max_gap (str, optional): Maximum gap between ablation periods that is
            filled. Defaults to ``"60D"``.
        shift_threshold (float, optional): Surface-height step larger than
            this value is considered spurious and removed. Defaults to 3 m.
        smooth_window (str, optional): Window used for each of the two
            centered smoothing operations. Defaults to ``"14D"``.
        interp_limit (int, optional): Maximum number of consecutive hourly
            values to interpolate. Defaults to 72.

    Returns:
        tuple: ``(ind_ablation, smoothed_PT, diff_series, ddiff_series,
        z_corrected)``.

    """
    # remove spurious shifts in z_ice_surf
    z = df["z_ice_surf"].copy()
    dz = z.diff()
    dz_clean = dz.mask(dz.abs() > shift_threshold, 0)

    first = z.first_valid_index()
    z_corrected = z.loc[first] + dz_clean.loc[first:].fillna(0).cumsum()
    z_corrected = z_corrected.reindex(df.index)

    # smoothing
    hourly_interp = z_corrected.resample("h").interpolate(limit=interp_limit)
    once_smoothed = hourly_interp.rolling(
        smooth_window, center=True, min_periods=1
    ).mean()
    smoothed_PT = once_smoothed.rolling(
        smooth_window, center=True, min_periods=1
    ).mean()

    # first and second derivatives
    diff_series = (smoothed_PT.shift(-1) - smoothed_PT.shift(1)) / 2
    ddiff_series = (diff_series.shift(-1) - diff_series.shift(1)) / 2

    # initial ablation detection
    ind_ablation = (
        (diff_series.values < threshold_ablation) &
        np.isin(diff_series.index.month, [6, 7, 8, 9]) &
        ~np.isnan(smoothed_PT.values)
    )

    # reindex back to df
    ind_ablation = (
        pd.Series(ind_ablation, index=diff_series.index)
        .reindex(df.index, method="ffill")
        .fillna(False)
        .to_numpy(dtype=bool, copy=True)
    )

    # remove short spurious ablation periods
    idx = np.argwhere(
        np.diff(np.r_[False, ind_ablation, False])
    ).reshape(-1, 2)
    idx[:, 1] -= 1

    for start, end in idx:
        if df.index[end] - df.index[start] < pd.Timedelta(min_period):
            ind_ablation[start:end + 1] = False

    # fill small gaps in the ice ablation periods
    idx = np.argwhere(
        np.diff(np.r_[False, ind_ablation, False])
    ).reshape(-1, 2)
    idx[:, 1] -= 1

    for i in range(len(idx) - 1):
        if (df.index[idx[i + 1, 0]] - df.index[idx[i, 1]]
                < pd.Timedelta(max_gap)):
            ind_ablation[idx[i, 1] + 1:idx[i + 1, 0]] = True

    # redefine start and end from curvature
    idx = np.argwhere(
        np.diff(np.r_[False, ind_ablation, False])
    ).reshape(-1, 2)
    idx[:, 1] -= 1

    for start, end in idx:
        rough_start = df.index[start]
        rough_end = df.index[end]

        search_start = ddiff_series.loc[
            rough_start - pd.Timedelta("10D"):
            rough_start + pd.Timedelta("30D")
        ].dropna()

        if len(search_start):
            start_date = search_start.idxmin()
            ind_ablation[
                (df.index >= rough_start) &
                (df.index < start_date)
            ] = False

        search_end = ddiff_series.loc[
            rough_end - pd.Timedelta("30D"):
            rough_end + pd.Timedelta("10D")
        ].dropna()

        if len(search_end):
            end_date = search_end.idxmax()
            ind_ablation[
                (df.index > end_date) &
                (df.index <= rough_end)
            ] = False

    return ind_ablation, smoothed_PT, diff_series, ddiff_series, z_corrected

def estimate_ablation_period_by_year(df, ind_ablation):
    """Estimate the beginning and end of the ablation period for each year.

    For each year, the function identifies the first and last timestamp flagged
    as ablation. If ``z_ice_surf_adj`` is entirely missing during June, July,
    and August, the full JJA period is used as the ablation season instead.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex and a
            ``z_ice_surf_adj`` column.
        ind_ablation (np.ndarray): Boolean array indicating ablation timestamps.

    Returns:
        tuple: ``(ind_ablation, years, ind_start, ind_end)``, where
        ``ind_start`` and ``ind_end`` contain the dataframe indices of the first
        and last ablation timestamps for each year. Years without an estimated
        ablation season are assigned ``-999``.

    """
    years = df.index.year.unique().values
    ind_start = years.copy()
    ind_end = years.copy()

    logger.debug('-> estimating ablation period for each year')

    for i, y in enumerate(years):
        # for each year
        ind_yr = df.index.year.values == y
        ind_abl_yr = np.logical_and(ind_yr, ind_ablation)

        if df.loc[
                np.logical_and(ind_yr, df.index.month.isin([6, 7, 8])),
                "z_ice_surf_adj"].isnull().all():

            ind_abl_yr = np.logical_and(
                ind_yr,
                df.index.month.isin([6, 7, 8])
            )
            ind_ablation[ind_yr] = ind_abl_yr[ind_yr]
            logger.debug(str(y) + ' no z_ice_surf, just using JJA')

        else:
            logger.debug(str(y) + ' derived from z_ice_surf')

        if np.any(ind_abl_yr):
            # if there are some ablation flagged for that year
            # then find begining and end
            ind_start[i] = np.argwhere(ind_abl_yr)[0][0]
            ind_end[i] = np.argwhere(ind_abl_yr)[-1][0]

        else:
            logger.debug(str(y) + ' could not estimate ablation season')
            # otherwise left as nan
            ind_start[i] = -999
            ind_end[i] = -999

    return ind_ablation, years, ind_start, ind_end

def align_surface_heights(hs1, hs2, z):
    """Align surface-height time series to a common relative reference.

    The surface-height series are shifted so that they are expressed relative
    to the beginning of the record. The pressure-transducer series is aligned
    either from its first-week mean, from the surface-height record when it was
    installed later, or from a fitted linear trend when needed.

    Args:
        hs1 (pd.Series): Primary surface-height time series.
        hs2 (pd.Series): Secondary surface-height time series.
        z (pd.Series): Pressure-transducer-derived ice surface height.

    Returns:
        tuple: Adjusted ``(hs1, hs2, z)`` series.

    """
    # the surface heights are adjusted so that they start at 0
    if any(~np.isnan(hs2.iloc[:24*7])):
        hs2 = hs2 - hs2.iloc[:24*7].mean()

    if any(~np.isnan(hs2.iloc[:24*7])) & any(~np.isnan(hs1.iloc[:24*7])):
        hs2 = hs2 + hs1.iloc[:24*7].mean() - hs2.iloc[:24*7].mean()

    if any(~np.isnan(z.iloc[:24*7])):
        # expressing ice surface height relative to its mean value in the
        # first week of the record
        z = z - z.iloc[:24*7].mean()

    elif z.notnull().any():
        # if there is no data in the first week but that there are some
        # PT data afterwards
        if ((z.first_valid_index() - hs1.first_valid_index()) < pd.to_timedelta('251D')) & \
           ((z.first_valid_index() - hs1.first_valid_index()) > pd.to_timedelta('0H')):

            # if the pressure transducer is installed the year after then
            # we use the mean surface height 1 on its first week as a 0
            # for the ice height
            z = (
                z
                - z.loc[
                    z.first_valid_index():
                    (z.first_valid_index() + pd.to_timedelta('14D'))
                ].mean()
                + hs1.loc[
                    hs1.first_valid_index():
                    (hs1.first_valid_index() + pd.to_timedelta('7D'))
                ].mean()
            )

        elif (z.last_valid_index() - z.first_valid_index()) > pd.to_timedelta('200D'):
            # if there is more than a year (actually 251 days) between the
            # initiation of the AWS and the installation of the pressure transducer
            # we remove the intercept in the pressure transducer data.
            # Removing the intercept
            # means that we consider the ice surface height at 0 when the AWS
            # is installed, and not when the pressure transducer is installed.
            Y = z.values.reshape(-1, 1)
            X = z.iloc[~np.isnan(Y)].index.astype(np.int64).values.reshape(-1, 1)
            Y = Y[~np.isnan(Y)]

            linear_regressor = LinearRegression()
            linear_regressor.fit(X, Y)

            Y_pred = linear_regressor.predict(
                z.index.astype(np.int64).values.reshape(-1, 1)
            )

            z = z - Y_pred[0]

    return hs1, hs2, z

def combine_surface_height(df, site_type, threshold_ablation = -0.0002):
    '''Combines the data from three sensor: the two sonic rangers and the
    pressure transducer, to recreate the surface height, the ice surface height
    and the snow depth through the years. For the accumulation sites, it is
    only the average of the two sonic rangers (after manual adjustments to
    correct maintenance shifts). For the ablation sites, first an ablation
    period is estimated each year (either the period when z_pt_cor decreases
    or JJA if no better estimate) then different adjustmnents are conducted
    to stitch the three time series together: z_ice_surface (adjusted from
    z_pt_cor) or if unavailable, z_surf_2 (adjusted from z_stake)
    are used in the ablation period while an average of z_surf_1 and z_surf_2
    are used otherwise, after they are being adjusted to z_ice_surf at the end
    of the ablation season.

    Parameters
    ----------
    df : pandas.dataframe
        Dataframe with datetime index and variables z_surf_1, z_surf_2 and z_ice_surf
    site_type : str
        Either 'accumulation' or 'ablation'
    threshold_ablation : float
        Threshold to which a z_pt_cor hourly decrease is compared. If the decrease
        is higher, then there is ablation.
    '''
    logger.info('Combining surface height')

    if 'z_surf_2' not in df.columns:
        logger.info('-> did not find z_surf_2')
        df["z_surf_2"] = df["z_surf_1"].values*np.nan

    if 'z_ice_surf' not in df.columns:
        logger.info('-> did not find z_ice_surf')
        df["z_ice_surf"] = df["z_surf_1"].values*np.nan

    if site_type in ['accumulation', 'bedrock']:
        logger.info('-> no z_pt or accumulation site: averaging z_surf_1 and z_surf_2')
        df["z_surf_1_adj"] = hampel(df["z_surf_1"].interpolate(limit=72)).values
        df["z_surf_2_adj"] = hampel(df["z_surf_2"].interpolate(limit=72)).values
        # adjusting z_surf_2 to z_surf_1
        df["z_surf_2_adj"]  = df["z_surf_2_adj"]  + (df["z_surf_1_adj"]- df["z_surf_2_adj"]).mean()
        # z_surf_combined is the average of the two z_surf
        if df.z_surf_1_adj.notnull().any() & df.z_surf_2_adj.notnull().any():
            df['z_surf_combined'] = df[['z_surf_1_adj', 'z_surf_2_adj']].mean(axis = 1).values
        elif df.z_surf_1_adj.notnull().any():
            df['z_surf_combined'] = df.z_surf_1_adj.values
        elif df.z_surf_2_adj.notnull().any():
            df['z_surf_combined'] = df.z_surf_2_adj.values

        # df["z_surf_combined"] = hampel(df["z_surf_combined"].interpolate(limit=72)).values
        return (df['z_surf_combined'], df["z_surf_combined"]*np.nan,
                    df["z_surf_1_adj"], df["z_surf_2_adj"])

    else:
        logger.info('-> ablation site')
        # smoothing and filtering pressure transducer data
        df["z_ice_surf_adj"] = hampel(df["z_ice_surf"].interpolate(limit=72)).values
        df["z_surf_1_adj"] = hampel(df["z_surf_1"].interpolate(limit=72)).values
        df["z_surf_2_adj"] = hampel(df["z_surf_2"].interpolate(limit=72)).values

        df["z_surf_1_adj"] = hampel(df["z_surf_1"].interpolate(limit=72), k=24, t0=5).values
        df["z_surf_2_adj"] = hampel(df["z_surf_2"].interpolate(limit=72), k=24, t0=5).values

        ind_ablation, _, _, _, _ =  find_ablation_periods(
                        df,  threshold_ablation=threshold_ablation)

        hs1=df["z_surf_1_adj"].interpolate(limit=24*2).copy()
        hs2=df["z_surf_2_adj"].interpolate(limit=24*2).copy()
        z=df["z_ice_surf_adj"].interpolate(limit=24*2).copy()

        # align the two surface-height records and the pressure-transducer ice height
        hs1, hs2, z = align_surface_heights(hs1, hs2, z)

        # finding start/end index of ablation period for each year
        ind_ablation, years, ind_start, ind_end = \
            estimate_ablation_period_by_year(df, ind_ablation)

        # adjustement loop
        missing_hs2 = 0 # if hs2 is missing then when it comes back it is adjusted to hs1
        hs2_ref = 0 # by default, the PT is the reference: hs1 and 2 will be adjusted to PT
        # but if it is missing one year or one winter, then it needs to be rajusted
        # to hs1 and hs2 the year after.


        for i, y in enumerate(years):
            logger.debug(f'{y}: Ablation from {z.index[ind_start[i]]} to {z.index[ind_end[i]]}')

            # defining subsets of hs1, hs2, z
            hs1_jja =  hs1[str(y)+'-06-01':str(y)+'-09-01']
            hs2_jja =  hs2[str(y)+'-06-01':str(y)+'-09-01']
            z_jja =  z[str(y)+'-06-01':str(y)+'-09-01']

            z_ablation = z.iloc[ind_start[i]:ind_end[i]]
            hs2_ablation = hs2.iloc[ind_start[i]:ind_end[i]]

            hs1_year = hs1[str(y)]
            hs2_year = hs2[str(y)]

            hs2_winter = hs2[str(y)+'-01-01':str(y)+'-03-01'].copy()
            z_winter = z[str(y)+'-01-01':str(y)+'-03-01'].copy()

            z_year = z[str(y)]
            if hs1_jja.isnull().all() and hs2_jja.isnull().all() and z_jja.isnull().all():
                    # if there is no height for a year between June and September
                    # then the adjustment cannot be made automatically
                    # it needs to be specified manually on the adjustment files
                    # on https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-data-issues
                    continue

            if all(np.isnan(z_jja)) and any(~np.isnan(hs2_jja)):
                # if there is no PT for a given year, but there is some hs2
                # then z will be adjusted to hs2 next time it is available
                hs2_ref = 1

            if all(np.isnan(z_winter)) and all(np.isnan(hs2_winter)):
                # if there is no PT nor hs2 during the winter, then again
                # we need to adjust z to match hs2 when ablation starts
                hs2_ref = 1

            # adjustment at the start of the ablation season
            if hs2_ref:
                # if hs2 has been taken as reference in the previous years
                # then we check if pressure transducer is reinstalled and needs
                # to be adjusted
                if ind_start[i] != -999:
                    # the first year there is both ablation and PT data available
                    # then PT is adjusted to hs2
                    if any(~np.isnan(z_ablation)) and any(~np.isnan(hs2_ablation)):
                        tmp1 = z_ablation.copy()
                        tmp2 = hs2_ablation.copy()
                        # tmp1[np.isnan(tmp2)] = np.nan
                        # tmp2[np.isnan(tmp1)] = np.nan

                        # in some instances, the PT data is available but no ablation
                        # is recorded, then hs2 remains the reference during that time.
                        # When eventually there is ablation, then we need to find the
                        # first index in these preceding ablation-free years
                        # the shift will be applied back from this point
                        # first_index = z[:z[str(y)].first_valid_index()].isnull().iloc[::-1].idxmax()
                        # z[first_index:] = z[first_index:] -  np.nanmean(tmp1)  +  np.nanmean(tmp2)
                        # hs2_ref = 0 # from now on PT is the reference

                        # in some other instance, z just need to be adjusted to hs2
                        # first_index = z[str(y)].first_valid_index()
                        first_index = z.iloc[ind_start[i]:].first_valid_index() # of ablation
                        if np.isnan(hs2[first_index]):
                            first_index_2 = hs2.iloc[ind_start[i]:].first_valid_index()
                            if (first_index_2 - first_index)>pd.Timedelta('30d'):
                                logger.debug('adjusting z to hs1')
                                if np.isnan(hs1[first_index]):
                                    first_index = hs1.iloc[ind_start[i]:].first_valid_index()
                                z[first_index:] = z[first_index:] -  z[first_index]   +  hs1[first_index]
                            else:
                                logger.debug('adjusting z to hs1')
                                first_index = hs2.iloc[ind_start[i]:].first_valid_index()
                                z[first_index:] = z[first_index:] -  z[first_index]   +  hs2[first_index]
                        else:
                            logger.debug('adjusting z to hs1')
                            z[first_index:] = z[first_index:] -  z[first_index]   +  hs2[first_index]
                        hs2_ref = 0 # from now on PT is the reference


            else:
                # if z_pt is the reference and there is some ablation
                # then hs1 and hs2 are adjusted to z_pt
                if (ind_start[i] != -999) & z_year.notnull().any():
                    # calculating first index with PT, hs1 and hs2
                    first_index = z_year.first_valid_index()
                    if hs1_year.notnull().any():
                        first_index = np.max(np.array(
                            [first_index,
                             hs1_year.first_valid_index()]))
                    if hs2_year.notnull().any():
                        first_index = np.max(np.array(
                            [first_index,
                             hs2_year.first_valid_index()]))

                    # if PT, hs1 and hs2 are all nan until station is reactivated, then
                    first_day_of_year = pd.to_datetime(str(y)+'-01-01')

                    if len(z[first_day_of_year:first_index-pd.to_timedelta('1D')])>0:
                        if z[first_day_of_year:first_index-pd.to_timedelta('1D')].isnull().all() & \
                            hs1[first_day_of_year:first_index-pd.to_timedelta('1D')].isnull().all() & \
                                hs2[first_day_of_year:first_index-pd.to_timedelta('1D')].isnull().all():
                                if (~np.isnan(np.nanmean(z[first_index:first_index+pd.to_timedelta('1D')])) \
                                    and ~np.isnan(np.nanmean(hs2[first_index:first_index+pd.to_timedelta('1D')]))):
                                    logger.debug(' ======= adjusting hs1 and hs2 to z_pt')
                                    if ~np.isnan(np.nanmean(hs1[first_index:first_index+pd.to_timedelta('1D')]) ):
                                        hs1[first_index:] = hs1[first_index:] \
                                            -  np.nanmean(hs1[first_index:first_index+pd.to_timedelta('1D')])  \
                                                +  np.nanmean(z[first_index:first_index+pd.to_timedelta('1D')])
                                    if ~np.isnan(np.nanmean(hs2[first_index:first_index+pd.to_timedelta('1D')]) ):
                                        hs2[first_index:] = hs2[first_index:] \
                                            -  np.nanmean(hs2[first_index:first_index+pd.to_timedelta('1D')])  \
                                                +  np.nanmean(z[first_index:first_index+pd.to_timedelta('1D')])

            # adjustment taking place at the end of the ablation period
            if (ind_end[i] != -999):
                # if y == 2023:
                #     import pdb; pdb.set_trace()
                # if there's ablation and
                # if there are PT data available at the end of the melt season
                if z.iloc[(ind_end[i]-24*7):ind_end[i]].notnull().any():
                    logger.debug('adjusting hs2 to z')
                    # then we adjust hs2 to the end-of-ablation z
                    # first trying at the end of melt season
                    if ~np.isnan(np.nanmean(hs2.iloc[(ind_end[i]-24*7):(ind_end[i]+24*30)])):
                        logger.debug('using end of melt season')
                        hs2.iloc[ind_end[i]:] = hs2.iloc[ind_end[i]:] - \
                            np.nanmean(hs2.iloc[(ind_end[i]-24*7):(ind_end[i]+24*30)])  + \
                                np.nanmean(z.iloc[(ind_end[i]-24*7):(ind_end[i]+24*30)])
                    # if not possible, then trying the end of the following accumulation season
                    elif (i+1 < len(ind_start)):
                        if ind_start[i+1]!=-999 and \
                            any(~np.isnan(hs2.iloc[(ind_start[i+1]-24*7):(ind_start[i+1]+24*7)] \
                                          + z.iloc[(ind_start[i+1]-24*7):(ind_start[i+1]+24*7)])):
                            logger.debug('using end of accumulation season')
                            hs2.iloc[ind_end[i]:] = hs2.iloc[ind_end[i]:] - \
                                np.nanmean(hs2.iloc[(ind_start[i+1]-24*7):(ind_start[i+1]+24*7)])  + \
                                    np.nanmean(z.iloc[(ind_start[i+1]-24*7):(ind_start[i+1]+24*7)])
            else:
                logger.debug('no ablation data')
                hs1_following_winter = hs1[str(y)+'-09-01':str(y+1)+'-03-01'].copy()
                hs2_following_winter = hs2[str(y)+'-09-01':str(y+1)+'-03-01'].copy()
                if all(np.isnan(hs2_following_winter)):
                    logger.debug('no hs2')
                    missing_hs2 = 1
                elif missing_hs2 == 1:
                    logger.debug('adjusting hs2')
                    # and if there are some hs2 during the accumulation period
                    if any(~np.isnan(hs1_following_winter)):
                        logger.debug('to hs1')
                        # then we adjust hs1 to hs2 during the accumulation area
                        # adjustment is done so that the mean hs1 and mean hs2 match
                        # for the period when both are available
                        hs2_following_winter[np.isnan(hs1_following_winter)] = np.nan
                        hs1_following_winter[np.isnan(hs2_following_winter)] = np.nan

                        hs2[str(y)+'-01-01':] = hs2[str(y)+'-01-01':] \
                            -  np.nanmean(hs2_following_winter)  +  np.nanmean(hs1_following_winter)
                        missing_hs2 = 0


                hs1_following_winter = hs1[str(y)+'-09-01':str(y+1)+'-03-01'].copy()
                hs2_following_winter = hs2[str(y)+'-09-01':str(y+1)+'-03-01'].copy()
                # adjusting hs1 to hs2 (no ablation case)
                if any(~np.isnan(hs1_following_winter)):
                    logger.debug('adjusting hs1')
                    # and if there are some hs2 during the accumulation period
                    if any(~np.isnan(hs2_following_winter)):
                        logger.debug('to hs2')
                        # then we adjust hs1 to hs2 during the accumulation area
                        # adjustment is done so that the mean hs1 and mean hs2 match
                        # for the period when both are available
                        hs1_following_winter[np.isnan(hs2_following_winter)] = np.nan
                        hs2_following_winter[np.isnan(hs1_following_winter)] = np.nan

                        hs1[str(y)+'-09-01':] = hs1[str(y)+'-09-01':] \
                            -  np.nanmean(hs1_following_winter)  +  np.nanmean(hs2_following_winter)
                        hs1_following_winter = hs1[str(y)+'-09-01':str(y+1)+'-03-01'].copy()

            if ind_end[i] != -999:
                # if there is some hs1
                hs1_following_winter = hs1[str(y)+'-09-01':str(y+1)+'-03-01'].copy()
                hs2_following_winter = hs2[str(y)+'-09-01':str(y+1)+'-03-01'].copy()
                if any(~np.isnan(hs1_following_winter)):
                    logger.debug('adjusting hs1')
                    # and if there are some hs2 during the accumulation period
                    if any(~np.isnan(hs2_following_winter)):
                        logger.debug('to hs2, minimizing winter difference')
                        # then we adjust hs1 to hs2 during the accumulation area
                        # adjustment is done so that the mean hs1 and mean hs2 match
                        # for the period when both are available
                        tmp1 = hs1.iloc[ind_end[i]:min(len(hs1),ind_end[i]+24*30*9)].copy()
                        tmp2 = hs2.iloc[ind_end[i]:min(len(hs2),ind_end[i]+24*30*9)].copy()

                        tmp1[np.isnan(tmp2)] = np.nan
                        tmp2[np.isnan(tmp1)] = np.nan
                        if tmp1.isnull().all():
                            tmp1 = hs1_following_winter.copy()
                            tmp2 = hs2_following_winter.copy()

                            tmp1[np.isnan(tmp2)] = np.nan
                            tmp2[np.isnan(tmp1)] = np.nan
                        hs1.iloc[ind_end[i]:] = hs1.iloc[ind_end[i]:] -  np.nanmean(tmp1)  +  np.nanmean(tmp2)

                    # if no hs2, then use PT data available at the end of the melt season
                    elif np.any(~np.isnan(z.iloc[(ind_end[i]-24*14):(ind_end[i]+24*7)])):
                        logger.debug('to z')
                        # then we adjust hs2 to the end-of-ablation z
                        # first trying at the end of melt season
                        if ~np.isnan(np.nanmean(hs1.iloc[(ind_end[i]-24*14):(ind_end[i]+24*30)])):
                            logger.debug('using end of melt season')
                            hs1.iloc[ind_end[i]:] = hs1.iloc[ind_end[i]:] - \
                                np.nanmean(hs1.iloc[(ind_end[i]-24*14):(ind_end[i]+24*30)])  + \
                                    np.nanmean(z.iloc[(ind_end[i]-24*14):(ind_end[i]+24*30)])
                        # if not possible, then trying the end of the following accumulation season
                        elif ind_start[i+1]!=-999 and any(~np.isnan(hs1.iloc[(ind_start[i+1]-24*14):(ind_start[i+1]+24*7)]+ z.iloc[(ind_start[i+1]-24*14):(ind_start[i+1]+24*7)])):
                            logger.debug('using end of accumulation season')
                            hs1.iloc[ind_end[i]:] = hs1.iloc[ind_end[i]:] - \
                                np.nanmean(hs1.iloc[(ind_start[i+1]-24*14):(ind_start[i+1]+24*7)])  + \
                                    np.nanmean(z.iloc[(ind_start[i+1]-24*14):(ind_start[i+1]+24*7)])
                    elif any(~np.isnan(hs2_year)):
                        logger.debug('to the last value of hs2')
                        # then we adjust hs1 to hs2 during the accumulation area
                        # adjustment is done so that the mean hs1 and mean hs2 match
                        # for the period when both are available
                        half_span = pd.to_timedelta('7D')
                        tmp1 = hs1_year.loc[(hs2_year.last_valid_index()-half_span):(hs2_year.last_valid_index()+half_span)].copy()
                        tmp2 = hs2_year.loc[(hs2_year.last_valid_index()-half_span):(hs2_year.last_valid_index()+half_span)].copy()

                        hs1.iloc[ind_end[i]:] = hs1.iloc[ind_end[i]:] -  np.nanmean(tmp1)  +  np.nanmean(tmp2)

        df["z_surf_1_adj"] = hs1.interpolate(limit=2*24).values
        df["z_surf_2_adj"] = hs2.interpolate(limit=2*24).values
        df["z_ice_surf_adj"] = z.interpolate(limit=2*24).values

        # making a summary of the surface height
        df["z_surf_combined"] = np.nan

        # in winter, both SR1 and SR2 are used
        df["z_surf_combined"] = df["z_surf_2_adj"].interpolate(limit=72).values

        # in ablation season we use SR2 instead of the SR1&2 average
        # here two options:
        # 1) we ignore the SR1 and only use SR2
        # 2) we use SR1 when SR2 is not available (commented)
        # the later one can cause jumps when SR2 starts to be available few days after SR1
        data_update = df[["z_surf_1_adj", "z_surf_2_adj"]].mean(axis=1).values

        ind_update = ~ind_ablation
        #ind_update = np.logical_and(ind_ablation,  ~np.isnan(data_update))
        df.loc[ind_update,"z_surf_combined"] = data_update[ind_update]

        # in ablation season we use pressure transducer over all other options
        data_update = df[ "z_ice_surf_adj"].interpolate(limit=72).values
        ind_update = np.logical_and(ind_ablation, ~np.isnan(data_update))
        df.loc[ind_update,"z_surf_combined"] = data_update[ind_update]

    logger.info('surface height combination finished')
    return df['z_surf_combined'], df["z_ice_surf_adj"], df["z_surf_1_adj"], df["z_surf_2_adj"]

def hampel(vals_orig, k=7*24, t0=15):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    #Make copy so original not edited
    vals=vals_orig.copy()
    #Hampel Filter
    L= 1.4826
    rolling_median=vals.rolling(k).median()
    difference=np.abs(rolling_median-vals)
    median_abs_deviation=difference.rolling(k).median()
    threshold= t0 *L * median_abs_deviation
    outlier_idx=difference>threshold
    outlier_idx[0:round(k/2)]=False
    vals.loc[outlier_idx]=np.nan
    return(vals)


def post_processing_z_ice_surf(z_ice_surf, z_surf_combined, z_surf_2_adj):
    '''Smooth ice surface height (z_ice_surf) with a rolling median and fill
    short gaps, for ablation-type sites. z_surf_combined is used as a fallback
    where z_ice_surf is missing, and its time axis is the target grid the
    result is reindexed onto (so this can be re-run on a merged, multi-station
    series, not just a single station's own dataset).

    Parameters
    ----------
    z_ice_surf : xr.DataArray
        Ice surface height
    z_surf_combined : xr.DataArray
        Combined surface height, used as fallback and as the target time axis
    z_surf_2_adj : xr.DataArray
        Adjusted pressure-transducer-derived surface height, used to determine
        where z_ice_surf should remain missing rather than being gap-filled

    Returns
    -------
    pandas.Series
        Smoothed and gap-filled ice surface height, indexed on z_surf_combined.time
    '''
    # here we make sure that the periods where both z_stake_best and z_pt are
    # missing are also missing in z_ice_surf
    msk = z_ice_surf.notnull() | z_surf_2_adj.notnull()

    # Calculate rolling minimum for ice surface height and snow height
    ts_interpolated = np.minimum(
        xr.where(z_ice_surf.notnull(),
                 z_ice_surf, z_surf_combined),
        z_surf_combined).to_series().resample('h').interpolate(limit=72)

    if len(ts_interpolated)>24*7:
        # Apply the rolling window with median calculation
        z_ice_surf = (ts_interpolated
                      .rolling('14D', center=True, min_periods=1)
                      .median())
        # Overprint the first and last 7 days with interpolated values
        # because of edge effect of rolling windows
        z_ice_surf.iloc[:24*7] = (ts_interpolated.iloc[:24*7]
                                  .rolling('1D', center=True, min_periods=1)
                                  .median().values)
        z_ice_surf.iloc[-24*7:] = (ts_interpolated.iloc[-24*7:]
                                   .rolling('1D', center=True, min_periods=1)
                                   .median().values)
    else:
        z_ice_surf = (ts_interpolated
                                   .rolling('1D', center=True, min_periods=1)
                                   .median())

    z_ice_surf = z_ice_surf.reindex(z_surf_combined.time,
                                    method=None).interpolate(method='time')

    # removing from z_ice_surf the periods where both z_stake_best and z_pt are missing
    z_ice_surf = z_ice_surf.where(msk)

    # taking running minimum to get ice
    z_ice_surf = z_ice_surf.cummin()

    # filling gaps only if they are less than a year long and if values on both
    # sides are less than 0.01 m appart

    # Forward and backward fill to identify bounds of gaps
    df_filled = z_ice_surf.ffill().bfill()

    # Identify gaps and their start and end dates
    gaps = pd.DataFrame(index=z_ice_surf[z_ice_surf.isna()].index)
    gaps['prev_value'] = df_filled.shift(1)
    gaps['next_value'] = df_filled.shift(-1)
    gaps['gap_start'] = gaps.index.to_series().shift(1)
    gaps['gap_end'] = gaps.index.to_series().shift(-1)
    gaps['gap_duration'] = (gaps['gap_end'] - gaps['gap_start']).dt.days
    gaps['value_diff'] = (gaps['next_value'] - gaps['prev_value']).abs()

    # Determine which gaps to fill
    mask = (gaps['gap_duration'] < 365) & (gaps['value_diff'] < 0.01)
    gaps_to_fill = gaps[mask].index

    # Fill gaps in the original Series
    z_ice_surf.loc[gaps_to_fill] = df_filled.loc[gaps_to_fill]
    return z_ice_surf
