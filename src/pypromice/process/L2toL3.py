#!/usr/bin/env python
"""
AWS Level 2 (L2) to Level 3 (L3) data processing
"""
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from pypromice.qc.github_data_issues import adjustData
from scipy.interpolate import interp1d
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def toL3(L2,
         data_adjustments_dir: Path,
         station_config={},
         T_0=273.15):
    '''Process one Level 2 (L2) product to Level 3 (L3) meaning calculating all
    derived variables:
        - Turbulent fluxes
        - smoothed and inter/extrapolated GPS coordinates
        - continuous surface height, ice surface height, snow height
        - thermistor depths


    Parameters
    ----------
    L2 : xarray:Dataset
        L2 AWS data
    station_config : Dict
        Dictionary containing the information necessary for the processing of
        L3 variables (relocation dates for coordinates processing, or thermistor
        string maintenance date for the thermistors depth)
    T_0 : int
        Freezing point temperature. Default is 273.15.
    '''
    ds = L2
    ds.attrs['level'] = 'L3'

    T_100 = T_0+100                                                            # Get steam point temperature as K

    # Turbulent heat flux calculation
    if ('t_u' in ds.keys()) and \
        ('p_u' in ds.keys()) and \
            ('rh_u_wrt_ice_or_water' in ds.keys()):
        # Upper boom bulk calculation
        T_h_u = ds['t_u'].copy()                                                   # Copy for processing
        p_h_u = ds['p_u'].copy()
        rh_h_u_wrt_ice_or_water = ds['rh_u_wrt_ice_or_water'].copy()

        q_h_u = calculate_specific_humidity(T_0, T_100, T_h_u, p_h_u, rh_h_u_wrt_ice_or_water)                  # Calculate specific humidity
        if ('wspd_u' in ds.keys()) and \
            ('t_surf' in ds.keys()) and \
                ('z_boom_u' in ds.keys()):
            WS_h_u = ds['wspd_u'].copy()
            Tsurf_h = ds['t_surf'].copy()                                              # T surf from derived upper boom product. TODO is this okay to use with lower boom parameters?
            z_WS_u = ds['z_boom_u'].copy() + 0.4                                       # Get height of Anemometer
            z_T_u = ds['z_boom_u'].copy() - 0.1                                        # Get height of thermometer

            if not ds.attrs['bedrock']:
                SHF_h_u, LHF_h_u= calculate_tubulent_heat_fluxes(T_0, T_h_u, Tsurf_h, WS_h_u,            # Calculate latent and sensible heat fluxes
                                                z_WS_u, z_T_u, q_h_u, p_h_u)

                ds['dshf_u'] = (('time'), SHF_h_u.data)
                ds['dlhf_u'] = (('time'), LHF_h_u.data)
        else:
            logger.info('wspd_u, t_surf or z_boom_u missing, cannot calulate tubrulent heat fluxes')

        q_h_u = 1000 * q_h_u                                                       # Convert sp.humid from kg/kg to g/kg
        ds['qh_u'] = (('time'), q_h_u.data)
    else:
        logger.info('t_u, p_u or rh_u_wrt_ice_or_water missing, cannot calulate tubrulent heat fluxes')

    # Lower boom bulk calculation
    if ds.attrs['number_of_booms']==2:
        if ('t_l' in ds.keys()) and \
            ('p_l' in ds.keys()) and \
                ('rh_l_wrt_ice_or_water' in ds.keys()):
            T_h_l = ds['t_l'].copy()                                               # Copy for processing
            p_h_l = ds['p_l'].copy()
            rh_h_l_wrt_ice_or_water = ds['rh_l_wrt_ice_or_water'].copy()

            q_h_l = calculate_specific_humidity(T_0, T_100, T_h_l, p_h_l, rh_h_l_wrt_ice_or_water)              # Calculate sp.humidity

            if ('wspd_l' in ds.keys()) and \
                ('t_surf' in ds.keys()) and \
                    ('z_boom_l' in ds.keys()):
                z_WS_l = ds['z_boom_l'].copy() + 0.4                                   # Get height of W
                z_T_l = ds['z_boom_l'].copy() - 0.1                                    # Get height of thermometer
                WS_h_l = ds['wspd_l'].copy()
                if not ds.attrs['bedrock']:
                    SHF_h_l, LHF_h_l= calculate_tubulent_heat_fluxes(T_0, T_h_l, Tsurf_h, WS_h_l, # Calculate latent and sensible heat fluxes
                                                    z_WS_l, z_T_l, q_h_l, p_h_l)

                    ds['dshf_l'] = (('time'), SHF_h_l.data)
                    ds['dlhf_l'] = (('time'), LHF_h_l.data)
            else:
                logger.info('wspd_l, t_surf or z_boom_l missing, cannot calulate tubrulent heat fluxes')

            q_h_l = 1000 * q_h_l                                                       # Convert sp.humid from kg/kg to g/kg
            ds['qh_l'] = (('time'), q_h_l.data)
        else:
            logger.info('t_l, p_l or rh_l_wrt_ice_or_water missing, cannot calulate tubrulent heat fluxes')

    if len(station_config)==0:
        logger.warning('\n***\nThe station configuration file is missing or improperly passed to pypromice. Some processing steps might fail.\n***\n')

    # Smoothing and inter/extrapolation of GPS coordinates
    for var in ['gps_lat', 'gps_lon', 'gps_alt']:
        ds[var.replace('gps_','')] = ('time', gps_coordinate_postprocessing(ds, var, station_config))

    # processing continuous surface height, ice surface height, snow height
    try:
        ds = process_surface_height(ds, data_adjustments_dir, station_config)
    except Exception as e:
        logger.error("Error processing surface height at %s"%L2.attrs['station_id'])
        logging.error(e, exc_info=True)

    # making sure dataset has the attributes contained in the config files
    if 'project' in station_config.keys():
        ds.attrs['project'] = station_config['project']
    else:
        logger.error('No project info in station_config. Using \"PROMICE\".')
        ds.attrs['project'] = "PROMICE"

    if 'location_type' in station_config.keys():
        ds.attrs['location_type'] = station_config['location_type']
    else:
        logger.error('No project info in station_config. Using \"ice sheet\".')
        ds.attrs['location_type'] = "ice sheet"

    return ds


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

    if ds.attrs['site_type'] == 'ablation':
        # Calculate surface heights for ablation sites
        ds['z_surf_1'] = 2.6 - ds['z_boom_u']
        if ds.z_stake.notnull().any():
            first_valid_index = ds.time.where((ds.z_stake + ds.z_boom_u).notnull(), drop=True).data[0]
            ds['z_surf_2'] = ds.z_surf_1.sel(time=first_valid_index) + ds.z_stake.sel(time=first_valid_index) - ds['z_stake']

        # Use corrected point data if available
        if 'z_pt_cor' in ds.data_vars:
            ds['z_ice_surf'] = ('time', ds['z_pt_cor'].data)

    else:
        # Calculate surface heights for other site types
        first_valid_index = ds.time.where(ds.z_boom_u.notnull(), drop=True).data[0]
        ds['z_surf_1'] = ds.z_boom_u.sel(time=first_valid_index) - ds['z_boom_u']
        if 'z_stake' in ds.data_vars and ds.z_stake.notnull().any():
            first_valid_index = ds.time.where(ds.z_stake.notnull(), drop=True).data[0]
            ds['z_surf_2'] = ds.z_stake.sel(time=first_valid_index) - ds['z_stake']
        if 'z_boom_l' in ds.data_vars:
            # need a combine first because KAN_U switches from having a z_stake
            # to having a z_boom_l
            first_valid_index = ds.time.where(ds.z_boom_l.notnull(), drop=True).data[0]
            ds['z_surf_2'] = ds['z_surf_2'].combine_first(
                ds.z_boom_l.sel(time=first_valid_index) - ds['z_boom_l'])

    # Adjust data for the created surface height variables
    ds = adjustData(ds, data_adjustments_dir, var_list=['z_surf_1', 'z_surf_2', 'z_ice_surf'])

    # Convert to dataframe and combine surface height variables
    df_in = ds[[v for v in ['z_surf_1', 'z_surf_2', 'z_ice_surf'] if v in ds.data_vars]].to_dataframe()

    (ds['z_surf_combined'], ds['z_ice_surf'],
     ds['z_surf_1_adj'], ds['z_surf_2_adj']) = combine_surface_height(df_in, ds.attrs['site_type'])


    if ds.attrs['site_type'] == 'ablation':
        # Calculate rolling minimum for ice surface height and snow height
        ts_interpolated = np.minimum(
            xr.where(ds.z_ice_surf.notnull(),
                     ds.z_ice_surf,ds.z_surf_combined),
            ds.z_surf_combined).to_series().resample('h').interpolate(limit=72)

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

        z_ice_surf = z_ice_surf.loc[ds.time]
        # here we make sure that the periods where both z_stake and z_pt are
        # missing are also missing in z_ice_surf
        msk = ds['z_ice_surf'].notnull() | ds['z_surf_2_adj'].notnull()
        z_ice_surf = z_ice_surf.where(msk)

        # taking running minimum to get ice
        z_ice_surf = z_ice_surf.cummin()

        # filling gaps only if they are less than a year long and if values on both
        # sides are less than 0.01 m appart

        # Forward and backward fill to identify bounds of gaps
        df_filled = z_ice_surf.fillna(method='ffill').fillna(method='bfill')

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

        # bringing the variable into the dataset
        ds['z_ice_surf'] = z_ice_surf

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

    if ds.attrs['site_type'] != 'bedrock':
        # Process ice temperature data and create depth variables
        ice_temp_vars = [v for v in ds.data_vars if 't_i_' in v]
        vars_out = [v.replace('t', 'd_t') for v in ice_temp_vars]
        vars_out.append('t_i_10m')
        df_out = get_thermistor_depth(
            ds[ice_temp_vars + ['z_surf_combined']].to_dataframe(),
            ds.attrs['station_id'],
            station_config)
        for var in df_out.columns:
            ds[var] = ('time', df_out[var].values)

    return ds

def combine_surface_height(df, site_type, threshold_ablation = -0.0002):
    '''Combines the data from three sensor: the two sonic rangers and the
    pressure transducer, to recreate the surface height, the ice surface height
    and the snow depth through the years. For the accumulation sites, it is
    only the average of the two sonic rangers (after manual adjustments to
    correct maintenance shifts). For the ablation sites, first an ablation
    period is estimated each year (either the period when z_pt_cor decreases
    or JJA if no better estimate) then different adjustmnents are conducted
    to stitch the three time series together: z_ice_surface (adjusted from
    z_pt_cor) or if unvailable, z_surf_2 (adjusted from z_stake)
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

        # defining ice ablation period from the decrease of a smoothed version of z_pt
        # meaning when smoothed_z_pt.diff() < threshold_ablation
        # first smoothing
        smoothed_PT =  (df['z_ice_surf']
                        .resample('h')
                        .interpolate(limit=72)
                        .rolling('14D',center=True, min_periods=1)
                        .mean())
        # second smoothing
        smoothed_PT = smoothed_PT.rolling('14D', center=True, min_periods=1).mean()

        smoothed_PT = smoothed_PT.reindex(df.index,method='ffill')
        # smoothed_PT.loc[df.z_ice_surf.isnull()] = np.nan

        # logical index where ablation is detected
        ind_ablation = np.logical_and(smoothed_PT.diff().values < threshold_ablation,
                                      np.isin(smoothed_PT.diff().index.month, [6, 7, 8, 9]))


        # finding the beginning and end of each period with True
        idx = np.argwhere(np.diff(np.r_[False,ind_ablation, False])).reshape(-1, 2)
        idx[:, 1] -= 1

        # fill small gaps in the ice ablation periods.
        for i in range(len(idx)-1):
            ind = idx[i]
            ind_next = idx[i+1]
            # if the end of an ablation period is less than 60 days away from
            # the next ablation, then it is still considered like the same ablation
            # season
            if df.index[ind_next[0]]-df.index[ind[1]]<pd.to_timedelta('60 days'):
                ind_ablation[ind[1]:ind_next[0]]=True

        # finding the beginning and end of each period with True
        idx = np.argwhere(np.diff(np.r_[False,ind_ablation, False])).reshape(-1, 2)
        idx[:, 1] -= 1

        # because the smooth_PT sees 7 days ahead, it starts showing a decline
        # 7 days in advance, we therefore need to exclude the first 7 days of
        # each ablation period
        for start, end in idx:
            period_start = df.index[start]
            period_end = period_start + pd.Timedelta(days=7)
            exclusion_period = (df.index >= period_start) & (df.index < period_end)
            ind_ablation[exclusion_period] = False

        hs1=df["z_surf_1_adj"].interpolate(limit=24*2).copy()
        hs2=df["z_surf_2_adj"].interpolate(limit=24*2).copy()
        z=df["z_ice_surf_adj"].interpolate(limit=24*2).copy()

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
            if ((z.first_valid_index() - hs1.first_valid_index()) < pd.to_timedelta('251D')) &\
              ((z.first_valid_index() - hs1.first_valid_index()) > pd.to_timedelta('0H')):
                # if the pressure transducer is installed the year after then
                # we use the mean surface height 1 on its first week as a 0
                # for the ice height
                z = z - z.loc[
                    z.first_valid_index():(z.first_valid_index()+pd.to_timedelta('14D'))
                    ].mean() + hs1.iloc[:24*7].mean()
            else:
                # if there is more than a year (actually 251 days) between the
                # initiation of the AWS and the installation of the pressure transducer
                # we remove the intercept in the pressure transducer data.
                # Removing the intercept
                # means that we consider the ice surface height at 0 when the AWS
                # is installed, and not when the pressure transducer is installed.
                Y = z.iloc[:].values.reshape(-1, 1)
                X = z.iloc[~np.isnan(Y)].index.astype(np.int64).values.reshape(-1, 1)
                Y = Y[~np.isnan(Y)]
                linear_regressor = LinearRegression()
                linear_regressor.fit(X, Y)
                Y_pred = linear_regressor.predict(z.index.astype(np.int64).values.reshape(-1, 1) )
                z = z-Y_pred[0]

        years = df.index.year.unique().values
        ind_start = years.copy()
        ind_end =  years.copy()
        logger.debug('-> estimating ablation period for each year')
        for i, y in enumerate(years):
            # for each year
            ind_yr = df.index.year.values==y
            ind_abl_yr = np.logical_and(ind_yr, ind_ablation)

            if df.loc[
                    np.logical_and(ind_yr, df.index.month.isin([6,7,8])),
                    "z_ice_surf_adj"].isnull().all():

                ind_abl_yr = np.logical_and(ind_yr, df.index.month.isin([6,7,8]))
                ind_ablation[ind_yr] = ind_abl_yr[ind_yr]
                logger.debug(str(y)+' no z_ice_surf, just using JJA')

            else:
                logger.debug(str(y)+ ' derived from z_ice_surf')

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

        # adjustement loop
        missing_hs2 = 0 # if hs2 is missing then when it comes back it is adjusted to hs1
        hs2_ref = 0 # by default, the PT is the reference: hs1 and 2 will be adjusted to PT
        # but if it is missing one year or one winter, then it needs to be rajusted
        # to hs1 and hs2 the year after.

        for i, y in enumerate(years):
            # if y == 2014:
            #     import pdb; pdb.set_trace()
            logger.debug(str(y))
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
                if z.iloc[(ind_end[i]-24*7):(ind_end[i]+24*7)].notnull().any():
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
                        if ind_start[i+1]!=-999 and any(~np.isnan(hs2.iloc[(ind_start[i+1]-24*7):(ind_start[i+1]+24*7)]+ z.iloc[(ind_start[i+1]-24*7):(ind_start[i+1]+24*7)])):
                            logger.debug('using end of accumulation season')
                            hs2.iloc[ind_end[i]:] = hs2.iloc[ind_end[i]:] - \
                                np.nanmean(hs2.iloc[(ind_start[i+1]-24*7):(ind_start[i+1]+24*7)])  + \
                                    np.nanmean(z.iloc[(ind_start[i+1]-24*7):(ind_start[i+1]+24*7)])
            else:
                logger.debug('no ablation')
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


def get_thermistor_depth(df_in, site, station_config):
    '''Calculates the depth of the thermistors through time based on their
    installation depth (collected in a google sheet) and on the change of surface
    height: instruments getting buried under new snow or surfacing due to ablation.
    There is a potential for additional filtering of thermistor data for surfaced
    (or just noisy) thermistors, but that is currently deactivated because slow.

    Parameters
    ----------
    df_in : pandas:dataframe
        dataframe containing the ice/firn temperature t_i_* as well as the
        combined surface height z_surf_combined
    site : str
        stid, so that maintenance date and sensor installation depths can be found
        in database
    station_config : dict
        potentially containing the key string_maintenance
        with station_config["string_maintenance"] being a list of dictionaries
        containing maintenance information in the format:
        [
            {"date": "2007-08-20", "installation_depth": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0]},
            {"date": "2008-07-17", "installation_depth": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 10.2]}
            # Add more entries as needed
        ]
    '''

    temp_cols_name = ['t_i_'+str(i) for i in range(12) if 't_i_'+str(i) in df_in.columns]
    num_therm = len(temp_cols_name)
    depth_cols_name = ['d_t_i_'+str(i) for i in range(1,num_therm+1)]

    if df_in['z_surf_combined'].isnull().all():
        logger.info('No valid surface height at '+site+', cannot calculate thermistor depth')
        df_in[depth_cols_name + ['t_i_10m']] = np.nan
    else:
        logger.info('Calculating thermistor depth')

        # Convert maintenance_info to DataFrame for easier manipulation
        maintenance_string = pd.DataFrame(
            station_config.get("string_maintenance",[]),
            columns = ['date', 'installation_depths']
            )
        maintenance_string["date"] = pd.to_datetime(maintenance_string["date"])
        maintenance_string = maintenance_string.sort_values(by='date', ascending=True)


        if num_therm == 8:
            ini_depth = [1, 2, 3, 4, 5, 6, 7, 10]
        else:
            ini_depth = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        df_in[depth_cols_name] = np.nan

        # filtering the surface height
        surface_height = df_in["z_surf_combined"].copy()
        ind_filter = surface_height.rolling(window=14, center=True).var() > 0.1
        if any(ind_filter):
            surface_height[ind_filter] = np.nan
        df_in["z_surf_combined"] = surface_height.values
        z_surf_interp = df_in["z_surf_combined"].interpolate()

        # first initialization of the depths
        for i, col in enumerate(depth_cols_name):
            df_in[col] = (
                ini_depth[i]
                + z_surf_interp.values
                - z_surf_interp[z_surf_interp.first_valid_index()]
            )

        # reseting depth at maintenance
        if len(maintenance_string.date) == 0:
            logger.info("No maintenance at "+site)

        for date in maintenance_string.date:
            if date > z_surf_interp.last_valid_index():
                continue
            new_depth = maintenance_string.loc[
                                            maintenance_string.date == date
                                        ].installation_depths.values[0]

            for i, col in enumerate(depth_cols_name[:len(new_depth)]):
                tmp = df_in[col].copy()
                tmp.loc[date:] = (
                    new_depth[i]
                    + z_surf_interp[date:].values
                    - z_surf_interp[date:][
                        z_surf_interp[date:].first_valid_index()
                    ]
                )
                df_in[col] = tmp.values

        # % Filtering thermistor data
        for i in range(len(temp_cols_name)):
            tmp = df_in[temp_cols_name[i]].copy()

            # variance filter
            # ind_filter = (
            #     df_in[temp_cols_name[i]]
            #     .interpolate(limit=14)
            #     .rolling(window=7)
            #     .var()
            #     > 0.5
            # )
            # month = (
            #     df_in[temp_cols_name[i]].interpolate(limit=14).index.month.values
            # )
            # ind_filter.loc[np.isin(month, [5, 6, 7])] = False
            # if any(ind_filter):
            #     tmp.loc[ind_filter] = np.nan

            # before and after maintenance adaptation filter
            if len(maintenance_string.date) > 0:
                for date in maintenance_string.date:
                    if isinstance(
                        maintenance_string.loc[
                            maintenance_string.date == date
                        ].installation_depths.values[0],
                        str,
                    ):
                        ind_adapt = np.abs(
                            tmp.interpolate(limit=14).index.values
                            - pd.to_datetime(date).to_datetime64()
                        ) < np.timedelta64(7, "D")
                        if any(ind_adapt):
                            tmp.loc[ind_adapt] = np.nan

            # surfaced thermistor
            ind_pos = df_in[depth_cols_name[i]] < 0.1
            if any(ind_pos):
                tmp.loc[ind_pos] = np.nan

            # copying the filtered values to the original table
            df_in[temp_cols_name[i]] = tmp.values

            # removing negative depth
            df_in.loc[df_in[depth_cols_name[i]]<0, depth_cols_name[i]] = np.nan
        logger.info("interpolating 10 m firn/ice temperature")
        df_in['t_i_10m'] = interpolate_temperature(
            df_in.index.values,
            df_in[depth_cols_name].values.astype(float),
            df_in[temp_cols_name].values.astype(float),
            kind="linear",
            min_diff_to_depth=1.5,
        ).set_index('date').values

        # filtering
        ind_pos = df_in["t_i_10m"] > 0.1
        ind_low = df_in["t_i_10m"] < -70
        df_in.loc[ind_pos, "t_i_10m"] = np.nan
        df_in.loc[ind_low, "t_i_10m"] = np.nan

    return df_in[depth_cols_name + ['t_i_10m']]


def interpolate_temperature(dates, depth_cor, temp, depth=10, min_diff_to_depth=2,
    kind="quadratic"):
    '''Calculates the depth of the thermistors through time based on their
    installation depth (collected in a google sheet) and on the change of surface
    height: instruments getting buried under new snow or surfacing due to ablation.
    There is a potential for additional filtering of thermistor data for surfaced
    (or just noisy) thermistors, but that is currently deactivated because slow.

    Parameters
    ----------
    dates : numpy.array
        array of datetime64
    depth_cor : numpy.ndarray
        matrix of depths
    temp : numpy.ndarray
        matrix of temperatures
    depth : float
        constant depth at which (depth_cor, temp) should be interpolated.
    min_diff_to_depth: float
        maximum difference allowed between the available depht and the target depth
        for the interpolation to be done.
    kind : str
        type of interpolation from scipy.interpolate.interp1d
    '''

    depth_cor = depth_cor.astype(float)
    df_interp = pd.DataFrame()
    df_interp["date"] = dates
    df_interp["temperatureObserved"] = np.nan

    # preprocessing temperatures for small gaps
    tmp = pd.DataFrame(temp)
    tmp["time"] = dates
    tmp = tmp.set_index("time")
    # tmp = tmp.resample("H").mean()
    # tmp = tmp.interpolate(limit=24*7)
    temp = tmp.loc[dates].values
    for i in (range(len(dates))):
        x = depth_cor[i, :].astype(float)
        y = temp[i, :].astype(float)
        ind_no_nan = ~np.isnan(x + y)
        x = x[ind_no_nan]
        y = y[ind_no_nan]
        x, indices = np.unique(x, return_index=True)
        y = y[indices]
        if len(x) < 2 or np.min(np.abs(x - depth)) > min_diff_to_depth:
            continue
        f = interp1d(x, y, kind, fill_value="extrapolate")
        df_interp.iloc[i, 1] = np.min(f(depth), 0)

    if df_interp.iloc[:5, 1].std() > 0.1:
        df_interp.iloc[:5, 1] = np.nan

    return df_interp

def gps_coordinate_postprocessing(ds, var, station_config={}):
        # saving the static value of 'lat','lon' or 'alt' stored in attribute
        # as it might be the only coordinate available for certain stations (e.g. bedrock)
        var_out = var.replace('gps_','')
        coord_names = {'lat':'latitude','lon':'longitude', 'alt':'altitude'}
        if coord_names[var_out] in list(ds.attrs.keys()):
            static_value = float(ds.attrs[coord_names[var_out]])
        else:
            static_value = np.nan

        # if there is no gps observations, then we use the static value repeated
        # for each time stamp
        if var not in ds.data_vars:
            print('no',var,'at', ds.attrs['station_id'])
            return np.ones_like(ds['t_u'].data)*static_value

        if ds[var].isnull().all():
            print('no',var,'at',ds.attrs['station_id'])
            return np.ones_like(ds['t_u'].data)*static_value

        # Extract station relocations from the config dict
        station_relocations = station_config.get("station_relocation", [])

        # Convert the ISO8601 strings to pandas datetime objects
        breaks = [pd.to_datetime(date_str) for date_str in station_relocations]
        if len(breaks)==0:
            logger.info('processing '+var+' without relocation')
        else:
            logger.info('processing '+var+' with relocation on ' + ', '.join([br.strftime('%Y-%m-%dT%H:%M:%S') for br in breaks]))

        return piecewise_smoothing_and_interpolation(ds[var].to_series(), breaks)

def piecewise_smoothing_and_interpolation(data_series, breaks):
    '''Smoothes, inter- or extrapolate the GPS observations. The processing is
    done piecewise so that each period between station relocations are done
    separately (no smoothing of the jump due to relocation). Piecewise linear
    regression is then used to smooth the available observations. Then this
    smoothed curve is interpolated linearly over internal gaps. Eventually, this
    interpolated curve is extrapolated linearly for timestamps before the first
    valid measurement and after the last valid measurement.

    Parameters
    ----------
    data_series : pandas.Series
        Series of observed latitude, longitude or elevation with datetime index.
    breaks: list
        List of timestamps of station relocation. First and last item should be
        None so that they can be used in slice(breaks[i], breaks[i+1])

    Returns
    -------
    np.ndarray
        Smoothed and interpolated values corresponding to the input series.
    '''
    df_all = pd.Series(dtype=float)  # Initialize an empty Series to gather all smoothed pieces
    breaks = [None] + breaks + [None]
    _inferred_series = []
    for i in range(len(breaks) - 1):
        df = data_series.loc[slice(breaks[i], breaks[i+1])]

        # Drop NaN values and calculate the number of segments based on valid data
        df_valid = df.dropna()
        if df_valid.shape[0] > 2:
            # Fit linear regression model to the valid data range
            x = pd.to_numeric(df_valid.index).values.reshape(-1, 1)
            y = df_valid.values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(x, y)

            # Predict using the model for the entire segment range
            x_pred = pd.to_numeric(df.index).values.reshape(-1, 1)

            y_pred = model.predict(x_pred)
            df =  pd.Series(y_pred.flatten(), index=df.index)
        # adds to list the predicted values for the current segment
        _inferred_series.append(df)

    df_all = pd.concat(_inferred_series)

    # Fill internal gaps with linear interpolation
    df_all = df_all.interpolate(method='linear', limit_area='inside')

    # Remove duplicate indices and return values as numpy array
    df_all = df_all[~df_all.index.duplicated(keep='last')]
    return df_all.values

def calculate_tubulent_heat_fluxes(T_0, T_h, Tsurf_h, WS_h, z_WS, z_T, q_h, p_h,
                kappa=0.4, WS_lim=1., z_0=0.001, g=9.82, es_0=6.1071, eps=0.622,
                gamma=16., L_sub=2.83e6, L_dif_max=0.01, c_pd=1005., aa=0.7,
                bb=0.75, cc=5., dd=0.35, R_d=287.05):
    '''Calculate latent and sensible heat flux using the bulk calculation
    method

    Parameters
    ----------
    T_0 : int
        Freezing point temperature
    T_h : xarray.DataArray
        Air temperature
    Tsurf_h : xarray.DataArray
        Surface temperature
    rho_atm : float
        Atmopsheric density
    WS_h : xarray.DataArray
        Wind speed
    z_WS : float
        Height of anemometer
    z_T : float
        Height of thermometer
    q_h : xarray.DataArray
        Specific humidity
    p_h : xarray.DataArray
        Air pressure
    kappa : int
        Von Karman constant (0.35-0.42). Default is 0.4.
    WS_lim : int
        Default is 1.
    z_0 : int
        Aerodynamic surface roughness length for momention, assumed constant
        for all ice/snow surfaces. Default is 0.001.
    g : int
        Gravitational acceleration (m/s2). Default is 9.82.
    es_0 : int
        Saturation vapour pressure at the melting point (hPa). Default is 6.1071.
    eps : int
        Ratio of molar masses of vapor and dry air (0.622).
    gamma : int
        Flux profile correction (Paulson & Dyer). Default is 16..
    L_sub : int
        Latent heat of sublimation (J/kg). Default is 2.83e6.
    L_dif_max : int
        Default is 0.01.
    c_pd : int
        Specific heat of dry air (J/kg/K). Default is 1005..
    aa : int
        Flux profile correction constants (Holtslag & De Bruin '88). Default is
        0.7.
    bb : int
        Flux profile correction constants (Holtslag & De Bruin '88). Default is
        0.75.
    cc : int
        Flux profile correction constants (Holtslag & De Bruin '88). Default is
        5.
    dd : int
        Flux profile correction constants (Holtslag & De Bruin '88). Default is
        0.35.
    R_d : int
        Gas constant of dry air. Default is 287.05.

    Returns
    -------
    SHF_h : xarray.DataArray
        Sensible heat flux
    LHF_h : xarray.DataArray
        Latent heat flux
    '''
    rho_atm = 100 * p_h / R_d / (T_h + T_0)                              # Calculate atmospheric density
    nu = calculate_viscosity(T_h, T_0, rho_atm)                                     # Calculate kinematic viscosity

    SHF_h = xr.zeros_like(T_h)                                                 # Create empty xarrays
    LHF_h = xr.zeros_like(T_h)
    L = xr.full_like(T_h, 1E5)

    u_star = kappa * WS_h.where(WS_h>0) / np.log(z_WS / z_0)                                 # Rough surfaces, from Smeets & Van den Broeke 2008
    Re = u_star * z_0 / nu
    z_0h = u_star
    z_0h = xr.where(WS_h <= 0,
                    1e-10,
                    z_0* np.exp(1.5 - 0.2 * np.log(Re) - 0.11 * np.log(Re)**2))
    es_ice_surf = 10**(-9.09718
                       * (T_0 / (Tsurf_h + T_0) -1) - 3.56654
                       * np.log10(T_0 / (Tsurf_h + T_0)) + 0.876793
                       * (1 - (Tsurf_h + T_0) / T_0)
                       + np.log10(es_0))
    q_surf = eps * es_ice_surf / (p_h - (1 - eps) * es_ice_surf)
    theta = T_h + z_T *g / c_pd
    stable = (theta > Tsurf_h) & (WS_h > WS_lim)
    unstable = (theta < Tsurf_h) & (WS_h > WS_lim)                             #TODO: check if unstable = ~stable? And if not why not
                                                                               #no_wind  = (WS_h <= WS_lim)
    # Calculate stable stratification
    for i in np.arange(0,31):
        psi_m1 = -(aa*         z_0/L[stable] + bb*(         z_0/L[stable]-cc/dd)*np.exp(-dd*         z_0/L[stable]) + bb*cc/dd)
        psi_m2 = -(aa*z_WS[stable]/L[stable] + bb*(z_WS[stable]/L[stable]-cc/dd)*np.exp(-dd*z_WS[stable]/L[stable]) + bb*cc/dd)
        psi_h1 = -(aa*z_0h[stable]/L[stable] + bb*(z_0h[stable]/L[stable]-cc/dd)*np.exp(-dd*z_0h[stable]/L[stable]) + bb*cc/dd)
        psi_h2 = -(aa* z_T[stable]/L[stable] + bb*( z_T[stable]/L[stable]-cc/dd)*np.exp(-dd* z_T[stable]/L[stable]) + bb*cc/dd)
        u_star[stable] = kappa*WS_h[stable]/(np.log(z_WS[stable]/z_0)-psi_m2+psi_m1)
        Re[stable] = u_star[stable]*z_0/nu[stable]
        z_0h[stable] = z_0*np.exp(1.5-0.2*np.log(Re[stable])-0.11*(np.log(Re[stable]))**2)

        # If n_elements(where(z_0h[stable] < 1e-6)) get 1 then
        # z_0h[stable[where(z_0h[stable] < 1e-6)]] = 1e-6
        z_0h[stable][z_0h[stable] < 1E-6] == 1E-6
        th_star = kappa \
            * (theta[stable] - Tsurf_h[stable]) \
            / (np.log(z_T[stable] / z_0h[stable]) - psi_h2 + psi_h1)
        q_star  = kappa *(q_h[stable] - q_surf[stable]) \
            / (np.log(z_T[stable] / z_0h[stable]) - psi_h2 + psi_h1)
        SHF_h[stable] = rho_atm[stable] * c_pd * u_star[stable] * th_star
        LHF_h[stable] = rho_atm[stable] * L_sub * u_star[stable] * q_star
        L_prev = L[stable]
        L[stable] = u_star[stable]**2 \
            * (theta[stable] + T_0)\
            * (1 + ((1-eps) / eps) * q_h[stable]) \
            / (g * kappa * th_star * (1 + ((1-eps)/eps) * q_star))
        L_dif = np.abs((L_prev-L[stable])/L_prev)

        # If n_elements(where(L_dif > L_dif_max)) eq 1 then break
        if np.all(L_dif <= L_dif_max):
            break

    # Calculate unstable stratification
    if len(unstable) > 0:
        for i in np.arange(0,21):
            x1  = (1-gamma*z_0           /L[unstable])**0.25
            x2  = (1-gamma*z_WS[unstable]/L[unstable])**0.25
            y1  = (1-gamma*z_0h[unstable]/L[unstable])**0.5
            y2  = (1-gamma*z_T[unstable] /L[unstable])**0.5
            psi_m1 = np.log(((1+x1)/2)**2*(1+x1**2)/2)-2*np.arctan(x1)+np.pi/2
            psi_m2 = np.log(((1+x2)/2)**2*(1+x2**2)/2)-2*np.arctan(x2)+np.pi/2
            psi_h1 = np.log(((1+y1)/2)**2)
            psi_h2 = np.log(((1+y2)/2)**2)
            u_star[unstable] = kappa*WS_h[unstable]/(np.log(z_WS[unstable]/z_0)-psi_m2+psi_m1)
            Re[unstable] = u_star[unstable]*z_0/nu[unstable]
            z_0h[unstable] = z_0 * np.exp(1.5 - 0.2 * np.log(Re[unstable]) - 0.11 \
                                          * (np.log(Re[unstable]))**2)

            # If n_elements(where(z_0h[unstable] < 1e-6)) > 1 then
            # z_0h[unstable[where(z_0h[unstable] < 1e-6)]] = 1e-6
            z_0h[stable][z_0h[stable] < 1E-6] == 1E-6
            th_star = kappa * (theta[unstable] - Tsurf_h[unstable]) \
                / (np.log(z_T[unstable] / z_0h[unstable]) - psi_h2 + psi_h1)
            q_star  = kappa * (q_h[unstable] - q_surf[unstable]) \
                / (np.log(z_T[unstable] / z_0h[unstable]) - psi_h2 + psi_h1)
            SHF_h[unstable] = rho_atm[unstable] * c_pd * u_star[unstable] * th_star
            LHF_h[unstable] = rho_atm[unstable] * L_sub * u_star[unstable] * q_star
            L_prev = L[unstable]
            L[unstable] = u_star[unstable]**2 * (theta[unstable]+T_0) \
                * ( 1 + ((1-eps) / eps) * q_h[unstable]) \
                / (g * kappa * th_star * ( 1 + ((1-eps) / eps) * q_star))
            L_dif = abs((L_prev-L[unstable])/L_prev)

            # If n_elements(where(L_dif > L_dif_max)) eq 1 then break
            if np.all(L_dif <= L_dif_max):
                break

    HF_nan = np.isnan(p_h) | np.isnan(T_h) | np.isnan(Tsurf_h) \
        | np.isnan(q_h) | np.isnan(WS_h) | np.isnan(z_T)
    SHF_h[HF_nan] = np.nan
    LHF_h[HF_nan] = np.nan
    return SHF_h, LHF_h

def calculate_viscosity(T_h, T_0, rho_atm):
    '''Calculate kinematic viscosity of air

    Parameters
    ----------
    T_h : xarray.DataArray
        Air temperature
    T_0 : float
        Steam point temperature
    rho_atm : xarray.DataArray
        Surface temperature

    Returns
    -------
    xarray.DataArray
        Kinematic viscosity
    '''
    # Dynamic viscosity of air in Pa s (Sutherlands' equation using C = 120 K)
    mu = 18.27e-6 * (291.15 + 120) / ((T_h + T_0) + 120) * ((T_h + T_0) / 291.15)**1.5

    # Kinematic viscosity of air in m^2/s
    return mu / rho_atm

def calculate_specific_humidity(T_0, T_100, T_h, p_h, rh_h_wrt_ice_or_water, es_0=6.1071, es_100=1013.246, eps=0.622):
    '''Calculate specific humidity
    Parameters
    ----------
    T_0 : float
        Steam point temperature. Default is 273.15.
    T_100 : float
        Steam point temperature in Kelvin
    T_h : xarray.DataArray
        Air temperature
    p_h : xarray.DataArray
        Air pressure
    rh_h_wrt_ice_or_water : xarray.DataArray
        Relative humidity corrected
    es_0 : float
        Saturation vapour pressure at the melting point (hPa)
    es_100 : float
        Saturation vapour pressure at steam point temperature (hPa)
    eps : int
        ratio of molar masses of vapor and dry air (0.622)

    Returns
    -------
    xarray.DataArray
        Specific humidity data array
    '''
    # Saturation vapour pressure above 0 C (hPa)
    es_wtr = 10**(-7.90298 * (T_100 / (T_h + T_0) - 1) + 5.02808 * np.log10(T_100 / (T_h + T_0))
                  - 1.3816E-7 * (10**(11.344 * (1 - (T_h + T_0) / T_100)) - 1)
                  + 8.1328E-3 * (10**(-3.49149 * (T_100 / (T_h + T_0) -1)) - 1) + np.log10(es_100))

    # Saturation vapour pressure below 0 C (hPa)
    es_ice = 10**(-9.09718 * (T_0 / (T_h + T_0) - 1) - 3.56654
                  * np.log10(T_0 / (T_h + T_0)) + 0.876793
                  * (1 - (T_h + T_0) / T_0)
                  + np.log10(es_0))

    # Specific humidity at saturation (incorrect below melting point)
    q_sat = eps * es_wtr / (p_h - (1 - eps) * es_wtr)

    # Replace saturation specific humidity values below melting point
    freezing = T_h < 0
    q_sat[freezing] = eps * es_ice[freezing] / (p_h[freezing] - (1 - eps) * es_ice[freezing])

    q_nan = np.isnan(T_h) | np.isnan(p_h)
    q_sat[q_nan] = np.nan

    # Convert to kg/kg
    return rh_h_wrt_ice_or_water * q_sat / 100

if __name__ == "__main__":
    # unittest.main()
    pass
