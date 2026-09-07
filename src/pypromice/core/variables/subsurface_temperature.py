import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

import logging
logger = logging.getLogger(__name__)

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

        logger.info("interpolating 10 m firn/ice temperature (on hourly values)")
        df_in_h = df_in[depth_cols_name+temp_cols_name].resample('h').mean()
        df_in_h['t_i_10m'] = interpolate_temperature(
            df_in_h.index.values,
            df_in_h[depth_cols_name].values.astype(float),
            df_in_h[temp_cols_name].values.astype(float),
            kind="linear",
            min_diff_to_depth=1.5,
        ).set_index('date').values
        df_in['t_i_10m'] = df_in_h['t_i_10m'].reindex(df_in.index,
                                        method=None)

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
