#!/usr/bin/env python
"""
AWS Level 2 (L2) to Level 3 (L3) data processing
"""
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from pathlib import Path
import logging

from pypromice.core.variables import humidity, surface_height, subsurface_temperature

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

    is_bedrock = (str(ds.attrs['bedrock']).lower() == 'true')

    # Turbulent heat flux calculation
    if ('t_u' in ds.keys()) and \
        ('p_u' in ds.keys()) and \
            ('rh_u_wrt_ice_or_water' in ds.keys()):
        # Upper boom bulk calculation
        T_h_u = ds['t_u'].copy()                                                   # Copy for processing
        p_h_u = ds['p_u'].copy()

        # Calculate specific humidity
        q_h_u = humidity.calculate_specific_humidity(ds["t_u"],
                                                     ds["p_u"],
                                                     ds["rh_u_wrt_ice_or_water"])

        if ('wspd_u' in ds.keys()) and \
            ('t_surf' in ds.keys()) and \
                ('z_boom_cor_u' in ds.keys()):
            WS_h_u = ds['wspd_u'].copy()
            Tsurf_h = ds['t_surf'].copy()                                              # T surf from derived upper boom product. TODO is this okay to use with lower boom parameters?

            z_WS_u = ds['z_boom_cor_u'].copy() + 0.4  # Get height of Anemometer
            z_T_u = ds['z_boom_cor_u'].copy() - 0.1  # Get height of thermometer

            if not is_bedrock:
                SHF_h_u, LHF_h_u= calculate_turbulent_heat_fluxes(T_0, T_h_u, Tsurf_h, WS_h_u,            # Calculate latent and sensible heat fluxes
                                                z_WS_u, z_T_u, q_h_u, p_h_u)

                ds['dshf_u'] = (('time'), SHF_h_u.data)
                ds['dlhf_u'] = (('time'), LHF_h_u.data)
        else:
            logger.info('wspd_u, t_surf or z_boom_cor_u missing, cannot calculate turbulent heat fluxes')

        # Convert specific humidity from kg/kg to g/kg
        ds['qh_u'] = humidity.convert(q_h_u)
    else:
        logger.info('t_u, p_u or rh_u_wrt_ice_or_water missing, cannot calculate turbulent heat fluxes')

    # Lower boom bulk calculation
    if ds.attrs['number_of_booms']==2:
        if ('t_l' in ds.keys()) and \
            ('p_l' in ds.keys()) and \
                ('rh_l_wrt_ice_or_water' in ds.keys()):
            T_h_l = ds['t_l'].copy()                                               # Copy for processing
            p_h_l = ds['p_l'].copy()

            # Calculate specific humidity
            q_h_l = humidity.calculate_specific_humidity(ds["t_l"],
                                                         ds["p_l"],
                                                         ds["rh_l_wrt_ice_or_water"])

            if ('wspd_l' in ds.keys()) and \
                ('t_surf' in ds.keys()) and \
                    ('z_boom_cor_l' in ds.keys()):
                z_WS_l = ds['z_boom_cor_l'].copy() + 0.4  # Get height of radiometer
                z_T_l = ds['z_boom_cor_l'].copy() - 0.1  # Get height of thermometer

                # Get wind speed lower boom measurements
                WS_h_l = ds['wspd_l'].copy()

                if not is_bedrock:
                    SHF_h_l, LHF_h_l= calculate_turbulent_heat_fluxes(T_0, T_h_l, Tsurf_h, WS_h_l, # Calculate latent and sensible heat fluxes
                                                    z_WS_l, z_T_l, q_h_l, p_h_l)

                    ds['dshf_l'] = (('time'), SHF_h_l.data)
                    ds['dlhf_l'] = (('time'), LHF_h_l.data)
            else:
                logger.info('wspd_l, t_surf or z_boom_cor_l missing, cannot calculate turbulent heat fluxes')

            # Convert specific humidity from kg/kg to g/kg
            ds['qh_l'] = humidity.convert(q_h_l)

        else:
            logger.info('t_l, p_l or rh_l_wrt_ice_or_water missing, cannot calculate turbulent heat fluxes')

    if len(station_config)==0:
        logger.warning('\n***\nThe station configuration file is missing or improperly passed to pypromice. Some processing steps might fail.\n***\n')

    # Smoothing and inter/extrapolation of GPS coordinates
    for var in ['gps_lat', 'gps_lon', 'gps_alt']:
        ds[var.replace('gps_','')] = ('time', gps_coordinate_postprocessing(ds, var, station_config))

    # processing continuous surface height, ice surface height, snow height
    try:
        ds = surface_height.process_surface_height(
                                                                    ds,
                                                                    data_adjustments_dir,
                                                                    station_config)
    except Exception as e:
        logger.error("Error processing surface height at %s"%L2.attrs['station_id'])
        logging.error(e, exc_info=True)

    try:
        if ds.attrs['site_type'] != 'bedrock':
            # Process ice temperature data and create depth variables
            ice_temp_vars = [v for v in ds.data_vars if 't_i_' in v]
            vars_out = [v.replace('t', 'd_t') for v in ice_temp_vars]
            vars_out.append('t_i_10m')

            df_out = subsurface_temperature.get_thermistor_depth(
                ds[ice_temp_vars + ['z_surf_combined']].to_dataframe(),
                ds.attrs['station_id'],
                station_config)

            for var in df_out.columns:
                ds[var] = ('time', df_out[var].values)
    except Exception as e:
        logger.error("Error processing thermistor depth and t_i_10m at %s"%L2.attrs['station_id'])
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

        return piecewise_smoothing_and_interpolation(
                    ds[var].to_series(),
                    breaks,
                    use_mean=(
                        var == 'gps_alt'
                        and ds.attrs.get('site_type') == 'accumulation'
                    )
                )

def piecewise_smoothing_and_interpolation(data_series, breaks, use_mean=False):
    '''Smoothes, inter- or extrapolate the GPS observations. The processing is
    done piecewise so that each period between station relocations are done
    separately (no smoothing of the jump due to relocation). Piecewise linear
    regression is then used to smooth the available observations. Then this
    smoothed curve is interpolated linearly over internal gaps. Eventually, this
    interpolated curve is extrapolated linearly for timestamps before the first
    valid measurement and after the last valid measurement.

    Parameters
    ----------
    data_series : pd.Series
        Series of observed latitude, longitude or elevation with datetime index.
    breaks: list
        List of timestamps of station relocation. First and last item should be
        None so that they can be used in slice(breaks[i], breaks[i+1])

    Returns
    -------
    np.ndarray
        Smoothed and interpolated values corresponding to the input series.
    '''
    breaks = [None] + breaks + [None]
    _inferred_series = []

    for i in range(len(breaks) - 1):
        df = data_series.loc[slice(breaks[i], breaks[i+1])]

        df_valid = df.dropna()

        if df_valid.shape[0] > 2:

            if use_mean:
                # Fill whole segment with mean value
                y_pred = np.full(len(df), df_valid.mean())
                df = pd.Series(y_pred, index=df.index)

            else:
                # Linear regression
                x = pd.to_numeric(df_valid.index).values.reshape(-1, 1)
                y = df_valid.values.reshape(-1, 1)

                model = LinearRegression()
                model.fit(x, y)

                x_pred = pd.to_numeric(df.index).values.reshape(-1, 1)
                y_pred = model.predict(x_pred)

                df = pd.Series(y_pred.flatten(), index=df.index)

        _inferred_series.append(df)

    df_all = pd.concat(_inferred_series)

    df_all = df_all.interpolate(
        method='linear',
        limit_area='inside'
    )

    df_all = df_all[~df_all.index.duplicated(keep='last')]

    return df_all.values

def calculate_turbulent_heat_fluxes(T_0, T_h, Tsurf_h, WS_h, z_WS, z_T, q_h, p_h,
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

if __name__ == "__main__":
    # unittest.main()
    pass
