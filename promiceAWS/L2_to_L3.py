#!/usr/bin/env python

import numpy as np
import xarray as xr

def to_L3(L2=None):

    ds = L2
    
    deg2rad = np.pi / 180
    rad2deg = 1 / deg2rad
    
    # ds_h = ds.resample({'time':"1H"}).mean() # this takes ~2-3 minutes
    ## https://github.com/pydata/xarray/issues/4498 & https://stackoverflow.com/questions/64282393/
    df_h = ds.to_dataframe().resample("1H").mean()  # what we want (quickly), but in Pandas form
    ## now, rebuild xarray dataset (https://www.theurbanist.com.au/2020/03/how-to-create-an-xarray-dataset-from-scratch/)
    vals = [xr.DataArray(
        data=df_h[c], dims=['time'], coords={'time':df_h.index}, attrs=ds[c].attrs)
            for c in df_h.columns]
    ds_h = xr.Dataset(dict(zip(df_h.columns,vals)), attrs=ds.attrs)
    
    ds_h['wdir'] = np.arctan2(ds_h['wspd_x'], ds_h['wspd_y']) * rad2deg
    ds_h['wdir'] = (ds_h['wdir'] + 360) % 360
    z_0    =    0.001    # aerodynamic surface roughness length for momention...
    # ...(assumed constant for all ice/snow surfaces)
    eps    =    0.622
    es_0   =    6.1071   # saturation vapour pressure at the melting point (hPa)
    es_100 = 1013.246    # saturation vapour pressure at steam point temperature (hPa)
    g      =    9.82     # gravitational acceleration (m/s2)
    gamma  =   16.       # flux profile correction (Paulson & Dyer)
    kappa  =    0.4      # Von Karman constant (0.35-0.42)
    L_sub  =    2.83e6   # latent heat of sublimation (J/kg)
    R_d    =  287.05     # gas constant of dry air
    aa     =    0.7      # flux profile correction constants (Holtslag & De Bruin '88)
    bb     =    0.75
    cc     =    5.
    dd     =    0.35
    c_pd   = 1005.       # specific heat of dry air (J/kg/K)
    WS_lim =    1.
    L_dif_max = 0.01
    
    T_0 = 273.15
    T_100 = T_0+100            # steam point temperature in K
    
    T_h = ds_h['t_1'].copy()
    p_h = ds_h['p'].copy()
    WS_h = ds_h['wspd'].copy()
    Tsurf_h = ds_h['t_surf'].copy()
    RH_cor_h = ds_h['rh_cor'].copy()

    # TODO: Why +0.4 and -0.1 here??
    z_WS = ds_h['z_boom'].copy() + 0.4  # height of W
    z_T = ds_h['z_boom'].copy() - 0.1   # height of thermometer
    
    rho_atm = 100 * p_h / R_d / (T_h + T_0)   # atmospheric density
    
    # dynamic viscosity of air (Pa s) (Sutherlands' equation using C = 120 K)
    mu = 18.27e-6 * (291.15 + 120) / ((T_h + T_0) + 120) * ((T_h + T_0) / 291.15)**1.5
    
    nu = mu / rho_atm                            # kinematic viscosity of air (m^2/s)
    u_star = kappa * WS_h / np.log(z_WS / z_0)
    Re = u_star * z_0 / nu
    # rough surfaces: Smeets & Van den Broeke 2008
    z_0h = z_0 * np.exp(1.5 - 0.2 * np.log(Re) - 0.11 * np.log(Re)**2)
    z_0h[WS_h <= 0] = 1e-10
    es_ice_surf = 10**(-9.09718
                       * (T_0 / (Tsurf_h + T_0) -1) - 3.56654
                       * np.log10(T_0 / (Tsurf_h + T_0)) + 0.876793
                       * (1 - (Tsurf_h + T_0) / T_0)
                       + np.log10(es_0))
    q_surf = eps * es_ice_surf / (p_h - (1 - eps) * es_ice_surf)
    
    # saturation vapour pressure above 0 C (hPa)
    es_wtr = 10**(-7.90298 * (T_100 / (T_h + T_0) - 1) + 5.02808 * np.log10(T_100 / (T_h + T_0))
                  - 1.3816E-7 * (10**(11.344 * (1 - (T_h + T_0) / T_100)) - 1)
                  + 8.1328E-3 * (10**(-3.49149 * (T_100 / (T_h + T_0) -1)) - 1) + np.log10(es_100))

    es_ice = 10**(-9.09718 * (T_0 / (T_h + T_0) - 1) - 3.56654
                  * np.log10(T_0 / (T_h + T_0)) + 0.876793
                  * (1 - (T_h + T_0) / T_0)
                  + np.log10(es_0)) # saturation vapour pressure below 0 C (hPa)

    # specific humidity at saturation (incorrect below melting point)
    q_sat = eps * es_wtr / (p_h - (1 - eps) * es_wtr) 
    freezing = T_h < 0  # replacing saturation specific humidity values below melting point
    q_sat[freezing] = eps * es_ice[freezing] / (p_h[freezing] - (1 - eps) * es_ice[freezing])
    q_h = RH_cor_h * q_sat / 100   # specific humidity in kg/kg
    theta = T_h + z_T *g / c_pd
    SHF_h = T_h
    SHF_h[:] = 0
    LHF_h = SHF_h
    L = SHF_h + 1E5
    
    stable = (theta > Tsurf_h) & (WS_h > WS_lim)
    unstable = (theta < Tsurf_h) & (WS_h > WS_lim)
    # no_wind  = (WS_h <= WS_lim)
    
    for i in np.arange(0,31): # stable stratification
        psi_m1 = -(aa*         z_0/L[stable] + bb*(         z_0/L[stable]-cc/dd)*np.exp(-dd*         z_0/L[stable]) + bb*cc/dd)
        psi_m2 = -(aa*z_WS[stable]/L[stable] + bb*(z_WS[stable]/L[stable]-cc/dd)*np.exp(-dd*z_WS[stable]/L[stable]) + bb*cc/dd)
        psi_h1 = -(aa*z_0h[stable]/L[stable] + bb*(z_0h[stable]/L[stable]-cc/dd)*np.exp(-dd*z_0h[stable]/L[stable]) + bb*cc/dd)
        psi_h2 = -(aa* z_T[stable]/L[stable] + bb*( z_T[stable]/L[stable]-cc/dd)*np.exp(-dd* z_T[stable]/L[stable]) + bb*cc/dd)
        u_star[stable] = kappa*WS_h[stable]/(np.log(z_WS[stable]/z_0)-psi_m2+psi_m1)
        Re[stable] = u_star[stable]*z_0/nu[stable]
        z_0h[stable] = z_0*np.exp(1.5-0.2*np.log(Re[stable])-0.11*(np.log(Re[stable]))**2)
        # if n_elements(where(z_0h[stable] lt 1e-6)) gt 1 then z_0h[stable[where(z_0h[stable] lt 1e-6)]] = 1e-6
        z_0h[stable][z_0h[stable] < 1E-6] == 1E-6
        th_star = kappa \
            * (theta[stable] - Tsurf_h[stable] ) \
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
        # print,"HF iterations stable stratification: ",i+1,n_elements(where(L_dif gt L_dif_max)),100.*n_elements(where(L_dif gt L_dif_max))/n_elements(where(L_dif))
        # if n_elements(where(L_dif gt L_dif_max)) eq 1 then break
        if np.all(L_dif <= L_dif_max):
            # print("LDIF BREAK: ", i)
            break
    
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
            # if n_elements(where(z_0h[unstable] lt 1e-6)) gt 1 then z_0h[unstable[where(z_0h[unstable] lt 1e-6)]] = 1e-6
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
            # print,"HF iterations unstable stratification: ",i+1,n_elements(where(L_dif gt L_dif_max)),100.*n_elements(where(L_dif gt L_dif_max))/n_elements(where(L_dif))
            # if n_elements(where(L_dif gt L_dif_max)) eq 1 then break
            if np.all(L_dif <= L_dif_max):
                # print("LDIF BREAK: ", i)
                break
    
               
    q_h = 1000 * q_h            # from kg/kg to g/kg
    HF_nan = np.isnan(p_h) \
        | np.isnan(T_h) \
        | np.isnan(Tsurf_h) \
        | np.isnan(RH_cor_h) \
        | np.isnan(WS_h) \
        | np.isnan(ds_h['z_boom'])
    qh_nan = np.isnan(T_h) | np.isnan(RH_cor_h) | np.isnan(p_h) | np.isnan(Tsurf_h)
    SHF_h[HF_nan] = np.nan
    LHF_h[HF_nan] = np.nan
    q_h[qh_nan] = np.nan

    
    ds_h['SHF'] = (('time'), SHF_h.data)
    ds_h['LHF'] = (('time'), LHF_h.data)
    ds_h['qh'] = (('time'), q_h.data)

    
    ## Compute daily average
    # ds_d = ds_h.resample({'time':"1D"}).mean() # this takes ~2-3 minutes
    ## https://github.com/pydata/xarray/issues/4498 & https://stackoverflow.com/questions/64282393/
    df_d = ds_h.to_dataframe().resample("1D").mean()
    vals = [xr.DataArray(
        data=df_d[c], dims=['time'], coords={'time':df_d.index}, attrs=ds_h[c].attrs)
            for c in df_d.columns]
    ds_d = xr.Dataset(dict(zip(df_d.columns,vals)), attrs=ds_h.attrs)

    return [ds_h, ds_d]
