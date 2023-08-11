#!/usr/bin/env python
"""
AWS Level 2 (L2) to Level 3 (L3) data processing
"""
import numpy as np
import xarray as xr

def toL3(L2, T_0=273.15, z_0=0.001, R_d=287.05, eps=0.622, es_0=6.1071, 
         es_100=1013.246):
    '''Process one Level 2 (L2) product to Level 3 (L3) 
    
    Parameters
    ----------
    L2 : xarray:Dataset
        L2 AWS data
    T_0 : int 
        Steam point temperature. Default is 273.15.
    z_0 : int
        Aerodynamic surface roughness length for momention, assumed constant 
        for all ice/snow surfaces. Default is 0.001.
    R_d : int 
        Gas constant of dry air. Default is 287.05.
    eps : int 
        Default is 0.622.
    es_0 : int 
        Saturation vapour pressure at the melting point (hPa). Default is 6.1071.
    es_100 : int
        Saturation vapour pressure at steam point temperature (hPa). Default is 
        1013.246.
    '''
    ds = L2

    T_100 = _getTempK(T_0)                                                     # Get steam point temperature as K 
    
    ds['wdir_u'] = ds['wdir_u'].where(ds['wspd_u'] != 0)                       # Get directional wind speed                    
    ds['wspd_x_u'], ds['wspd_y_u'] = calcDirWindSpeeds(ds['wspd_u'], ds['wdir_u']) 

    # Upper boom bulk calculation
    T_h_u = ds['t_u'].copy()                                                   # Copy for processing
    p_h_u = ds['p_u'].copy()
    WS_h_u = ds['wspd_u'].copy()
    RH_cor_h_u = ds['rh_u_cor'].copy()
    Tsurf_h = ds['t_surf'].copy()                                              # T surf from derived upper boom product. TODO is this okay to use with lower boom parameters?
    z_WS_u = ds['z_boom_u'].copy() + 0.4                                       # Get height of Anemometer                
    z_T_u = ds['z_boom_u'].copy() - 0.1                                        # Get height of thermometer  
        
    rho_atm_u = 100 * p_h_u / R_d / (T_h_u + T_0)                              # Calculate atmospheric density                                  
    nu_u = calcVisc(T_h_u, T_0, rho_atm_u)                                     # Calculate kinematic viscosity  
    q_h_u = calcHumid(T_0, T_100, T_h_u, es_0, es_100, eps,                    # Calculate specific humidity
                       p_h_u, RH_cor_h_u)   
    if not ds.attrs['bedrock']:       
        SHF_h_u, LHF_h_u= calcHeatFlux(T_0, T_h_u, Tsurf_h, rho_atm_u, WS_h_u,     # Calculate latent and sensible heat fluxes
                                        z_WS_u, z_T_u, nu_u, q_h_u, p_h_u)     
        SHF_h_u, LHF_h_u = cleanHeatFlux(SHF_h_u, LHF_h_u, T_h_u, Tsurf_h, p_h_u,  # Clean heat flux values
                                          WS_h_u, RH_cor_h_u, ds['z_boom_u'])
        ds['dshf_u'] = (('time'), SHF_h_u.data)
        ds['dlhf_u'] = (('time'), LHF_h_u.data)
    q_h_u = 1000 * q_h_u                                                       # Convert sp.humid from kg/kg to g/kg
    q_h_u = cleanSpHumid(q_h_u, T_h_u, Tsurf_h, p_h_u, RH_cor_h_u)             # Clean sp.humid values    
    ds['qh_u'] = (('time'), q_h_u.data)    

    # Lower boom bulk calculation
    if ds.attrs['number_of_booms']==2:                                         
        # ds['wdir_l'] = _calcWindDir(ds['wspd_x_l'], ds['wspd_y_l'])          # Calculatate wind direction

        T_h_l = ds['t_l'].copy()                                               # Copy for processing
        p_h_l = ds['p_l'].copy()
        WS_h_l = ds['wspd_l'].copy()                                      
        RH_cor_h_l = ds['rh_l_cor'].copy()
        z_WS_l = ds['z_boom_l'].copy() + 0.4                                   # Get height of W                  
        z_T_l = ds['z_boom_l'].copy() - 0.1                                    # Get height of thermometer 
           
        rho_atm_l = 100 * p_h_l / R_d / (T_h_l + T_0)                          # Calculate atmospheric density                                  
        nu_l = calcVisc(T_h_l, T_0, rho_atm_l)                                 # Calculate kinematic viscosity  
        q_h_l = calcHumid(T_0, T_100, T_h_l, es_0, es_100, eps,                # Calculate sp.humidity
                           p_h_l, RH_cor_h_l)
        if not ds.attrs['bedrock']:       
            SHF_h_l, LHF_h_l= calcHeatFlux(T_0, T_h_l, Tsurf_h, rho_atm_l, WS_h_l, # Calculate latent and sensible heat fluxes 
                                            z_WS_l, z_T_l, nu_l, q_h_l, p_h_l)        
            SHF_h_l, LHF_h_l = cleanHeatFlux(SHF_h_l, LHF_h_l, T_h_l, Tsurf_h, p_h_l, # Clean heat flux values
                                              WS_h_l, RH_cor_h_l, ds['z_boom_l'])
            ds['dshf_l'] = (('time'), SHF_h_l.data)
            ds['dlhf_l'] = (('time'), LHF_h_l.data)
        q_h_l = 1000 * q_h_l                                                   # Convert sp.humid from kg/kg to g/kg
        q_h_l = cleanSpHumid(q_h_l, T_h_l, Tsurf_h, p_h_l, RH_cor_h_l)         # Clean sp.humid values
        ds['qh_l'] = (('time'), q_h_l.data)    

        ds['wdir_l'] = ds['wdir_l'].where(ds['wspd_l'] != 0)                   # Get directional wind speed    
        ds['wspd_x_l'], ds['wspd_y_l'] = calcDirWindSpeeds(ds['wspd_l'], ds['wdir_l'])
     
    if hasattr(ds, 'wdir_i'):    
        if ~ds['wdir_i'].isnull().all() and ~ds['wspd_i'].isnull().all():      # Instantaneous msg processing
            ds['wdir_i'] = ds['wdir_i'].where(ds['wspd_i'] != 0)               # Get directional wind speed                    
            ds['wspd_x_i'], ds['wspd_y_i'] = calcDirWindSpeeds(ds['wspd_i'], ds['wdir_i'])   
    
    return ds


def calcDirWindSpeeds(wspd, wdir, deg2rad=np.pi/180):
    '''Calculate directional wind speed from wind speed and direction
    
    Parameters
    ----------
    wspd : xr.Dataarray
        Wind speed data array
    wdir : xr.Dataarray
        Wind direction data array
    deg2rad : float
        Degree to radians coefficient. The default is np.pi/180
    
    Returns
    -------
    wspd_x : xr.Dataarray
        Wind speed in X direction
    wspd_y : xr.Datarray
        Wind speed in Y direction
    '''        
    wspd_x = wspd * np.sin(wdir * deg2rad)
    wspd_y = wspd * np.cos(wdir * deg2rad) 
    return wspd_x, wspd_y


def calcHeatFlux(T_0, T_h, Tsurf_h, rho_atm, WS_h, z_WS, z_T, nu, q_h, p_h, 
                kappa=0.4, WS_lim=1., z_0=0.001, g=9.82, es_0=6.1071, eps=0.622, 
                gamma=16., L_sub=2.83e6, L_dif_max=0.01, c_pd=1005., aa=0.7, 
                bb=0.75, cc=5., dd=0.35):    
    '''Calculate latent and sensible heat flux using the bulk calculation 
    method 
    
    Parameters
    ----------
    T_0 : int 
        Steam point temperature
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
    nu  : float
        Kinematic viscosity of air
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
        Default is 0.622.
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
    
    Returns
    -------
    SHF_h : xarray.DataArray
        Sensible heat flux
    LHF_h : xarray.DataArray
        Latent heat flux
    '''   
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
                   
    return SHF_h, LHF_h

def calcVisc(T_h, T_0, rho_atm):    
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

def calcHumid(T_0, T_100, T_h, es_0, es_100, eps, p_h, RH_cor_h):
    '''Calculate specific humidity
    
    Parameters
    ----------
    T_0 : float 
        Steam point temperature. Default is 273.15.
    T_100 : float
        Steam point temperature in Kelvin
    T_h : xarray.DataArray
        Air temperature
    eps : int 
        DESCRIPTION
    es_0 : float
        Saturation vapour pressure at the melting point (hPa)
    es_100 : float
        Saturation vapour pressure at steam point temperature (hPa)
    p_h : xarray.DataArray
        Air pressure
    RH_cor_h : xarray.DataArray
        Relative humidity corrected
    
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
    
    # Convert to kg/kg
    return RH_cor_h * q_sat / 100 

def cleanHeatFlux(SHF, LHF, T, Tsurf, p, WS, RH_cor, z_boom):
    '''Find invalid heat flux data values and replace with NaNs, based on 
    air temperature, surface temperature, air pressure, wind speed, 
    corrected relative humidity, and boom height
    
    Parameters
    ----------
    SHF : xarray.DataArray
        Sensible heat flux
    LHF : xarray.DataArray
        Latent heat flux
    T : xarray.DataArray
        Air temperature
    Tsurf : xarray.DataArray
        Surface temperature
    p : xarray.DataArray
        Air pressure
    WS : xarray.DataArray
        Wind speed
    RH_cor : xarray.DataArray
        Relative humidity corrected
    z_boom : xarray.DataArray
        Boom height
    
    Returns
    -------
    SHF : xarray.DataArray
        Sensible heat flux corrected
    LHF : xarray.DataArray
        Latent heat flux corrected
    '''
    HF_nan = np.isnan(p) | np.isnan(T) | np.isnan(Tsurf) \
        | np.isnan(RH_cor) | np.isnan(WS) | np.isnan(z_boom)
    SHF[HF_nan] = np.nan
    LHF[HF_nan] = np.nan 
    return SHF, LHF
      
def cleanSpHumid(q_h, T, Tsurf, p, RH_cor):
    '''Find invalid specific humidity data values and replace with NaNs, 
    based on air temperature, surface temperature, air pressure, 
    and corrected relative humidity
    
    Parameters
    ----------
    q_h : xarray.DataArray
        Specific humidity
    T : xarray.DataArray
        Air temperature
    Tsurf : xarray.DataArray
        Surface temperature
    p : xarray.DataArray
        Air pressure
    RH_cor : xarray.DataArray
        Relative humidity corrected
    
    Returns
    -------
    q_h : xarray.DataArray
        Specific humidity corrected'''
    q_nan = np.isnan(T) | np.isnan(RH_cor) | np.isnan(p) | np.isnan(Tsurf)
    q_h[q_nan] = np.nan
    return q_h
 

def _calcAtmosDens(p_h, R_d, T_h, T_0):                                        # TODO: check this shouldn't be in this step somewhere
    '''Calculate atmospheric density'''
    return 100 * p_h / R_d / (T_h + T_0)

def _getTempK(T_0):
    '''Return steam point temperature in K'''
    return T_0+100
  
def _getRotation():
    '''Return degrees-to-radians and radians-to-degrees''' 
    deg2rad = np.pi / 180
    rad2deg = 1 / deg2rad
    return deg2rad, rad2deg
  
if __name__ == "__main__": 
    # unittest.main() 
    pass    
