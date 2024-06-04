#!/usr/bin/env python
"""
AWS Level 2 (L2) to Level 3 (L3) data processing
"""
import numpy as np
import xarray as xr
from statsmodels.nonparametric.smoothers_lowess import lowess
import logging
logger = logging.getLogger(__name__)

def toL3(L2, T_0=273.15):
    '''Process one Level 2 (L2) product to Level 3 (L3) meaning calculating all
    derived variables:
        - Turbulent fluxes
        - smoothed and inter/extrapolated GPS coordinates
    
    
    Parameters
    ----------
    L2 : xarray:Dataset
        L2 AWS data
    T_0 : int 
        Steam point temperature. Default is 273.15.
    '''
    ds = L2

    T_100 = _getTempK(T_0)                                                     # Get steam point temperature as K 
    
    # Turbulent heat flux calculation
    # Upper boom bulk calculation
    T_h_u = ds['t_u'].copy()                                                   # Copy for processing
    p_h_u = ds['p_u'].copy()
    WS_h_u = ds['wspd_u'].copy()
    RH_cor_h_u = ds['rh_u_cor'].copy()
    Tsurf_h = ds['t_surf'].copy()                                              # T surf from derived upper boom product. TODO is this okay to use with lower boom parameters?
    z_WS_u = ds['z_boom_u'].copy() + 0.4                                       # Get height of Anemometer                
    z_T_u = ds['z_boom_u'].copy() - 0.1                                        # Get height of thermometer  
        
    q_h_u = calcSpHumid(T_0, T_100, T_h_u, p_h_u, RH_cor_h_u)                  # Calculate specific humidity
    
    if not ds.attrs['bedrock']:       
        SHF_h_u, LHF_h_u= calcHeatFlux(T_0, T_h_u, Tsurf_h, WS_h_u,            # Calculate latent and sensible heat fluxes
                                        z_WS_u, z_T_u, q_h_u, p_h_u)     

        ds['dshf_u'] = (('time'), SHF_h_u.data)
        ds['dlhf_u'] = (('time'), LHF_h_u.data)

    q_h_u = 1000 * q_h_u                                                       # Convert sp.humid from kg/kg to g/kg
    ds['qh_u'] = (('time'), q_h_u.data)    

    # Lower boom bulk calculation
    if ds.attrs['number_of_booms']==2:
        T_h_l = ds['t_l'].copy()                                               # Copy for processing
        p_h_l = ds['p_l'].copy()
        WS_h_l = ds['wspd_l'].copy()                                      
        RH_cor_h_l = ds['rh_l_cor'].copy()
        z_WS_l = ds['z_boom_l'].copy() + 0.4                                   # Get height of W                  
        z_T_l = ds['z_boom_l'].copy() - 0.1                                    # Get height of thermometer 
        
        q_h_l = calcSpHumid(T_0, T_100, T_h_l, p_h_l, RH_cor_h_l)              # Calculate sp.humidity
                           
        if not ds.attrs['bedrock']:       
            SHF_h_l, LHF_h_l= calcHeatFlux(T_0, T_h_l, Tsurf_h, WS_h_l, # Calculate latent and sensible heat fluxes 
                                            z_WS_l, z_T_l, q_h_l, p_h_l)        

            ds['dshf_l'] = (('time'), SHF_h_l.data)
            ds['dlhf_l'] = (('time'), LHF_h_l.data)
        q_h_l = 1000 * q_h_l                                                   # Convert sp.humid from kg/kg to g/kg

        ds['qh_l'] = (('time'), q_h_l.data)    
    
    # Smoothing and inter/extrapolation of GPS coordinates

    for var in ['gps_lat', 'gps_lon', 'gps_alt']:
        logger.info('Postprocessing '+var)

        # saving the static value and droping 'lat','lon' or 'alt' as they are 
        # being reassigned as timeseries
        var_out = var.replace('gps_','')
        
        if var_out == 'alt':
            if 'altitude' in list(ds.attrs.keys()):
                static_value = float(ds.attrs['altitude'])
            else:
                print('no standard altitude for', ds.station_id)
                static_value = np.nan
        elif  var_out == 'lat':
            static_value = float(ds.attrs['latitude'])
        elif  var_out == 'lon':
            static_value = float(ds.attrs['longitude'])
        ds=ds.drop_vars(var_out)
        
        # if there is no gps observations, then we use the static value repeated
        # for each time stamp
        if var not in ds.data_vars: 
            print('no',var,'at', ds.station_id)
            ds[var_out] = ('time', np.ones_like(ds['t_u'].data)*static_value)
            ds[var_out+'_avg'] = static_value
            continue
        
        if ds[var].isnull().all():
            print('no',var,'at',ds.station_id)
            ds[var_out] = ('time', np.ones_like(ds['t_u'].data)*static_value)
            ds[var_out+'_avg'] = static_value
            continue
        
        # here we detect potential relocation of the station in the form of a 
        # break in the general trend of the latitude, longitude and altitude
        # in the future, this could/should be listed in an external file to 
        # avoid missed relocations or sensor issues interpreted as a relocation
        if var == 'gps_alt':
            _, breaks = find_breaks(ds[var].to_series(), alpha=8)
        else:
            _, breaks = find_breaks(ds[var].to_series(), alpha=6)
        
        # smoothing and inter/extrapolation of the coordinate
        ds[var_out] = \
            ('time',  piecewise_smoothing_and_interpolation(ds[var].to_series(), breaks))
        
        ds['lat_avg'] = ds['lat'].mean()
        ds['lon_avg'] = ds['lon'].mean()
        ds['alt_avg'] = ds['alt'].mean()
    return ds


def find_breaks(df,alpha):
    '''Detects potential relocation of the station from the GPS measurements.
    The code first makes a forward linear interpolation of the coordinates and
    then looks for important jumps in latitude, longitude and altitude. The jumps
    that are higher than a given threshold (expressed as a multiple of the 
    standard deviation) are mostly caused by the station being moved during
    maintenance. To avoid misclassification, only the jumps detected in May-Sept. 
    are kept.
    
    Parameters
    ----------
    df : pandas.Series
        series of observed latitude, longitude or elevation
    alpha: float
        coefficient to be applied to the the standard deviation of the daily
        coordinate fluctuation
    '''
    diff = df.resample('D').median().interpolate(
        method='linear', limit_area='inside', limit_direction='forward').diff()        
    thresh = diff.std() * alpha
    list_diff = diff.loc[diff.abs()>thresh].reset_index()
    list_diff = list_diff.loc[list_diff.time.dt.month.isin([5,6,7,8,9])]
    list_diff['year']=list_diff.time.dt.year
    list_diff=list_diff.groupby('year').max()
    return diff, [None]+list_diff.time.to_list()+[None]


def piecewise_smoothing_and_interpolation(df_in, breaks):
    '''Smoothes, inter- or extrapolate the gps observations. The processing is 
    done piecewise so that each period between station relocation are done 
    separately (no smoothing of the jump due to relocation). Locally Weighted
    Scatterplot Smoothing (lowess) is then used to smooth the available
    observations. Then this smoothed curve is interpolated linearly over internal
    gaps. Eventually, this interpolated curve is extrapolated linearly for 
    timestamps before the first valid measurement and after the last valid
    measurement.
    
    Parameters
    ----------
    df_in : pandas.Series
        series of observed latitude, longitude or elevation
    breaks: list
        List of timestamps of station relocation. First and last item should be
        None so that they can be used in slice(breaks[i], breaks[i+1])
    '''
    df_all = pd.Series() # dataframe gathering all the smoothed pieces
    for i in range(len(breaks)-1):
        df = df_in.loc[slice(breaks[i], breaks[i+1])].copy()
        
        y_sm = lowess(df,
                      pd.to_numeric(df.index),
                      is_sorted=True, frac=1/3, it=0,
                      )
        df.loc[df.notnull()] = y_sm[:,1]
        df = df.interpolate(method='linear', limit_area='inside')
        
        last_valid_6_months = slice(df.last_valid_index()-pd.to_timedelta('180D'),None)
        df.loc[last_valid_6_months] = (df.loc[last_valid_6_months].interpolate( axis=0,
            method='spline',order=1, limit_direction='forward', fill_value="extrapolate")).values
        
        first_valid_6_months = slice(None, df.first_valid_index()+pd.to_timedelta('180D'))
        df.loc[first_valid_6_months] = (df.loc[first_valid_6_months].interpolate( axis=0,
            method='spline',order=1, limit_direction='backward', fill_value="extrapolate")).values
        df_all=pd.concat((df_all, df))
        
    df_all = df_all[~df_all.index.duplicated(keep='first')]
    return df_all.values  
    

def calcHeatFlux(T_0, T_h, Tsurf_h, WS_h, z_WS, z_T, q_h, p_h, 
                kappa=0.4, WS_lim=1., z_0=0.001, g=9.82, es_0=6.1071, eps=0.622, 
                gamma=16., L_sub=2.83e6, L_dif_max=0.01, c_pd=1005., aa=0.7, 
                bb=0.75, cc=5., dd=0.35, R_d=287.05):    
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
    es_100 : int
        Saturation vapour pressure at steam point temperature (hPa). Default is 
        1013.246.        
    eps : int 
        Ratio of molar masses of vapor and dry air (0.622).
    R_d : int 
        Gas constant of dry air. Default is 287.05.
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
    z_0 : int
        Aerodynamic surface roughness length for momention, assumed constant 
        for all ice/snow surfaces. Default is 0.001.
    
    Returns
    -------
    SHF_h : xarray.DataArray
        Sensible heat flux
    LHF_h : xarray.DataArray
        Latent heat flux
    '''   
    rho_atm = 100 * p_h / R_d / (T_h + T_0)                              # Calculate atmospheric density                                  
    nu = calcVisc(T_h, T_0, rho_atm)                                     # Calculate kinematic viscosity  
    
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

def calcSpHumid(T_0, T_100, T_h, p_h, RH_cor_h, es_0=6.1071, es_100=1013.246, eps=0.622):
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
        ratio of molar masses of vapor and dry air (0.622)
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
    
    q_nan = np.isnan(T_h) | np.isnan(p_h)
    q_sat[q_nan] = np.nan

    # Convert to kg/kg
    return RH_cor_h * q_sat / 100 


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
