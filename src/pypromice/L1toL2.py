#!/usr/bin/env python
"""
pypromice L1 to L2 processing
"""
import numpy as np
  
def toL2(L1, T_0=273.15, ews=1013.246, ei0=6.1071, eps_overcast=1., 
         eps_clear=9.36508e-6, emissivity=0.97):
    '''Process one Level 1 (L1) product to Level 2

    Parameters
    ----------
    L1 : xarray.Dataset
        Level 1 dataset
    T_0 : int, optional
        Steam point temperature. The default is 273.15.
    ews : int, optional
        Saturation pressure (normal atmosphere) at steam point temperature. 
        The default is 1013.246.
    ei0 : int, optional
        DESCRIPTION. The default is 6.1071.
    eps_overcast : int, optional
        Cloud overcast. The default is 1..
    eps_clear : int, optional
        Cloud clear. The default is 9.36508e-6.
    emissivity : int, optional
        Emissivity. The default is 0.97.

    Returns
    -------
    ds : xarray.Dataset
        Level 2 dataset
    '''
    ds = L1                                                                    # Reassign dataset  
    
    T_u = ds['t_u'].copy(deep=True)                                            # Correct relative humidity
    T_100_u = _getTempK(T_u)  
    ds['rh_u_cor'] = _correctHumidity(ds['rh_u'], ds['t_u'], T_u,  
                                    T_0, T_100_u, ews, ei0)                       
        
    # Determiune cloud cover
    cc = _calcCloudCoverage(T_u, T_0, eps_overcast, eps_clear,                 # Calculate cloud coverage
                              ds['dlr'], ds.attrs['station_id'])  
    ds['cc'] = (('time'), cc.data)
    
    # Determine surface temperature
    ds['t_surf'] = _calcSurfaceTemperature(T_0, ds['ulr'], ds['dlr'],          # Calculate surface temperature
                                           emissivity)
    
    # Determine station position relative to sun    
    doy = ds['time'].to_dataframe().index.dayofyear.values                     # Gather variables to calculate sun pos
    hour = ds['time'].to_dataframe().index.hour.values
    minute = ds['time'].to_dataframe().index.minute.values
    lat = ds.attrs['latitude']
    lon = ds.attrs['longitude']
    
    deg2rad, rad2deg = _getRotation()                                          # Get degree-radian conversions  
    phi_sensor_rad, theta_sensor_rad = _calcTilt(ds['tilt_x'], ds['tilt_y'],   # Calculate station tilt 
                                                 deg2rad)    
    Declination_rad = _calcDeclination(doy, hour, minute)                      # Calculate declination
    HourAngle_rad = _calcHourAngle(hour, minute, lon)                          # Calculate hour angle       
    ZenithAngle_rad, ZenithAngle_deg = _calcZenith(lat, Declination_rad,       # Calculate zenith
                                                   HourAngle_rad, deg2rad, 
                                                   rad2deg)
    
    # Correct Downwelling shortwave radiation
    DifFrac = 0.2 + 0.8 * cc 
    CorFac_all = _calcCorrectionFactor(Declination_rad, phi_sensor_rad,        # Calculate correction
                                       theta_sensor_rad, HourAngle_rad, 
                                       ZenithAngle_rad, ZenithAngle_deg, 
                                       lat, DifFrac, deg2rad)        
    ds['dsr_cor'] = ds['dsr'].copy(deep=True) * CorFac_all                     # Apply correction

    AngleDif_deg = _calcAngleDiff(ZenithAngle_rad, HourAngle_rad,              # Calculate angle between sun and sensor
                                  phi_sensor_rad, theta_sensor_rad)   
    
    ds['albedo'], OKalbedos = _calcAlbedo(ds['usr'], ds['dsr_cor'],                       # Determine albedo
                               AngleDif_deg, ZenithAngle_deg)

    # Correct upwelling and downwelling shortwave radiation
    sunonlowerdome =(AngleDif_deg >= 90) & (ZenithAngle_deg <= 90)             # Determine when sun is in FOV of lower sensor, assuming sensor measures only diffuse radiation
    ds['dsr_cor'] = ds['dsr_cor'].where(~sunonlowerdome, 
                                        other=ds['dsr'] / DifFrac)             # Apply to downwelling
    ds['usr_cor'] = ds['usr'].copy(deep=True)
    ds['usr_cor'] = ds['usr_cor'].where(~sunonlowerdome, 
                                        other=ds['albedo'] * ds['dsr'] / DifFrac) # Apply to upwelling
    bad = (ZenithAngle_deg > 95) | (ds['dsr_cor'] <= 0) | (ds['usr_cor'] <= 0) # Set to zero for solar zenith angles larger than 95 deg or either values are (less than) zero
    ds['dsr_cor'][bad] = 0
    ds['usr_cor'][bad] = 0
    ds['dsr_cor'] = ds['usr_cor'].copy(deep=True) / ds['albedo']               # Correct DWR using more reliable USWR when sun not in sight of upper sensor  
    ds['albedo'] = ds['albedo'].where(OKalbedos)                               #TODO remove?      
    
    # Remove data where TOA shortwave radiation invalid
    isr_toa = _calcTOA(ZenithAngle_deg, ZenithAngle_rad)                       # Calculate TOA shortwave radiation                         
    TOA_crit_nopass = (ds['dsr_cor'] > (0.9 * isr_toa + 10))                   # Determine filter
    ds['dsr_cor'][TOA_crit_nopass] = np.nan                                    # Apply filter and interpolate
    ds['usr_cor'][TOA_crit_nopass] = np.nan
    ds['dsr_cor'] = ds['dsr_cor'].interpolate_na(dim='time', use_coordinate=False)
    ds['usr_cor'] = ds['usr_cor'].interpolate_na(dim='time', use_coordinate=False)

    # # Check sun position
    # sundown = ZenithAngle_deg >= 90
    # _checkSunPos(ds, OKalbedos, sundown, sunonlowerdome, TOA_crit_nopass)    
    
    if ds.attrs['number_of_booms']==2:      
        T_l = ds['t_l'].copy(deep=True) 
        T_100_l = _getTempK(T_l)                                               # Get steam point temperature in K
        ds['rh_l_cor'] = _correctHumidity(ds['rh_l'], ds['t_l'], T_l,          # Correct relative humidity
                                        T_0, T_100_l, ews, ei0)                          
    
    if hasattr(ds,'t_i'):       
        if ~ds['t_i'].isnull().all():                                        # Instantaneous msg processing
            T_i = ds['t_i'].copy(deep=True) 
            T_100_i = _getTempK(T_i)                                           # Get steam point temperature in K
            ds['rh_i_cor'] = _correctHumidity(ds['rh_i'], ds['t_i'], T_i,      # Correct relative humidity
                                            T_0, T_100_i, ews, ei0)                   
    return ds


def _getTempK(T_0):                                                            #TODO same as L2toL3._getTempK()
    '''Return steam point temperature in K'''
    return T_0+100

def _getRotation():                                                            #TODO same as L2toL3._getRotation()
    '''Return degrees-to-radians and radians-to-degrees''' 
    deg2rad = np.pi / 180
    rad2deg = 1 / deg2rad
    return deg2rad, rad2deg

def _calcCloudCoverage(T, T_0, eps_overcast, eps_clear, dlr, station_id):
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

def _calcSurfaceTemperature(T_0, ulr, dlr, emissivity):
    '''Calculate surface temperature from air temperature, upwelling and 
    downwelling radiation and emissivity
    
    Parameters
    ----------
    T_0 : xarray.DataArray
        Air temperature
    ulr : xarray.DataArray
        Upwelling longwave radiation
    dlr : xarray.DataArray
        Downwelling longwave radiation
    emissivity : int
        Assumed emissivity
    
    Returns
    -------
    xarray.DataArray
        Calculated surface temperature
    '''
    t_surf = ((ulr - (1 - emissivity) * dlr) / emissivity / 5.67e-8)**0.25 - T_0
    return t_surf.where(t_surf <= 0, other = 0)
    
def _calcTilt(tilt_x, tilt_y, deg2rad):
    '''Calculate station tilt'''
    # Tilt as radians
    tx = tilt_x * deg2rad
    ty = tilt_y * deg2rad
    
    # Calculate cartesian coordinates
    X = np.sin(tx) * np.cos(tx) * np.sin(ty)**2 + np.sin(tx) * np.cos(ty)**2
    Y = np.sin(ty) * np.cos(ty) * np.sin(tx)**2 + np.sin(ty) * np.cos(tx)**2
    Z = np.cos(tx) * np.cos(ty) + np.sin(tx)**2 * np.sin(ty)**2
 
    # Calculate spherical coordinates
    phi_sensor_rad = -np.pi /2 - np.arctan(Y/X)
    phi_sensor_rad[X > 0] += np.pi
    phi_sensor_rad[(X == 0) & (Y < 0)] = np.pi
    phi_sensor_rad[(X == 0) & (Y == 0)] = 0
    phi_sensor_rad[phi_sensor_rad < 0] += 2*np.pi

    # Total tilt of the sensor, i.e. 0 when horizontal
    theta_sensor_rad = np.arccos(Z / (X**2 + Y**2 + Z**2)**0.5) 
    # phi_sensor_deg = phi_sensor_rad * rad2deg                                #TODO take these out if not needed
    # theta_sensor_deg = theta_sensor_rad * rad2deg
    return phi_sensor_rad, theta_sensor_rad 

def _checkSunPos(ds, OKalbedos, sundown, sunonlowerdome, TOA_crit_nopass):
    '''Check sun position'''
    valid = (~(ds['dsr_cor'].isnull())).sum()
    print('Sun in view of upper sensor / workable albedos:', OKalbedos.sum().values,
          (100*OKalbedos.sum()/valid).round().values, "%")
    print('Sun below horizon:', sundown.sum(),
          (100*sundown.sum()/valid).round().values, "%")
    print('Sun in view of lower sensor:', sunonlowerdome.sum().values,
          (100*sunonlowerdome.sum()/valid).round().values, "%")
    print('Spikes removed using TOA criteria:', TOA_crit_nopass.sum().values,
          (100*TOA_crit_nopass.sum()/valid).round().values, "%")
    print('Mean net SR change by corrections:',
          (ds['dsr_cor']-ds['usr_cor']-ds['dsr']+ds['usr']).sum().values/valid.values,
          "W/m2")

def _correctHumidity(rh, t_1, T, T_0, T_100, ews, ei0):                        #TODO figure out if T replicate is needed
    '''Correct relative humidity using Groff & Gratch method

    Parameters
    ----------
    rh : xarray.DataArray
        Relative humidity
    t_1 : xarray.DataArray
        Air temperature
    T : xarray.DataArray
        Air temperature replicate
    T_0 : int
        Steam point temperature
    T_100 : int
        Steam point temperature in K
    ews : int
        Saturation pressure (normal atmosphere) at steam point temperature
    ei0 : int
        DESCRIPTION
        
    Returns
    -------
    xarray.DataArray
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
    freezing = (t_1 < 0) & (t_1 > -100).values 

    # Set to Groff & Gratch values when freezing, otherwise just rh                         
    return rh.where(~freezing, other = rh*(e_s_wtr / e_s_ice))                                 
                                  
def _calcDeclination(doy, hour, minute):
    '''Calculate sun declination based on time'''
    d0_rad = 2 * np.pi * (doy + (hour + minute / 60) / 24 -1) / 365
    return np.arcsin(0.006918 - 0.399912
                     * np.cos(d0_rad) + 0.070257
                     * np.sin(d0_rad) - 0.006758
                     * np.cos(2 * d0_rad) + 0.000907
                     * np.sin(2 * d0_rad) - 0.002697
                     * np.cos(3 * d0_rad) + 0.00148
                     * np.sin(3 * d0_rad))

def _calcHourAngle(hour, minute, lon):
    '''Calculate hour angle of sun based on time and longitude. Make sure that
    time is set to UTC and longitude is positive when west. Hour angle should
    be 0 at noon'''
    return 2 * np.pi * (((hour + minute / 60) / 24 - 0.5) - lon/360)
     # ; - 15.*timezone/360.)

def _calcDirectionDeg(HourAngle_rad):                                          #TODO remove if not plan to use this
    '''Calculate sun direction as degrees. This is an alternative to 
    _calcHourAngle that is currently not implemented into the offical L0>>L3
    workflow. Here, 180 degrees is at noon (NH), as opposed to HourAngle'''
    DirectionSun_deg = HourAngle_rad * 180/np.pi - 180
    DirectionSun_deg[DirectionSun_deg < 0] += 360
    DirectionSun_deg[DirectionSun_deg < 0] += 360
    return DirectionSun_deg

def _calcZenith(lat, Declination_rad, HourAngle_rad, deg2rad, rad2deg):
    '''Calculate sun zenith in radians and degrees'''
    ZenithAngle_rad = np.arccos(np.cos(lat * deg2rad)
                                * np.cos(Declination_rad)
                                * np.cos(HourAngle_rad)
                                + np.sin(lat * deg2rad)
                                * np.sin(Declination_rad))
    
    ZenithAngle_deg = ZenithAngle_rad * rad2deg   
    return ZenithAngle_rad, ZenithAngle_deg

def _calcAngleDiff(ZenithAngle_rad, HourAngle_rad, phi_sensor_rad, 
                  theta_sensor_rad):
    '''Calculate angle between sun and upper sensor (to determine when sun is
    in sight of upper sensor'''
    return 180 / np.pi * np.arccos(np.sin(ZenithAngle_rad) 
                                   * np.cos(HourAngle_rad + np.pi)
                                   * np.sin(theta_sensor_rad)
                                   * np.cos(phi_sensor_rad)
                                   + np.sin(ZenithAngle_rad)
                                   * np.sin(HourAngle_rad + np.pi)
                                   * np.sin(theta_sensor_rad)
                                   * np.sin(phi_sensor_rad)
                                   + np.cos(ZenithAngle_rad)
                                   * np.cos(theta_sensor_rad))  

def _calcAlbedo(usr, dsr_cor, AngleDif_deg, ZenithAngle_deg):
    '''Calculate surface albedo based on upwelling and downwelling shorwave 
    flux, the angle between the sun and sensor, and the sun zenith'''
    albedo = usr / dsr_cor    
    
    # NaN bad data
    OKalbedos = (AngleDif_deg < 70) & (ZenithAngle_deg < 70) & (albedo < 1) & (albedo > 0)    
    albedo[~OKalbedos] = np.nan             
    
    # Interpolate all. Note "use_coordinate=False" is used here to force 
    # comparison against the GDL code when that is run with *only* a TX file. 
    # Should eventually set to default (True) and interpolate based on time, 
    # not index.                                           
    albedo = albedo.interpolate_na(dim='time', use_coordinate=False)           
    albedo = albedo.ffill(dim='time').bfill(dim='time')                        #TODO remove this line and one above?
    return albedo, OKalbedos
    
def _calcTOA(ZenithAngle_deg, ZenithAngle_rad):
    '''Calculate incoming shortwave radiation at the top of the atmosphere,
    accounting for sunset periods'''
    sundown = ZenithAngle_deg >= 90
    
    # Incoming shortware radiation at the top of the atmosphere
    isr_toa = 1372 * np.cos(ZenithAngle_rad) 
    isr_toa[sundown] = 0
    return isr_toa 
    
def _calcCorrectionFactor(Declination_rad, phi_sensor_rad, theta_sensor_rad, 
                          HourAngle_rad, ZenithAngle_rad, ZenithAngle_deg, 
                          lat, DifFrac, deg2rad):    
    '''Calculate correction factor for direct beam radiation, as described 
    here: http://solardat.uoregon.edu/SolarRadiationBasics.html
    
    Offset correction (where solar zenith angles larger than 110 degrees) not
    implemented as it should not improve the accuracy of well-calibrated
    instruments. It would go something like this:
    ds['dsr'] = ds['dsr'] - ds['dwr_offset']
    SRout = SRout - SRout_offset
    '''
    CorFac = np.sin(Declination_rad) * np.sin(lat * deg2rad) \
        * np.cos(theta_sensor_rad) \
        - np.sin(Declination_rad) \
        * np.cos(lat * deg2rad) \
        * np.sin(theta_sensor_rad) \
        * np.cos(phi_sensor_rad + np.pi) \
        + np.cos(Declination_rad) \
        * np.cos(lat * deg2rad) \
        * np.cos(theta_sensor_rad) \
        * np.cos(HourAngle_rad) \
        + np.cos(Declination_rad) \
        * np.sin(lat * deg2rad) \
        * np.sin(theta_sensor_rad) \
        * np.cos(phi_sensor_rad + np.pi) \
        * np.cos(HourAngle_rad) \
        + np.cos(Declination_rad) \
        * np.sin(theta_sensor_rad) \
        * np.sin(phi_sensor_rad + np.pi) \
        * np.sin(HourAngle_rad) \
    
    CorFac = np.cos(ZenithAngle_rad) / CorFac
    # sun out of field of view upper sensor
    CorFac[(CorFac < 0) | (ZenithAngle_deg > 90)] = 1
    
    # Calculating ds['dsr'] over a horizontal surface corrected for station/sensor tilt
    CorFac_all = CorFac / (1 - DifFrac + CorFac * DifFrac)
    
    return CorFac_all
    
 
if __name__ == "__main__": 
    # unittest.main() 
    pass 