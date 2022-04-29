#!/usr/bin/env python

import numpy as np
import pandas as pd

def to_L2(L1=None):

    ds = L1
    
    deg2rad = np.pi / 180
    rad2deg = 1 / deg2rad
    T_0 = 273.15
    
    T_100 = T_0+100            # steam point temperature in K
    ews = 1013.246             # saturation pressure at steam point temperature, normal atmosphere
    ei0 = 6.1071
    
    T = ds['t_1'].copy(deep=True)
    
    # in hPa (Goff & Gratch)
    e_s_wtr = 10**(-7.90298 * (T_100 / (T + T_0) - 1)
                   + 5.02808 * np.log10(T_100 / (T + T_0)) 
                   - 1.3816E-7 * (10**(11.344 * (1 - (T + T_0) / T_100)) - 1)
                   + 8.1328E-3 * (10**(-3.49149 * (T_100/(T + T_0) - 1)) -1)
                   + np.log10(ews))
    
    # in hPa (Goff & Gratch)
    e_s_ice = 10**(-9.09718 * (T_0 / (T + T_0) - 1)
                   - 3.56654 * np.log10(T_0 / (T + T_0))
                   + 0.876793 * (1 - (T + T_0) / T_0)
                   + np.log10(ei0))
    
    # ds['rh_cor'] = (e_s_wtr / e_s_ice) * ds['rh'].where((ds['t_1'] < 0) & (ds['t_1'] > -100))
    freezing = (ds['t_1'] < 0) & (ds['t_1'] > -100).values # why > -100?
    # set to Geoff & Gratch values when freezing, otherwise just rh.
    ds['rh_cor'] = ds['rh'].where(~freezing, other = ds['rh']*(e_s_wtr / e_s_ice))
    
    # https://github.com/GEUS-PROMICE/PROMICE-AWS-processing/issues/23
    # Just adding special treatment here in service of replication. rh_cor is clipped not NaN'd
    # https://github.com/GEUS-PROMICE/PROMICE-AWS-processing/issues/20
    df = pd.read_csv("./variables.csv", index_col=0, comment="#", usecols=('field','lo','hi','OOL'))
    df = df.dropna(how='all')
    for var in df.index:
        if var not in list(ds.variables): continue
        if var == 'rh_cor':
             ds[var] = ds[var].where(ds[var] >= df.loc[var, 'lo'], other = 0)
             ds[var] = ds[var].where(ds[var] <= df.loc[var, 'hi'], other = 100)
        else:
            ds[var] = ds[var].where(ds[var] >= df.loc[var, 'lo'])
            ds[var] = ds[var].where(ds[var] <= df.loc[var, 'hi'])
        other_vars = df.loc[var]['OOL'] # either NaN or "foo" or "foo bar baz ..."
        if isinstance(other_vars, str): 
            for o in other_vars.split():
                if o not in list(ds.variables): continue
                ds[o] = ds[o].where(ds[var] >= df.loc[var, 'lo'])
                ds[o] = ds[o].where(ds[var] <= df.loc[var, 'hi'])
    
    eps_overcast = 1.
    eps_clear = 9.36508e-6
    LR_overcast = eps_overcast * 5.67e-8 *(T + T_0)**4   # assumption
    LR_clear = eps_clear * 5.67e-8 * (T + T_0)**6        # Swinbank (1963)
    
    # Special case for selected stations (will need this for all stations eventually)
    if ds.attrs['station_id'] == 'KAN_M':
       # print,'KAN_M cloud cover calculations'
       LR_overcast = 315 + 4*T
       LR_clear = 30 + 4.6e-13 * (T + T_0)**6
    elif ds.attrs['station_id'] == 'KAN_U':
       # print,'KAN_U cloud cover calculations'
       LR_overcast = 305 + 4*T
       LR_clear = 220 + 3.5*T
    
    cc = (ds['dlr'] - LR_clear) / (LR_overcast - LR_clear)
    cc[cc > 1] = 1
    cc[cc < 0] = 0
    DifFrac = 0.2 + 0.8 * cc
    
    ds['cc'] = (('time'), cc.data)
    
    emissivity = 0.97
    ds['t_surf'] = ((ds['ulr'] - (1 - emissivity) * ds['dlr']) / emissivity / 5.67e-8)**0.25 - T_0
    ds['t_surf'] = ds['t_surf'].where(ds['t_surf'] <= 0, other = 0) # if > 0, set to 0
    tx = ds['tilt_x'] * deg2rad
    ty = ds['tilt_y'] * deg2rad
    
    ## cartesian coords
    X = np.sin(tx) * np.cos(tx) * np.sin(ty)**2 + np.sin(tx) * np.cos(ty)**2
    Y = np.sin(ty) * np.cos(ty) * np.sin(tx)**2 + np.sin(ty) * np.cos(tx)**2
    Z = np.cos(tx) * np.cos(ty) + np.sin(tx)**2 * np.sin(ty)**2
    
    # spherical coords
    phi_sensor_rad = -np.pi /2 - np.arctan(Y/X)
    phi_sensor_rad[X > 0] += np.pi
    phi_sensor_rad[(X == 0) & (Y < 0)] = np.pi
    phi_sensor_rad[(X == 0) & (Y == 0)] = 0
    phi_sensor_rad[phi_sensor_rad < 0] += 2*np.pi
    
    phi_sensor_deg = phi_sensor_rad * rad2deg
    
    # spherical coordinate (or actually total tilt of the sensor, i.e. 0 when horizontal)
    theta_sensor_rad = np.arccos(Z / (X**2 + Y**2 + Z**2)**0.5) 
    theta_sensor_deg = theta_sensor_rad * rad2deg
    
    ## Offset correction (determine offset yourself using data for solar
    ## zenith angles larger than 110 deg) I actually don't do this as it
    ## shouldn't improve accuracy for well calibrated instruments
    # ;ds['dsr'] = ds['dsr'] - ds['dwr_offset']
    # ;SRout = SRout - SRout_offset
    
    # Calculating zenith and hour angle of the sun
    doy = ds['time'].to_dataframe().index.dayofyear.values
    hour = ds['time'].to_dataframe().index.hour.values
    minute = ds['time'].to_dataframe().index.minute.values
    # lat = ds['gps_lat']
    # lon = ds['gps_lon']
    lat = ds.attrs['latitude']
    lon = ds.attrs['longitude']
    
    d0_rad = 2 * np.pi * (doy + (hour + minute / 60) / 24 -1) / 365
    
    Declination_rad = np.arcsin(0.006918 - 0.399912
                                * np.cos(d0_rad) + 0.070257
                                * np.sin(d0_rad) - 0.006758
                                * np.cos(2 * d0_rad) + 0.000907
                                * np.sin(2 * d0_rad) - 0.002697
                                * np.cos(3 * d0_rad) + 0.00148
                                * np.sin(3 * d0_rad))
    
    HourAngle_rad = 2 * np.pi * (((hour + minute / 60) / 24 - 0.5) - lon/360)
    # ; - 15.*timezone/360.) ; NB: Make sure time is in UTC and longitude is positive when west! Hour angle should be 0 at noon.
    
    # This is 180 deg at noon (NH), as opposed to HourAngle.
    DirectionSun_deg = HourAngle_rad * 180/np.pi - 180
    
    DirectionSun_deg[DirectionSun_deg < 0] += 360
    DirectionSun_deg[DirectionSun_deg < 0] += 360
    
    ZenithAngle_rad = np.arccos(np.cos(lat * deg2rad)
                                * np.cos(Declination_rad)
                                * np.cos(HourAngle_rad)
                                + np.sin(lat * deg2rad)
                                * np.sin(Declination_rad))
    
    ZenithAngle_deg = ZenithAngle_rad * rad2deg
    
    sundown = ZenithAngle_deg >= 90
    isr_toa = 1372 * np.cos(ZenithAngle_rad) # Incoming shortware radiation at the top of the atmosphere
    isr_toa[sundown] = 0
    
    # Calculating the correction factor for direct beam radiation
    # http://solardat.uoregon.edu/SolarRadiationBasics.html
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
    ds['dsr_cor'] = ds['dsr'].copy(deep=True) * CorFac_all
    
    # Calculating albedo based on albedo values when sun is in sight of the upper sensor
    AngleDif_deg = 180 / np.pi * np.arccos(np.sin(ZenithAngle_rad)
                                           * np.cos(HourAngle_rad + np.pi)
                                           * np.sin(theta_sensor_rad)
                                           * np.cos(phi_sensor_rad)
                                           + np.sin(ZenithAngle_rad)
                                           * np.sin(HourAngle_rad + np.pi)
                                           * np.sin(theta_sensor_rad)
                                           * np.sin(phi_sensor_rad)
                                           + np.cos(ZenithAngle_rad)
                                           * np.cos(theta_sensor_rad)) # angle between sun and sensor
    
    # ds['add'] = (('time'),AngleDif_deg)
    # ds['zar'] = (('time'),ZenithAngle_rad)
    # ds['har'] = (('time'),HourAngle_rad)
    # ds['tsr'] = (('time'),theta_sensor_rad)
    # ds['X'] = (('time'),X)
    # ds['Y'] = (('time'),Y)
    # ds['Z'] = (('time'),Z)
    # from IPython import embed; embed()
    # ds[['dsr','dsr_cor','usr','add','X','Y','Z','tilt_x','tilt_y']].to_dataframe().head(40)
    
    
    # ;AngleDif_deg = 180./!pi*acos(cos(!pi/2.-ZenithAngle_rad)*cos(!pi/2.-theta_sensor_rad)*cos(HourAngle_rad-phi_sensor_rad)+sin(!pi/2.-ZenithAngle_rad)*sin(!pi/2.-theta_sensor_rad)) ; angle between sun and sensor
    
    # from IPython import embed; embed()
    
    ds['albedo'] = ds['usr'] / ds['dsr_cor']
    # albedo_nan = np.isnan(ds['albedo']) # store existing NaN
    OKalbedos = (AngleDif_deg < 70) & (ZenithAngle_deg < 70) & (ds['albedo'] < 1) & (ds['albedo'] > 0)
    ds['albedo'][~OKalbedos] = np.nan
    
    # NOTE: "use_coordinate=False" is used here to force comparison against the GDL code when that is run with *only* a TX file.
    # Should eventually set to default (True) and interpolate based on time, not index.
    ds['albedo'] = ds['albedo'].interpolate_na(dim='time', use_coordinate=False) # Interpolate all NaN (old and new NotOK)
    ds['albedo'] = ds['albedo'].ffill(dim='time').bfill(dim='time')
    # TODO: Remove above?
    
    # ds['albedo'] = ds['albedo'].ffill(dim='time') # Interpolate all NaN (old and new NotOK)
    # ds['albedo'][albedo_nan] = np.nan # restore old NaN
    
    # ;OKalbedos = where(angleDif_deg lt 82.5 and ZenithAngle_deg lt 70 and albedo lt 1 and albedo gt 0, complement=notOKalbedos)
    # ;The running mean calculation doesn't work for non-continuous data sets or variable temporal resolution (e.g. with multiple files)
    # ;albedo_rm = 0*albedo
    # ;albedo_rm[OKalbedos] = smooth(albedo[OKalbedos],obsrate+1,/edge_truncate) ; boxcar average of reliable albedo values
    # ;albedo[notOKalbedos] = interpol(albedo_rm[OKalbedos],OKalbedos,notOKalbedos) ; interpolate over gaps
    # ;albedo_rm[notOKalbedos] = albedo[notOKalbedos]
    # ;So instead:
    
    # albedo[notOKalbedos] = interpol(albedo[OKalbedos],OKalbedos,notOKalbedos) ; interpolate over gaps - gives problems for discontinuous data sets, but is not the end of the world
    
    # Correcting SR using DWR when sun is in field of view of lower sensor assuming sensor measures only diffuse radiation
    sunonlowerdome =(AngleDif_deg >= 90) & (ZenithAngle_deg <= 90)
    # ds['dsr_cor'][sunonlowerdome] = ds['dsr'][sunonlowerdome] / DifFrac[sunonlowerdome]
    ds['dsr_cor'] = ds['dsr_cor'].where(~sunonlowerdome, other=ds['dsr'] / DifFrac)
    ds['usr_cor'] = ds['usr'].copy(deep=True)
    # ds['usr_cor'][sunonlowerdome] = albedo * ds['dsr'][sunonlowerdome] / DifFrac[sunonlowerdome]
    ds['usr_cor'] = ds['usr_cor'].where(~sunonlowerdome, other=ds['albedo'] * ds['dsr'] / DifFrac)
    
    
    # Setting DWR and USWR to zero for solar zenith angles larger than 95 deg or either DWR or USWR are (less than) zero
    bad = (ZenithAngle_deg > 95) | (ds['dsr_cor'] <= 0) | (ds['usr_cor'] <= 0)
    ds['dsr_cor'][bad] = 0
    ds['usr_cor'][bad] = 0
    
    # Correcting DWR using more reliable USWR when sun not in sight of upper sensor
    ds['dsr_cor'] = ds['usr_cor'].copy(deep=True) / ds['albedo']
    # albedo[~OKalbedos] = np.nan
    ds['albedo'] = ds['albedo'].where(OKalbedos)
    # albedo[OKalbedos[n_elements(OKalbedos)-1]:*] = -999 ; Removing albedos that were extrapolated (as opposed to interpolated) at the end of the time series - see above
    # ds['dsr']_cor[OKalbedos[n_elements(OKalbedos)-1]:*] = -999 ; Removing the corresponding ds['dsr']_cor as well
    # ds['uswr_cor'][OKalbedos[n_elements(OKalbedos)-1]:*] = -999 ; Removing the corresponding ds['uswr_cor'] as well
    
    # ; Removing spikes by interpolation based on a simple top-of-the-atmosphere limitation
    #      TOA_crit_nopass = where(ds['dsr']_cor gt 0.9*dwr_toa+10)
    #      TOA_crit_pass = where(ds['dsr']_cor le 0.9*dwr_toa+10)
    #      if total(TOA_crit_nopass) ne -1 then begin
    #         ds['dsr']_cor[TOA_crit_nopass] = interpol(ds['dsr']_cor[TOA_crit_pass],TOA_crit_pass,TOA_crit_nopass)
    #         ds['uswr_cor'][TOA_crit_nopass] = interpol(ds['uswr_cor'][TOA_crit_pass],TOA_crit_pass,TOA_crit_nopass)
    #      endif
    TOA_crit_nopass = (ds['dsr_cor'] > (0.9 * isr_toa + 10))
    
    ds['dsr_cor'][TOA_crit_nopass] = np.nan
    ds['usr_cor'][TOA_crit_nopass] = np.nan
    ds['dsr_cor'] = ds['dsr_cor'].interpolate_na(dim='time', use_coordinate=False)
    ds['usr_cor'] = ds['usr_cor'].interpolate_na(dim='time', use_coordinate=False)
    #ds['dsr_cor'] = ds['dsr_cor'].ffill(dim='time')
    #ds['usr_cor'] = ds['usr_cor'].ffill(dim='time')
    # ds['dsr_cor'] = ds['dsr_cor'].interpolate_na(dim='time', method='linear', limit=12, max_gap='2H')
    # ds['usr_cor'] = ds['usr_cor'].interpolate_na(dim='time', method='linear', limit=12, max_gap='2H')
    
    # from IPython import embed; embed()
    # print,'- Sun in view of upper sensor / workable albedos:',n_elements(OKalbedos),100*n_elements(OKalbedos)/n_elements(ds['dsr']),'%'
    valid = (~(ds['dsr_cor'].isnull())).sum()
    # print('- Sun in view of upper sensor / workable albedos:',
    #       OKalbedos.sum().values,
    #       (100*OKalbedos.sum()/valid).round().values,
    #       "%")
    
    # print('- Sun below horizon:',
    #       sundown.sum(),
    #       (100*sundown.sum()/valid).round().values,
    #       "%")
    
    # print('- Sun in view of lower sensor:',
    #       sunonlowerdome.sum().values,
    #       (100*sunonlowerdome.sum()/valid).round().values,
    #       "%")
    
    # print('- Spikes removed using TOA criteria:',
    #       TOA_crit_nopass.sum().values,
    #       (100*TOA_crit_nopass.sum()/valid).round().values,
    #       "%")
    
    # print('- Mean net SR change by corrections:',
    #       (ds['dsr_cor']-ds['usr_cor']-ds['dsr']+ds['usr']).sum().values/valid.values,
    #       "W/m2")
    
    
    # ds['wspd_x'] = ds['wspd'] * np.sin(ds['wdir'] * deg2rad)
    # ds['wspd_y'] = ds['wspd'] * np.cos(ds['wdir'] * deg2rad)
    
    # adjust properties
    # https://github.com/GEUS-PROMICE/PROMICE-AWS-processing/issues/23
    # Just adding special treatment here in service of replication. rh_cor is clipped not NaN'd
    # https://github.com/GEUS-PROMICE/PROMICE-AWS-processing/issues/20
    df = pd.read_csv("./variables.csv", index_col=0, comment="#", usecols=('field','lo','hi','OOL'))
    df = df.dropna(how='all')
    for var in df.index:
        if var not in list(ds.variables): continue
        if var == 'rh_cor':
             ds[var] = ds[var].where(ds[var] >= df.loc[var, 'lo'], other = 0)
             ds[var] = ds[var].where(ds[var] <= df.loc[var, 'hi'], other = 100)
        else:
            ds[var] = ds[var].where(ds[var] >= df.loc[var, 'lo'])
            ds[var] = ds[var].where(ds[var] <= df.loc[var, 'hi'])
        other_vars = df.loc[var]['OOL'] # either NaN or "foo" or "foo bar baz ..."
        if isinstance(other_vars, str): 
            for o in other_vars.split():
                if o not in list(ds.variables): continue
                ds[o] = ds[o].where(ds[var] >= df.loc[var, 'lo'])
                ds[o] = ds[o].where(ds[var] <= df.loc[var, 'hi'])

    return ds
