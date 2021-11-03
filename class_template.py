"""
Created on Wed Nov  3 11:55:16 2021

@author: Penelope How (pho@geus.dk)
"""


class StationLogger(object):
    '''  
    '''
    def __init__(self, fan_dc, batt_v_ss, batt_v):
        self.fan_current = fan_dc
        self.battery_voltage_at_sample_start = batt_v_ss
        self.battery_voltage = batt_v
        
        
class StationTime(object):
    '''
    '''
    def __init__(self, time, rec, min_y):
        self.time = time
        self.record = rec
        self.minutes = min_y


class GPS(object):
    '''
    '''
    def __init__(self, gps_time, gps_lat, gps_lon, gps_alt, gps_giodal, 
                 gps_geounit, gps_q, gps_numsat, gps_hdop):
        self.gps_time = gps_time
        self.gps_latitude = gps_lat
        self.gps_longitude = gps_lon
        self.gps_altitude = gps_alt
        self.gps_geoid_separation = gps_giodal
        self.gps_geounit = gps_geounit
        self.gps_q = gps_q
        self.gps_numsat = gps_numsat
        self.gps_hdop = gps_hdop


class StationPosition(GPS):
    '''
    '''
    def __init__(self, gps, tilt_x, tilt_y, rot):
        
        gps_time, gps_lat, gps_lon, gps_alt, gps_giodal, gps_geounit, gps_q, gps_numsat, gps_hdop = gps
        
        GPS.__init__(self, gps_time, gps_lat, gps_lon, gps_alt, gps_giodal, 
                     gps_geounit, gps_q, gps_numsat, gps_hdop)
        
        self.platform_view_angle_x = tilt_x
        self.platform_view_angle_y = tilt_y
        self.platform_azimuth_angle = rot
        

class AirPressure(object):
    '''
    '''
    def __init__(self, p):
        self.air_pressure = p


class AirTemperature(object):
    '''
    '''
    def __init__(self, t, t_hydroclip):
        self.air_temperature_at_boom = t
        self.air_temperature_at_hygroclip = t_hydroclip
       
        
class RelHumidity(object):
    '''
    '''
    def __init__(self, rh):
        self.relative_humidity = rh


class Wind(object):
    '''
    '''
    def __init__(self, wspd, wdir, wd_std):
        self.wind_speed = wspd
        self.wind_from_direction = wdir
        self.wind_from_direction_standard_deviation = wd_std


class Radiation(object):
    '''
    '''
    def __init__(self, dswr, uswr, dlwr, ulwr, t_rad, dshf, dlhf, albedo, cc):
        self.surface_downwelling_shortwave_flux = dswr
        self.surface_upwelling_shortwave_flux = uswr
        
        self.surface_downwelling_longwave_flux = dlwr
        self.surface_upwelling_longwave_flux = ulwr
        
        self.temperature_at_radiation_sensor = t_rad

        self.surface_downward_sensible_heat_flux = dshf
        self.surface_downward_latent_heat_flux = dlhf
        
        self.surface_albedo = albedo
        self.cloud_area_fraction = cc



class Height(object): 
    '''
    '''  
    def __init__(self, z_boom, z_boom_q, z_stake, z_stake_q, z_ice):
        self.snow_height_at_boom = z_boom
        self.snow_height_at_boom_quality = z_boom_q
        self.snow_height_at_stake = z_stake
        self.snow_height_at_stake_quality = z_stake_q
        self.ice_height = z_ice


class IceTemperature(object):
    '''
    '''
    def __init__(self, t_i_1, t_i_2, t_i_3, t_i_4, t_i_5, t_i_6, t_i_7, t_i_10):
        self.ice_temperature_at_1m = t_i_1
        self.ice_temperature_at_2m = t_i_2
        self.ice_temperature_at_3m = t_i_3
        self.ice_temperature_at_4m = t_i_4
        self.ice_temperature_at_5m = t_i_5
        self.ice_temperature_at_6m = t_i_6
        self.ice_temperature_at_7m = t_i_7
        self.ice_temperature_at_10m = t_i_10


class AWS(StationLogger, StationTime, StationPosition, AirPressure, 
          AirTemperature, RelHumidity, Wind, Radiation, Height, IceTemperature):
    '''
    '''
    def __init__(self, name, fan_dc, batt_v_ss, batt_v, time, rec, min_y,
                 gps, tilt_x, tilt_y, rot, p, t, t_hydroclip, rh, wspd, wdir, 
                 wd_std, dswr, uswr, dlwr, ulwr, t_rad, dshf, dlhf, albedo, cc,
                 z_boom, z_boom_q, z_stake, z_stake_q, z_ice, t_i_1, t_i_2, 
                 t_i_3, t_i_4, t_i_5, t_i_6, t_i_7, t_i_10):
        
        StationLogger.__init__(self, fan_dc, batt_v_ss, batt_v)
        StationTime.__init__(self, time, rec, min_y)
        StationPosition.__init__(self, gps, tilt_x, tilt_y, rot)
        AirPressure.__init__(self, p)
        AirTemperature.__init__(self, t, t_hydroclip) 
        RelHumidity.__init__(self, rh)
        Wind.__init__(self, wspd, wdir, wd_std) 
        
        Radiation.__init__(self, dswr, uswr, dlwr, ulwr, 
                           t_rad, dshf, dlhf, albedo, cc)
        
        Height.__init__(self, z_boom, z_boom_q, z_stake, z_stake_q, z_ice) 
        
        IceTemperature.__init__(self, t_i_1, t_i_2, t_i_3, t_i_4, 
                                t_i_5, t_i_6, t_i_7, t_i_10)
        
        self.station_name = name
