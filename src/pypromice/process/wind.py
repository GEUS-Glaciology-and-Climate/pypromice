import numpy as np

def filter_wind_direction(wdir, wspd):
    return wdir.where(wspd != 0)


def calculate_directional_wind_speed(wspd, wdir, deg2rad=np.pi/180):
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