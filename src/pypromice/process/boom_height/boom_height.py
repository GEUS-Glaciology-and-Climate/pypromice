import pandas as pd

def correct(z_boom, t, T_0):
    '''Adjust sonic ranger readings for sensitivity to air temperature'''
    return z_boom * ((t + T_0) / T_0) ** 0.5


def correct_with_temp_interp(z_boom, t, vars_df, T_0):
    '''Adjust sonic ranger readings for sensitivity to air temperature'''
    t_interp = _interp_air_temperature(t, vars_df)                          # TODO retire vars_df
    return z_boom * ((t_interp + T_0) / T_0) ** 0.5


def _interp_air_temperature(t, var_configurations, max_interp=pd.Timedelta(12, 'h')):      # Move to air temperature
    '''Clip and interpolate temperature dataset for use in corrections

    Parameters
    ----------
    t : `xarray.DataArray`
        Array of temperature data
    vars_df : `pandas.DataFrame`
        Dataframe to retrieve attribute hi-lo values from for temperature clipping
    max_interp : `pandas.Timedelta`
        Maximum time steps to interpolate across. The default is 12 hours.

    Returns
    -------
    t_interp : `xarray.DataArray`
        Array of interpolatedtemperature data
    '''
    # Determine if upper or lower temperature array
    var = t.name.lower()

    # Find range threshold and use it to clip measurements
    cols = ["lo", "hi", "OOL"]                                      # TODO lo hi values should be explicitly defined here
    assert set(cols) <= set(var_configurations.columns)
    variable_limits = var_configurations[cols].dropna(how="all")
    temp = t.where(t >= variable_limits.loc[var, 'lo'])
    temp = t.where(t <= variable_limits.loc[var, 'hi'])

    # Drop duplicates and interpolate across NaN values
    #    temp_interp = temp.drop_duplicates(dim='time', keep='first')
    t_interp = t.interpolate_na(dim='time', max_gap=max_interp)

    return t_interp


