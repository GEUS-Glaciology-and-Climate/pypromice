def convert_sr(sr, sr_eng_coef):
    '''Convert shortwave radiation (upwelling or downwelling) from engineering to physical units'''

    # Can be used to correct both dsr and usr:
    #     ds['dsr'] = (ds['dsr'] * 10) / ds.attrs['dsr_eng_coef']
    #     ds['usr'] = (ds['usr'] * 10) / ds.attrs['usr_eng_coef']

    return (sr * 10) / sr_eng_coef

def convert_lr(lr, t_rad, lr_eng_coef, T_0):
    '''Convert longwave radiation (upwelling or downwelling) from engineering to physical units

    Parameters
    ----------
    T_0 : int
        Air temperature for sonic ranger adjustment. The default is 273.15.
    '''

    # Can be used to correct both dlr and ulr:
    # ds['dlr'] = ((ds['dlr'] * 10) / ds.attrs['dlr_eng_coef']) + 5.67E-8 * (ds['t_rad'] + T_0) ** 4
    # ds['ulr'] = ((ds['ulr'] * 10) / ds.attrs['ulr_eng_coef']) + 5.67E-8 * (ds['t_rad'] + T_0) ** 4

    return ((lr * 10) / lr_eng_coef) + 5.67E-8 * (t_rad + T_0) ** 4

