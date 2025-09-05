
import xarray as xr

# Define air temperature for radiometer adjustments
T_0=273.15

def convert_sr(sr: xr.DataArray,
               sr_eng_coef: float) -> xr.DataArray:
    """Convert shortwave radiation measurements from engineering to
    physical units, using a calibration coefficient (defined by
    manufacturers usually)

    Parameters
    ----------
    sr : xr.DataArray
        Shortwave radiation (upwelling or downwelling) measurements
    sr_eng_coef : float
        Shortwave engineering calibration coefficient

    Returns
    -------
    xr.DataArray
        Converted shortwave measurements
    """
    return (sr * 10) / sr_eng_coef

def convert_lr(lr: xr.DataArray,
               t_rad: xr.DataArray,
               lr_eng_coef: float) -> xr.DataArray:
    """Convert longwave radiation measurements from engineering to
    physical units, using the reported radiometer temperature and
    a calibration coefficient (defined by manufacturers usually)

    Parameters
    ----------
    lr : xr.DataArray
        Longwave radiation (upwelling or downwelling) measurements
    t_rad : xr.DataArray
        Radiometer temperature
    lr_eng_coef : float
        Longwave engineering calibration coefficient
    T_0 : float

    Returns
    -------
    xr.DataArray
        Converted shortwave measurements
    """
    return ((lr * 10) / lr_eng_coef) + 5.67e-8 * (t_rad + T_0) **4

def filter_lr(lr: xr.DataArray,
           t_rad: xr.DataArray) -> xr.DataArray:
    """Remove longwave radiation measurements that are missing
    simultaneous radiometer temperature measurements

    Parameters
    ----------
    lr : xr.DataArray
        Longwave radiation measurements (upwelling or downwelling)
    t_rad : xr.DataArray
        Radiometer temperature

    Returns
    -------
    xr.DataArray
        Filtered radiation measurements
    """
    return lr.where(t_rad.notnull())