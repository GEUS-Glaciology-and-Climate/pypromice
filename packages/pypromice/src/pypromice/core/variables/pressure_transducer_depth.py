__all__ = ["correct_and_calculate_depth", "apply_offset"]

import xarray as xr
import numpy as np
import logging
logger = logging.getLogger(__name__)

def correct_and_calculate_depth(z_pt: xr.DataArray,
                                air_pressure: xr.DataArray,
                                pt_antifreeze: float,
                                pt_z_factor: float,
                                pt_z_coef: float,
                                pt_z_p_coef: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """Adjust pressure depth and calculate pressure transducer depth based on
    pressure transducer fluid density

    Parameters
    ----------
    z_pt : xr.DataArray
        Pressure transducer height (corrected for offset)
    air_pressure : xr.DataArray
        Air pressure
    pt_antifreeze : float
        Pressure transducer anti-freeze percentage for fluid density
        correction
    pt_z_factor : float
        Pressure transducer factor
    pt_z_coef : float
        Pressure transducer coefficient
    pt_z_p_coef : float
        Pressure transducer coefficient

    Returns
    -------
    z_pt_cor : xr.DataArray
        Pressure transducer height corrected
    z_pt : xr.DataArray
        Pressure transducer depth
    """
    # Calculate pressure transducer fluid density
    # TODO: Implement function w/ reference (analytical or from LUT)
    # TODO: Track uncertainty
    if pt_antifreeze == 50:
        rho_af = 1092
    elif pt_antifreeze == 100:
        rho_af = 1145
    else:
        rho_af = np.nan
        logger.info('ERROR: Incorrect metadata: "pt_antifreeze" = ' +
              f'{pt_antifreeze}. Antifreeze mix only supported at 50% or 100%')
        # assert(False)

    # Correct pressure depth
    z_pt_cor = z_pt * pt_z_coef * pt_z_factor * 998.0 / rho_af + 100 * (pt_z_p_coef - air_pressure) / (rho_af * 9.81)

    # Calculate pressure transducer depth
    z_pt = z_pt * pt_z_coef * pt_z_factor * 998.0 / rho_af

    return z_pt_cor, z_pt

def apply_offset(z_pt: xr.DataArray,
                 z_pt_offset: int
) -> xr.DataArray:
    """Apply defined offset to pressure transducer height

    Parameters
    ----------
    z_pt : xr.DataArray
        Pressure transducer height
    z_pt_offset : xr.DataArray
        Transducer height offset

    Returns
    -------
    xr.DataArray
        Adjusted pressure transducer height
    """
    return z_pt + z_pt_offset