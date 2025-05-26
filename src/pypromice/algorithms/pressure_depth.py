def correct(z_pt, p, pt_antifreeze, pt_z_factor, pt_z_coef, pt_z_p_coef):
    '''Adjust pressure depth and calculate pressure transducer depth based on
    pressure transducer fluid density

    Parameters
    ----------
    z_pt : xr.Dataarray
        Pressure transducer height (corrected for offset)
    p : xr.Dataarray
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
    z_pt_cor : xr.Dataarray
        Pressure transducer height corrected
    z_pt : xr.Dataarray
        Pressure transducer depth
    '''
    # Calculate pressure transducer fluid density
    if pt_antifreeze == 50:  # TODO: Implement function w/ reference (analytical or from LUT)
        rho_af = 1092  # TODO: Track uncertainty
    elif pt_antifreeze == 100:
        rho_af = 1145
    else:
        rho_af = np.nan
        logger.info('ERROR: Incorrect metadata: "pt_antifreeze" = ' +
                    f'{pt_antifreeze}. Antifreeze mix only supported at 50% or 100%')
        # assert(False)

    # Correct pressure depth
    z_pt_cor = z_pt * pt_z_coef * pt_z_factor * 998.0 / rho_af + 100 * (pt_z_p_coef - p) / (rho_af * 9.81)

    # Calculate pressure transducer depth
    z_pt = z_pt * pt_z_coef * pt_z_factor * 998.0 / rho_af

    return z_pt_cor, z_pt
