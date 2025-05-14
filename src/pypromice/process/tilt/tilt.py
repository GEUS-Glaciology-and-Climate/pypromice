import pandas as pd
import numpy as np

def smooth(tilt, win_size):
    '''Smooth tilt values using a rolling window. This is translated from the
    previous IDL/GDL smoothing algorithm:
    tiltX = smooth(tiltX,7,/EDGE_MIRROR,MISSING=-999) & tiltY = smooth(tiltY,7,/EDGE_MIRROR, MISSING=-999)
    endif
    In Python, this should be
    dstxy = dstxy.rolling(time=7, win_type='boxcar', center=True).mean()
    But the EDGE_MIRROR makes it a bit more complicated

    Parameters
    ----------
    tilt : xarray.DataArray
        Array (either 'tilt_x' or 'tilt_y'), tilt values (can be in degrees or voltage)
    win_size : int
        Window size to use in pandas 'rolling' method.
        e.g. a value of 7 spans 70 minutes using 10 minute data.

    Returns
    -------
    tdf_rolling : tuple, as: (str, numpy.ndarray)
        The numpy array is the tilt values, smoothed with a rolling mean
    '''
    s = int(win_size / 2)
    tdf = tilt.to_dataframe()
    mirror_start = tdf.iloc[:s][::-1]
    mirror_end = tdf.iloc[-s:][::-1]
    mirrored_tdf = pd.concat([mirror_start, tdf, mirror_end])

    tdf_rolling = (
        ('time'),
        mirrored_tdf.rolling(
            win_size, win_type='boxcar', min_periods=1, center=True
        ).mean()[s:-s].values.flatten()
    )
    return tdf_rolling


def filter(tilt, threshold):                                                    # TODO split this function, one for filtering, one for conversion. PHO.
    '''Filter tilt with given threshold, and convert from voltage to degrees.
    Voltage-to-degrees converseion is based on the equation in 3.2.9 at
    https://essd.copernicus.org/articles/13/3819/2021/#section3

    Parameters
    ----------
    tilt : xarray.DataArray
        Array (either 'tilt_x' or 'tilt_y'), tilt values (voltage)
    threshold : int
        Values below this threshold (-100) will not be retained.

    Returns
    -------
    dst.interpolate_na() : xarray.DataArray
        Array (either 'tilt_x' or 'tilt_y'), tilt values (degrees)
    '''
    # notOKtiltX = where(tiltX lt -100, complement=OKtiltX) & notOKtiltY = where(tiltY lt -100, complement=OKtiltY)
    notOKtilt = (tilt < threshold)
    OKtilt = (tilt >= threshold)
    tilt = tilt / 10

    # IDL version:
    # tiltX = tiltX/10.
    # tiltnonzero = where(tiltX ne 0 and tiltX gt -40 and tiltX lt 40)
    # if n_elements(tiltnonzero) ne 1 then tiltX[tiltnonzero] = tiltX[tiltnonzero]/abs(tiltX[tiltnonzero])*(-0.49*(abs(tiltX[tiltnonzero]))^4 + 3.6*(abs(tiltX[tiltnonzero]))^3 - 10.4*(abs(tiltX[tiltnonzero]))^2 +21.1*(abs(tiltX[tiltnonzero])))
    # tiltY = tiltY/10.
    # tiltnonzero = where(tiltY ne 0 and tiltY gt -40 and tiltY lt 40)
    # if n_elements(tiltnonzero) ne 1 then tiltY[tiltnonzero] = tiltY[tiltnonzero]/abs(tiltY[tiltnonzero])*(-0.49*(abs(tiltY[tiltnonzero]))^4 + 3.6*(abs(tiltY[tiltnonzero]))^3 - 10.4*(abs(tiltY[tiltnonzero]))^2 +21.1*(abs(tiltY[tiltnonzero])))

    dst = tilt
    nz = (dst != 0) & (np.abs(dst) < 40)

    dst = dst.where(~nz, other=dst / np.abs(dst)
                               * (-0.49
                                  * (np.abs(dst)) ** 4 + 3.6
                                  * (np.abs(dst)) ** 3 - 10.4
                                  * (np.abs(dst)) ** 2 + 21.1
                                  * (np.abs(dst))))

    # if n_elements(OKtiltX) gt 1 then tiltX[notOKtiltX] = interpol(tiltX[OKtiltX],OKtiltX,notOKtiltX) ; Interpolate over gaps for radiation correction; set to -999 again below.
    dst = dst.where(~notOKtilt)
    return dst.interpolate_na(dim='time',
                              use_coordinate=False)  # TODO: Filling w/o considering time gaps to re-create IDL/GDL outputs. Should fill with coordinate not False. Also consider 'max_gap' option?

def apply_correction(tilt, correction_factor):
    '''Apply tilt factor (e.g. -1 will invert tilt angle)'''
    return tilt * correction_factor
