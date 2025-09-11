__all__ = ["decode_and_convert",  "filter", "piecewise_smoothing_and_interpolation",
           "decode", "convert_from_degrees_and_decimal_minutes",
           "convert_from_decimal_minutes"]

import xarray as xr
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import logging
logger = logging.getLogger(__name__)

def piecewise_smoothing_and_interpolation(data_series, breaks):
    '''Smoothes, inter- or extrapolate the GPS observations. The processing is
    done piecewise so that each period between station relocations are done
    separately (no smoothing of the jump due to relocation). Piecewise linear
    regression is then used to smooth the available observations. Then this
    smoothed curve is interpolated linearly over internal gaps. Eventually, this
    interpolated curve is extrapolated linearly for timestamps before the first
    valid measurement and after the last valid measurement.

    Parameters
    ----------
    data_series : pd.Series
        Series of observed latitude, longitude or elevation with datetime index.
    breaks: list
        List of timestamps of station relocation. First and last item should be
        None so that they can be used in slice(breaks[i], breaks[i+1])

    Returns
    -------
    np.ndarray
        Smoothed and interpolated values corresponding to the input series.
    '''
    df_all = pd.Series(dtype=float)  # Initialize an empty Series to gather all smoothed pieces
    breaks = [None] + breaks + [None]
    _inferred_series = []
    for i in range(len(breaks) - 1):
        df = data_series.loc[slice(breaks[i], breaks[i+1])]

        # Drop NaN values and calculate the number of segments based on valid data
        df_valid = df.dropna()
        if df_valid.shape[0] > 2:
            # Fit linear regression model to the valid data range
            x = pd.to_numeric(df_valid.index).values.reshape(-1, 1)
            y = df_valid.values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(x, y)

            # Predict using the model for the entire segment range
            x_pred = pd.to_numeric(df.index).values.reshape(-1, 1)

            y_pred = model.predict(x_pred)
            df =  pd.Series(y_pred.flatten(), index=df.index)
        # adds to list the predicted values for the current segment
        _inferred_series.append(df)

    df_all = pd.concat(_inferred_series)

    # Fill internal gaps with linear interpolation
    df_all = df_all.interpolate(method='linear', limit_area='inside')

    # Remove duplicate indices and return values as numpy array
    df_all = df_all[~df_all.index.duplicated(keep='last')]
    return df_all.values

def decode_and_convert(gps_lat: xr.DataArray,
                       gps_lon: xr.DataArray,
                       gps_time: xr.DataArray,
                       latitude: float,
                       longitude: float
) -> tuple[xr.DataArray,xr.DataArray,xr.DataArray]:
    """Decode and convert GPS latitude, longtitude and time values."flag_decimal_minutes",
           "flag_for_decoding"
    Decoding is performed if values are detected as string types.
    Conversion consists of transforming to decimal degrees (DD),
    from either decimal minutes (mm.mmmmm) or degrees and
    decimal minutes (ddmm.mmmm)

    Parameters
    ----------
    gps_lat : `xr.DataArray`
        GPS latitude
    gps_lon : `xr.DataArray`
        GPS longitude
    gps_time : `xr.DataArray`
        GPS time

    Returns
    -------
    gps_lat : `xr.DataArray`
        Decoded and converted GPS latitude
    gps_lon : `xr.DataArray`
        Decoded and converted GPS longitude
    gps_time : `xr.DataArray`
        Decoded and converted GPS time
    """
    # Retain GPS array attributes
    lat_attrs = gps_lat.attrs
    lon_attrs = gps_lon.attrs
    time_attrs = gps_time.attrs

    # Decode GPS information if array is an object array
    if gps_lat.dtype.kind == "O":
        lat, lon, time = decode(gps_lat, gps_lon, gps_time)
        if lat is None:
            logger.warning("GPS decoding failed, skipping this routine.")
        else:
            gps_lat, gps_lon, gps_time = lat, lon, time

    # Reformat values to numeric
    gps_lat.values = pd.to_numeric(gps_lat, errors='coerce')
    gps_lon.values = pd.to_numeric(gps_lon, errors='coerce')
    gps_time.values = pd.to_numeric(gps_time, errors='coerce')

    # Convert GPS positions to decimal degrees
    if np.any((gps_lat <= 90) & (gps_lat > 0)):
        gps_lat = convert_from_decimal_minutes(gps_lat, latitude)
        gps_lon = convert_from_decimal_minutes(gps_lon, longitude)
    else:
        gps_lat = convert_from_degrees_and_decimal_minutes(gps_lat)
        gps_lon = convert_from_degrees_and_decimal_minutes(gps_lon)

    # Reassign GPS array attributes
    gps_lat.attrs = lat_attrs
    gps_lon.attrs = lon_attrs
    gps_time.attrs = time_attrs

    return gps_lat, gps_lon, gps_time


def filter(gps_lat: xr.DataArray,
           gps_lon: xr.DataArray,
           gps_alt: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """ Filter GPS latitude, longitude and altitude based on the difference
    to a baseline elevation. The baseline elevation is a gap-filled monthly
    median elevation based on the inputted GPS altitude.

    Parameters
    ----------
    gps_lat : xr.DataArray
        GPS latitude
    gps_lon : xr.DataArray
        GPS longitude
    gps_alt : xr.DataArray
        GPS altitude values with a time dimension

    Returns
    ----------
    gps_lat_filtered : xr.DataArray
        Filtered latitude values
    gps_lon_filtered : xr.DataArray
        Filtered longitude values
    gps_alt_filtered : xr.DataArray
        Filtered altitude values
    """
    # Get altitude monthly median (at month start)
    # This will serve as baseline elevations for filtering
    ser = gps_alt.to_series()
    monthly_median = ser.resample("MS").median()
    baseline_elevation = (
        monthly_median
        .reindex(ser.index, method="nearest")
        .ffill()
        .bfill()
    )

    # Produce conditional mask
    mask = (np.abs(gps_alt - baseline_elevation) < 100) | gps_alt.isnull()

    # Apply mask
    gps_lat_filtered = gps_lat.where(mask)
    gps_lon_filtered = gps_lon.where(mask)
    gps_alt_filtered = gps_alt.where(mask)

    return gps_lat_filtered, gps_lon_filtered, gps_alt_filtered


def convert_from_degrees_and_decimal_minutes(gps):
    """Convert positions (i.e. latitude, longitude) from degrees
    and decimal minutes (ddmm.mmmm) to decimal degree values (DD)"""
    return np.floor(gps / 100) + (gps / 100 - np.floor(gps / 100)) * 100 / 60


def convert_from_decimal_minutes(gps: xr.DataArray, pos: float
) -> xr.DataArray:
    """Convert decimal minutes (mm.mmmmm) to decimal degree
    values (DD), using a predefined position to append values to.
    Needed in the case of PROMICE v1 stations, where logger
    programs saved positions only in decimal minutes."""
    new_gps = np.sign(pos) * (gps + 100 * np.floor(np.abs(pos)))
    return convert_from_degrees_and_decimal_minutes(new_gps)


def decode(gps_lat: xr.DataArray,
           gps_lon: xr.DataArray,
           gps_time: xr.DataArray
) -> tuple[xr.DataArray,xr.DataArray,xr.DataArray]:
    """Decode GPS information. This should be applied if gps information
    consists of strings and not float values. GPS information is returned in
    decimal degrees (ddmm.mmmm) format.

    Parameters
    ----------
    gps_lat : `xr.DataArray`
        GPS latitude
    gps_lon : `xr.DataArray`
        GPS longitude
    gps_time : `xr.DataArray`
        GPS time

    Returns
    -------
    new_lat : `xr.DataArray`
        Decoded GPS latitude
    new_lon : `xr.DataArray`
        Decoded GPS longitude
    new_time : `xr.DataArray`
        Decoded GPS time
    """
    # Pick the first non-null sample safely and detect decoding format
    non_null = gps_lat.dropna(dim='time').values
    sample_value = str(non_null[0])

    try:
        # Object decoding
        if "NH" in sample_value:
            new_lat = gps_object_decoder(gps_lat)
            new_lon = gps_object_decoder(gps_lon)
            new_time = gps_object_decoder(gps_time)
            return new_lat, new_lon, new_time

        # L-string decoding
        elif "L" in sample_value:
            logger.info("Found 'L' in GPS string; applying decode + scaling.")
            new_lat = gps_l_string_decoder(gps_lat)
            new_lon = gps_l_string_decoder(gps_lon)
            new_time = gps_object_decoder(gps_time)
            return new_lat, new_lon, new_time

        # Unknown format, attempt to decode
        else:
            logger.info("Unknown GPS string format; attempting generic decode.")
            new_lat = gps_object_decoder(gps_lat)
            new_lon = gps_object_decoder(gps_lon)
            new_time = gps_object_decoder(gps_time)
            return new_lat, new_lon, new_time

    except Exception as e:
        logger.error(f"Failed to decode GPS data: {e!r} "
                     f"(dtype={gps_lat.dtype})")
        return None, None, None


def gps_object_decoder(gps : xr.DataArray) -> xr.DataArray:
    """GPS decoder for object array formatting. For example, PROMICE v2
    stations should send information as 'NH6429.01544,WH04932.86061'
    original formatting (NUK_L 2022); PROMICE v3 stations should send
    coordinates as '6430,4916' (NUK_Uv3); and GC-Net stations should
    send coordinates as '6628.93936',04617.59187' (DY2)"""
    str2nums = [re.findall(r"[-+]?\d*\.\d+|\d+", _) if isinstance(_, str) else [np.nan] for _ in gps.values]
    gps[:] = pd.DataFrame(str2nums).astype(float).T.values[0]
    gps = gps.astype(float)
    return gps


def gps_l_string_decoder(gps : xr.DataArray) -> xr.DataArray:
    """GPS L-string decoder"""
    # Convert from object array
    gps = gps_object_decoder(gps)

    # Convert from integer-like values to degrees
    gps = gps/100000
    return gps
