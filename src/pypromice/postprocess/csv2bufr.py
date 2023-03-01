#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing functions for AWS station data, such as converting PROMICE and GC-Net data files to WMO-compliant BUFR files
"""
import pandas as pd
import sys, traceback
from datetime import datetime, timedelta
from eccodes import codes_set, codes_write, codes_release, \
                    codes_bufr_new_from_samples, CodesInternalError, \
                    codes_is_defined
import math
import numpy as np
from sklearn.linear_model import LinearRegression

from pypromice.postprocess.wmo_config import ibufr_settings, stid_to_skip, vars_to_skip

# from IPython import embed

# To suppress pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None # default='warn'

#------------------------------------------------------------------------------

def getBUFR(s1, outBUFR, stid, land_stids):
    '''Construct and export .bufr messages to file from Series or DataFrame.
    PRIMARY DRIVER FUNCTION

    Parameters
    ----------
    s1 : pandas.Series
        Pandas series of single most recent obset for a station
    outBUFR : str
        File path that .bufr file will be exported to
    stid : str
        The station ID to be processed. e.g. 'KPC_U'
    land_stids : list
        List of station IDs for land-based stations
    '''
    # Open bufr file
    fout = open(outBUFR, 'wb')

    # for i1, r1 in df1.iterrows(): # If dataframe passed w/ multiple rows

    #Create new bufr message to write to
    ibufr = codes_bufr_new_from_samples('BUFR4')
    timestamp = datetime.strptime(s1['time'], '%Y-%m-%d %H:%M:%S')
    config_key = 'mobile'
    if stid in land_stids:
        config_key = 'land'
    try:
        setTemplate(ibufr, timestamp, stid, config_key)
        setStation(ibufr, stid, config_key)
        setAWSvariables(ibufr, s1, timestamp, stid)

        #Encode keys in data section
        codes_set(ibufr, 'pack', 1)

        #Write bufr message to bufr file
        codes_write(ibufr, fout)

    except CodesInternalError as ec:
        print(traceback.format_exc())
        print(ec)
        sys.exit('-----> CodesInternalError in getBUFR!')
    except Exception as e:
        # Catch anything else here...
        print(traceback.format_exc())
        print(e)
        sys.exit('!!!!!!!!!! ERROR in getBUFR')

    codes_release(ibufr)

    fout.close()


def setTemplate(ibufr, timestamp, stid, config_key):
    '''Set bufr message template.

    Parameters
    ----------
    ibufr : bufr.msg
        Bufr message object
    timestamp : datetime.Datetime
        Timestamp of observation
    stid : str
        The station ID to be processed. e.g. 'KPC_U'
    config_key : str
        Defines which config dict to use in wmo_config.ibufr_settings, 'mobile' or 'land'
    '''
    for k, v in ibufr_settings[config_key]['template'].items():
        if codes_is_defined(ibufr, k) == 1:
            codes_set(ibufr, k, v)
        else:
            print('-----> setTemplate Key not defined: {}'.format(k))
            continue

    codes_set(ibufr, 'typicalYear', timestamp.year)
    codes_set(ibufr, 'typicalMonth', timestamp.month)
    codes_set(ibufr, 'typicalDay', timestamp.day)
    codes_set(ibufr, 'typicalHour', timestamp.hour)
    codes_set(ibufr, 'typicalMinute', timestamp.minute)
    # codes_set(ibufr, 'typicalSecond', timestamp.second)


def setStation(ibufr, stid, config_key):
    '''Set station-specific info to bufr message.

    Parameters
    ----------
    ibufr : bufr.msg
        Bufr message object
    stid : str
        The station ID to be processed. e.g. 'KPC_U'
    config_key : str
        Defines which config dict to use in wmo_config.ibufr_settings, 'mobile' or 'land'
    '''
    station_indentifier_keys = ('shipOrMobileLandStationIdentifier','stationNumber')
    for k, v in ibufr_settings[config_key]['station'].items():
        if k in station_indentifier_keys:
            # Deal with any string replacement of stid names before indexing
            if ('v3' in stid) and (stid.replace('v3','') in stid_to_skip['use_v3']):
                # We are reading the v3 station ID file, and the config says to use it!
                # But we need to write to BUFR without v3 name
                stid = stid.replace('v3','')
            if stid == 'THU_U2':
                stid = 'THU_U'
            if stid in ('JAR_O','SWC_O'):
                stid = stid.replace('_O','')
            if stid == 'CEN2':
                stid = 'CEN'
            if stid in v:
                codes_set(ibufr, k, v[stid])
            else:
                sys.exit('!!!!!!!!!! ID not found for {}'.format(stid))
        else:
            if codes_is_defined(ibufr, k) == 1:
                codes_set(ibufr, k, v)
            else:
               print('-----> setStation Key not defined: {}'.format(k))
               continue


def setAWSvariables(ibufr, row, timestamp, stid):
    '''Set AWS measurements to bufr message.
    
    Parameters
    ----------
    ibufr : bufr.msg
        Bufr message object
    row : pandas.DataFrame row, or pandas.Series
        DataFrame row (or Series) with AWS variable data
    timestamp : datetime.datetime
        timestamp for this row
    stid : str
        The station ID to be processed. e.g. 'KPC_U'
    '''
    # Set timestamp fields
    setBUFRvalue(ibufr, 'year', timestamp.year)
    setBUFRvalue(ibufr, 'month', timestamp.month)
    setBUFRvalue(ibufr, 'day', timestamp.day)
    setBUFRvalue(ibufr, 'hour', timestamp.hour)
    setBUFRvalue(ibufr, 'minute', timestamp.minute)

    vars_dict = {
        'relativeHumidity': 'rh_i', # DMI wants non-corrected rh
        'airTemperature': 't_i',
        'pressure': 'p_i',
        'windDirection': 'wdir_i',
        'windSpeed': 'wspd_i'
    }
    for bufr_key, source_var in vars_dict.items():
        if (stid in vars_to_skip) and (source_var in vars_to_skip[stid]):
            print('----> Skipping var: {} {}'.format(stid,source_var))
        else:
            setBUFRvalue(ibufr, bufr_key, row[source_var])

    # Set position metadata
    setBUFRvalue(ibufr, 'latitude', row['gps_lat_fit'])
    setBUFRvalue(ibufr, 'longitude', row['gps_lon_fit'])
    setBUFRvalue(ibufr, 'heightOfStationGroundAboveMeanSeaLevel', row['gps_alt_fit']) # also height and heightOfStation?

    # The ## in the codes_set() indicate the position in the BUFR for the parameter.
    # e.g. #10#timePeriod will assign to the 10th occurence of "timePeriod", which corresponds
    # to the wind speed section. Note that both the "synopMobil" and "synopLand" templates
    # appear to have the same positions for all parameters that are set here.
    # View the output BUFR to see section keys with 'bufr_dump filename.bufr'.
    if math.isnan(row['wspd_i']) is False:
        #Set time significance (2=temporally averaged)
        codes_set(ibufr, '#1#timeSignificance', 2)
        #Set monitoring time period (-10=10 minutes)
        codes_set(ibufr, '#10#timePeriod', -10)

    #Set measurement heights
    if math.isnan(row['z_boom_u_smooth']) is False:
        codes_set(ibufr,
                  '#1#heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform',
                  row['z_boom_u_smooth']-0.1) # For air temp and RH
        codes_set(ibufr,
                  '#7#heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform',
                  row['z_boom_u_smooth']+0.4) # For wind speed
        if math.isnan(row['gps_alt_fit']) is False:
            codes_set(ibufr, 'heightOfBarometerAboveMeanSeaLevel',
                      row['gps_alt_fit']+row['z_boom_u_smooth']) # For pressure


def setBUFRvalue(ibufr, b_name, value):
    '''Set variable in BUFR message
    Called in setAWSvariables() to make sure we aren't passing NaNs

    Parameters
    ----------
    ibufr : bufr.msg                
        Active BUFR message
    b_name : str
        BUFR message variable name
    value : int/float
        Value to be assigned to variable
    '''
    if math.isnan(value) is False:
        try:
            codes_set(ibufr, b_name, value)
        except CodesInternalError as ec:
            print(f'{ec}: {b_name}')
            sys.exit('-----> CodesInternalError in setBUFRvalue!')
    else:
        print('----> {} {}'.format(b_name, value))


def linear_fit(df, column, decimals, stid):
    '''Apply a linear regression to the input column

    Linear regression is following:
    https://realpython.com/linear-regression-in-python/#simple-linear-regression-with-scikit-learn

    Parameters
    ----------
    df : pandas.Dataframe
        datetime-indexed df, limited to desired time length for linear fit
    column : str
        The target column for applying linear fit
    decimals : int
        How many decimals to round the output fit values
    stid : str
        The station ID to be processed. e.g. 'KPC_U'

    Returns
    -------
    df : pandas.Dataframe
        The original input df, with added column for the linear regression values
    '''
    if column in df:
        df_dropna = df[df[column].notna()] # limit to only non-nan for the target column
        if len(df_dropna[column]) > 0:
            # Get datetime x values into epoch sec integers
            x_epoch = df_dropna.index.values.astype(np.int64) // 10 ** 9
            x = x_epoch.reshape(-1,1)
            y = df_dropna[column].values # can also reshape this, but not necessary
            model = LinearRegression().fit(x, y)
            y_pred = model.predict(x).round(decimals=decimals)

            # Plot data if desired
            # if stid == 'LYN_T':
            #     if (column == 'gps_lat') or (column == 'gps_lon') or (column == 'gps_alt'):
            #         import matplotlib.pyplot as plt
            #         plt.scatter(x,y)
            #         plt.plot(x,y_pred, color='red')
            #         plt.title('{} {}'.format(stid, column))
            #         plt.show()

            # Add y_pred back to original df
            df_dropna['y_pred'] = y_pred
            df['{}_fit'.format(column)] = df_dropna['y_pred']
        else:
            # All data is NaN! Just write NaNs using the original column
            print('----> No {} data for {}!'.format(column, stid))
            df['{}_fit'.format(column)] = df[column]
    else:
        print('----> {} not found in dataframe!'.format(column))
        pass

    return df


def rolling_window(df, column, window, min_periods, decimals):
    '''Apply a rolling window (smoothing) to the input column

    Parameters
    ----------
    df : pandas.Dataframe
        datetime-indexed df
    column : str
        The target column for applying rolling window
    window : str
        Window size (e.g. '24H' or 30D')
    min_periods : int
        Minimum number of observations in window required to have a value;
        otherwise, result is np.nan.
    decimals : int
        How many decimal places to round the output smoothed values

    Returns
    -------
    df : pandas.Dataframe
        The original input df, with added column for the smoothed values
    '''
    df['{}_smooth'.format(column)] = df[column].rolling(
        window,
        min_periods=min_periods,
        center=True, # set the window labels as the center of the window
        closed='both' # no points in the window are excluded (first or last)
        ).median().round(decimals=decimals) # could also round to whole meters (decimals=0)
    return df

def round_values(s):
    '''Enforce precision
    Note the sensor accuracies listed here:
    https://essd.copernicus.org/articles/13/3819/2021/#section8
    In addition to sensor accuracy, WMO requires pressure and heights
    to be reported at 0.1 precision.
    
    Parameters
    ----------
    s : pandas series (could also be a dataframe)

    Returns
    -------
    s : modified pandas series (could also be a dataframe)
    '''
    s['rh_i'] = s['rh_i'].round(decimals=0)
    s['wspd_i'] = s['wspd_i'].round(decimals=1)
    s['wdir_i'] = s['wdir_i'].round(decimals=0)
    s['t_i'] = s['t_i'].round(decimals=1)
    s['p_i'] = s['p_i'].round(decimals=1)

    # gps_lat,gps_lon,gps_alt,z_boom_u are all rounded in linear_fit() or rolling_window()
    return s

def write_positions(s, stid, positions):
    '''Set valid lat, lon, alt to the positions dict.
    For submitting registration metadata to DMI/WMO, and for writing
    positions to AWS_station_locations.csv. Must pass --positions arg.

    Parameters
    ----------
    s : pandas series
        The current obset we are working with (for BUFR submission)
    stid : str
        The station ID, such as NUK_L
    positions : dict
        Dict storing current station positions.

    Returns
    -------
    positions : dict
        Modified dict storing current station positions.
    '''
    print('writing positions for {}'.format(stid))
    to_write = ['lat','lon','alt']
    for i in to_write:
        if (f'gps_{i}_fit' in s) and (pd.isna(s[f'gps_{i}_fit']) is False):
            positions[stid][i] = s[f'gps_{i}_fit']

    # Add timestamp
    positions[stid]['timestamp'] = s['time']
    return positions

def fetch_old_positions(df, stid, time_limit, positions):
    '''Set valid lat, lon, alt to the positions dict.
    For submitting registration metadata to DMI/WMO, and for writing
    positions to AWS_station_locations.csv. We run this if a station
    is skipped for BUFR processing or does not have new or recent-enough
    obs, but we still want to find the last position if we have it.
    Must pass --positions arg. The last position is using the previous
    3 months best fit from the last transmission we have, could be many
    months or even years ago.

    Parameters
    ----------
    df : pandas dataframe
        The full tx dataframe
    stid : str
        The station ID, such as NUK_L
    time_limit : str
        Previous time to limit dataframe before applying linear regression.
        (e.g. '3M')
    positions : dict
        Dict storing current station positions.

    Returns
    -------
    positions : dict
        Modified dict storing most-recent station positions.
    '''
    # Combine gps and msg lat and lon using combine_first()
    # If any GPS positions are missing, we will fill the missing GPS positions with modem
    # positions (if they are present). Important to do this first, and then apply linear fit
    # to the resulting single array. Otherwise, we can have jumps when the GPS data goes out
    # or comes back. The message coordinates can sometimes all be 0.0, so we check for this.
    print('fetching old positions for {}'.format(stid))
    df_limited = df.last(time_limit)
    print('last transmission: {}'.format(df_limited.index.max()))

    if ('msg_lat' in df_limited) and ('msg_lon' in df_limited):
        if (0.0 not in df_limited['msg_lat'].values):
            df_limited['gps_lat'] = df_limited['gps_lat'].combine_first(df_limited['msg_lat'])
        if (0.0 not in df_limited['msg_lon'].values):
            df_limited['gps_lon'] = df_limited['gps_lon'].combine_first(df_limited['msg_lon'])

    df_limited = linear_fit(df_limited, 'gps_alt', 1, stid)
    df_limited = linear_fit(df_limited, 'gps_lat', 6, stid)
    df_limited = linear_fit(df_limited, 'gps_lon', 6, stid)

    # s = df_limited.loc[df_limited.index.max()] # just use max index

    # Go through gps_lat_fit, gps_lon_fit and gps_alt_fit and keep the most recent
    # valid index. They should all be the same. Or, if modem-derived, we will have
    # only lat and lon (with same index). But this treatment covers all possible
    # scenarios of missing data.
    pos_strings = ['lat','lon','alt']
    pos_timestamps = []
    valid_timestamp_found = False
    recent_timestamp = pd.to_datetime('1900-01-01') # initialize with an old date
    for p in pos_strings:
        p_timestamp = df_limited[f'gps_{p}_fit'].last_valid_index()
        if (p_timestamp is not None) and (p_timestamp > recent_timestamp):
            recent_timestamp = p_timestamp
            valid_timestamp_found = True

    if valid_timestamp_found:
        s = df_limited.loc[recent_timestamp]

        to_write = ['lat','lon','alt']
        for p in pos_strings:
            if (f'gps_{p}_fit' in s) and (pd.isna(s[f'gps_{p}_fit']) is False):
                positions[stid][p] = s[f'gps_{p}_fit']
                # positions[stid][f'{p}_source'] = 'OLD' #source flag

        # Add timestamp
        positions[stid]['timestamp'] = s['time']
    return positions


def min_data_check(s, stid):
    '''Check that we have minimum required fields to proceed with writing to BUFR
    For wx vars, we currently require both air temp and pressure to be non-NaN.
    If you know a specific var is reporting bad data, you can ignore just that var
    using the vars_to_skip dict in wmo_config.

    Parameters
    ----------
    s : pandas series
        The current obset we are working with (for BUFR submission)
    stid : str
        The station ID, such as NUK_L

    Returns
    -------
    min_data_wx_result : bool
        True (default), the test for min wx data passed. False, the test failed.
    min_data_pos_result : bool
        True (default), the test for min position data passed. False, the test failed.
    '''
    min_data_wx_result = True
    min_data_pos_result = True

    # Can use pd.isna() or math.isnan() below...

    # Always require valid air temp and valid pressure (both must be non-nan)
    # if (pd.isna(s['t_i']) is False) and (pd.isna(s['p_i']) is False):
    #     pass
    # else:
    #     print('----> Failed min_data_check for air temp and pressure!')
    #     min_data_wx_result = False

    # If both air temp and pressure are nan, do not submit.
    # This will allow the case of having only one or the other.
    if (pd.isna(s['t_i']) is True) and (pd.isna(s['p_i']) is True):
        print('----> Failed min_data_check for air temp and pressure!')
        min_data_wx_result = False

    # Must have a valid position
    # Note that gps_ variables have already had replacement with msg_ positions if needed

    # Missing just elevation OK
    # if (pd.isna(s['gps_lat_fit']) is False) and (pd.isna(s['gps_lon_fit']) is False):
    #     pass
    # Require all three: lat, lon, elev
    if ((pd.isna(s['gps_lat_fit']) is False) and
        (pd.isna(s['gps_lon_fit']) is False) and
        (pd.isna(s['gps_alt_fit']) is False)):
        pass
    else:
        print('----> Failed min_data_check for position!')
        min_data_pos_result = False

    return min_data_wx_result, min_data_pos_result
