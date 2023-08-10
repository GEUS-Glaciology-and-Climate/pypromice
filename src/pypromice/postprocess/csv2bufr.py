#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing functions for AWS station data, such as converting PROMICE and GC-Net data files to WMO-compliant BUFR files
"""
import pandas as pd
import sys, traceback
import os
from datetime import datetime, timedelta
from eccodes import codes_set, codes_write, codes_release, \
                    codes_bufr_new_from_samples, CodesInternalError, \
                    codes_is_defined
import math
import numpy as np
from sklearn.linear_model import LinearRegression

from pypromice.postprocess.wmo_config import ibufr_settings, stid_to_skip, vars_to_skip, positions_update_timestamp_only

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

    Returns
    -------
    remove_file : boolean
        Status object to return to getBUFR indicating successful completion
    '''
    remove_file = False

    # Open bufr file
    fout = open(outBUFR, 'wb')

    # Create new bufr message to write to
    ibufr = codes_bufr_new_from_samples('BUFR4')
    timestamp = datetime.strptime(s1['time'], '%Y-%m-%d %H:%M:%S')
    config_key = 'mobile'
    if stid in land_stids:
        config_key = 'land'
    try:
        # we must pass all the following functions without error.
        # If handled (or unhandled) errors occur, we re-raise and
        # the exceptions below will set remove_file to True.
        setTemplate(ibufr, timestamp, stid, config_key)
        setStation(ibufr, stid, config_key)
        setAWSvariables(ibufr, s1, timestamp, stid)

        #Encode keys in data section
        codes_set(ibufr, 'pack', 1)

        #Write bufr message to bufr file
        codes_write(ibufr, fout)

    except CodesInternalError as ec:
        print(traceback.format_exc())
        # print(ec)
        print(f'-----> CodesInternalError in getBUFR for {stid}!')
        remove_file = True
    except Exception as e:
        # Catch anything else here...
        print(traceback.format_exc())
        # print(e)
        print(f'-----> ERROR in getBUFR for {stid}')
        remove_file = True

    codes_release(ibufr)

    fout.close()

    if remove_file is True:
        print(f'-----> Removing file for {stid}')
        os.remove(fout.name)
    return remove_file


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
            try:
                codes_set(ibufr, k, v[stid])
            except KeyError as e:
                print(f'-----> ID not found for {stid}')
                raise # throw error back to getBUFR where it is handled
        else:
            if codes_is_defined(ibufr, k) == 1:
                codes_set(ibufr, k, v)
            else:
               print(f'-----> setStation Key for {stid} not defined: {k}')
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
            print('-----> CodesInternalError in setBUFRvalue!')
            raise # throw error back to getBUFR where it is handled
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
    extrapolate : boolean
        If False (default), only apply linear fit to timestamps with valid data
        If True, then extrapolate positions based on linear fit model

    Returns
    -------
    df : pandas.Dataframe
        The original input df, with added column for the linear regression values
    pos_valid : boolean
        If True (default), sufficient valid data found in recent (limited) data.
        If False, we need to return this status to find_positions and use full station history instead.
    '''
    # print('=========== linear_fit ===========')
    pos_valid = True
    if column in df:
        df_dropna = df[df[column].notna()] # limit to only non-nan for the target column
        # if len(df_dropna[column].index.normalize().unique()) >= 10: # must have at least 10 unique days
        if len(df_dropna[column]) >= 15: # must have at least 15 data points (could be hourly or daily)
            # Get datetime x values into epoch sec integers
            x_epoch = df_dropna.index.values.astype(np.int64) // 10 ** 9
            x = x_epoch.reshape(-1,1)
            y = df_dropna[column].values # can also reshape this, but not necessary
            model = LinearRegression().fit(x, y)

            # Adding prediction back to original df
            x_all = df.index.values.astype(np.int64) // 10 ** 9
            df['{}_fit'.format(column)] = model.predict(x_all.reshape(-1,1)).round(decimals=decimals)

            # Plot data if desired
            # if stid == 'LYN_T':
            #     if (column == 'gps_lat') or (column == 'gps_lon') or (column == 'gps_alt'):
            #         import matplotlib.pyplot as plt
            #         plt.figure()
            #         df_dropna[column].plot(marker='o',ls='None')
            #         df['{}_fit'.format(column)].plot(marker='o', ls='None', color='red')
            #         plt.title('{} {}'.format(stid, column))
            #         plt.xlim(df.index.min(),df.index.max())
            #         plt.show()
        else:
            # Do not have 10 days of valid data, or all data is NaN.
            print('----> Insufficient {} data for {}!'.format(column, stid))
            pos_valid = False
    else:
        print('----> {} not found in dataframe!'.format(column))
        pass
    return df, pos_valid


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


def find_positions(df, stid, time_limit, current_timestamp=None, positions=None):
    ''' Driver function to run linear_fit() and set valid lat, lon, and alt
    to df_limited, which is then used to set position data in BUFR.
    If 'positions' is not None (must pass --positions arg), we also write to
    the positions dict which will be written to AWS_latest_locations.csv for
    all stations (whether processed or skipped)

    Parameters
    ----------
    df : pandas dataframe
        The full tx dataframe
    stid : str
        The station ID, such as NUK_L
    time_limit : str
        Previous time to limit dataframe before applying linear regression.
        (e.g. '3M')
    current_timestamp : datetime64 time
        The timestamp for the most recent valid instantaneous data
    positions : dict, or None
        Dict storing current station positions. If present, we are writing
        positions to file.

    Returns
    -------
    df_limited : pandas dataframe
        Dataframe limited to time_limit, and including position data
    positions : dict
        Modified dict storing most-recent station positions.
    '''
    if stid in positions_update_timestamp_only:
        # we don't have a position-associated timestamp, just use the most recent transmission.
        # e.g. KAN_B (does not transmit position, and currently skipped because does not transmit
        # instantaneous obs). If KAN_B ever submits inst data (but not position) we will need to use
        # the config-seeded position coordinates to set positions here in df_limited.
        positions[stid]['timestamp'] = df.index.max()
        df_limited = df # just to return something
    else:
        print(f'finding positions for {stid}')
        df_limited = df.last(time_limit).copy()
        print(f'last transmission: {df_limited.index.max()}')

        # Extrapolate recommended for altitude, optional for lat and lon.
        df_limited, lat_valid = linear_fit(df_limited, 'gps_lat', 6, stid)
        df_limited, lon_valid = linear_fit(df_limited, 'gps_lon', 6, stid)
        df_limited, alt_valid = linear_fit(df_limited, 'gps_alt', 1, stid)

        # If we have no valid lat, lon or alt data in the df_limited window, then interpolate
        # using full tx dataset.
        check_valid = {'gps_lat': lat_valid, 'gps_lon': lon_valid, 'gps_alt': alt_valid}
        check_valid_again = {}
        for k,v in check_valid.items():
            if v is False:
                print(f'----> Using full history for linear extrapolation: {k}')
                print(f'first transmission: {df.index.min()}')
                if k == 'gps_alt':
                    df, valid = linear_fit(df, k, 1, stid)
                else:
                    df, valid = linear_fit(df, k, 6, stid)
                check_valid_again[k] = valid
                if check_valid_again[k] is True:
                    df_limited[f'{k}_fit'] = df.last(time_limit)[f'{k}_fit']
                else:
                    print(f'----> No data exists for {k}. Stubbing out with NaN.')
                    df_limited[f'{k}_fit'] = pd.Series(np.nan, index= df.last(time_limit).index)

        # SET POSITIONS FOR CSV FILE
        if positions is not None:
            if current_timestamp is None:
                # This is old data (> 2 days), not submitting to DMI, but writing to positions csv
                # Find the most recent row that has valid lat, lon and alt
                last_valid_timestamp = df_limited[['gps_lon_fit','gps_lat_fit','gps_alt_fit']].dropna().last_valid_index()
                if last_valid_timestamp is None:
                    # we are likely missing gps_alt_fit
                    last_valid_timestamp = df_limited[['gps_lon_fit','gps_lat_fit']].dropna().last_valid_index()
                    if last_valid_timestamp is None:
                        # last ditch effort
                        last_valid_timestamp = df_limited.index.max()
                s = df_limited.loc[last_valid_timestamp]
            else:
                s = df_limited.loc[current_timestamp]
            print(f'writing positions for {stid}')
            pos_strings = ['lat','lon','alt']
            for p in pos_strings:
                if (f'gps_{p}_fit' in s) and (pd.isna(s[f'gps_{p}_fit']) is False):
                    positions[stid][p] = s[f'gps_{p}_fit']
            # Add timestamp
            positions[stid]['timestamp'] = s['time']

    return df_limited, positions if positions else df_limited


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
