#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 13:44:49 2021
Major modifications: Oct 2022

@author: pho, pajwr

Script for converting PROMICE .csv files to WMO-compliant BUFR files

This script uses the package eccodes to run. 
https://confluence.ecmwf.int/display/ECC/ecCodes+installation
Eccodes is the official package for WMO BUFR file construction. Eccodes must 
be configured on your computer BEFORE downloading the eccodes python bindings.
Eccodes can be configured with the conda python bindings, using the command
'conda install eccodes', however this didn't seem to work for me. Instead, I 
built eccodes separately and then installed the python bindings using pip3.
 
See here for a step-by-step guide on the eccodes set-up:
https://gist.github.com/MHBalsmeier/a01ad4e07ecf467c90fad2ac7719844a

Processing steps based on this example:
https://confluence.ecmwf.int/display/UDOC/How+do+I+create+BUFR+from+a+CSV+-+ecCodes+BUFR+FAQ
"""
import pandas as pd
import glob, os, sys, traceback
from datetime import datetime, timedelta
from eccodes import codes_set, codes_write, codes_release, \
                    codes_bufr_new_from_samples, CodesInternalError, \
                    codes_is_defined
import math
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

from args import args

from wmo_config import ibufr_settings, stid_to_skip

from IPython import embed

# To suppress pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None # default='warn'

#------------------------------------------------------------------------------

def getBUFR(s1, outBUFR, stid):
    '''Construct and export .bufr messages to file from Series or DataFrame.
    PRIMARY DRIVER FUNCTION

    Parameters
    ----------
    s1 : pandas.Series
        Pandas series of single most recent obset for a station
    outBUFR : str
        File path that .bufr file will be exported to
    stid: str
        The station ID to be processed. e.g. 'KPC_U'
    '''
    # Open bufr file
    fout = open(outBUFR, 'wb')

    # for i1, r1 in df1.iterrows(): # If dataframe passed w/ multiple rows

    #Create new bufr message to write to
    ibufr = codes_bufr_new_from_samples('BUFR4')
    timestamp = datetime.strptime(s1['time'], '%Y-%m-%d %H:%M:%S')

    try:
        setTemplate(ibufr, timestamp)
        setStation(ibufr, stid)
        setAWSvariables(ibufr, s1, timestamp)

        #Encode keys in data section
        codes_set(ibufr, 'pack', 1)

        #Write bufr message to bufr file
        codes_write(ibufr, fout)

    except CodesInternalError as ec:
        print(traceback.format_exc())
        print(ec)
        sys.exit('-----> CodesInternalError in getBUFR!')
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        sys.exit('!!!!!!!!!! ERROR in getBUFR')

    codes_release(ibufr)

    fout.close()


def setTemplate(ibufr, timestamp):
    '''Set bufr message template.

    Parameters
    ----------
    ibufr : bufr.msg
        Bufr message object
    timestamp: datetime.Datetime
        Timestamp of observation
    '''
    for k, v in ibufr_settings['template'].items():
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


def setStation(ibufr, stid):
    '''Set station-specific info to bufr message.

    Parameters
    ----------
    ibufr : bufr.msg
        Bufr message object
    stid: str
        The station ID to be processed. e.g. 'KPC_U'
    '''
    for k, v in ibufr_settings['station'].items():
        if codes_is_defined(ibufr, k) == 1:
            if k == 'shipOrMobileLandStationIdentifier':
                if stid in v:
                    codes_set(ibufr, k, str(v[stid]))
                else:
                    codes_set(ibufr, k, '1111111') # for testing
                    # sys.exit('!!!!!!!!!! WMO ID not found for {}'.format(stid))
            else:
                codes_set(ibufr, k, v)
        else:
            print('-----> setStation Key not defined: {}'.format(k))
            continue


def setAWSvariables(ibufr, row, timestamp):
    '''Set AWS measurements to bufr message.
    
    Parameters
    ----------
    ibufr : bufr.msg
        Bufr message object
    row : pandas.DataFrame row, or pandas.Series
        DataFrame row (or Series) with AWS variable data
    timestamp: datetime.datetime
        timestamp for this row
    '''
    setBUFRvalue(ibufr, 'year', timestamp.year)
    setBUFRvalue(ibufr, 'month', timestamp.month)
    setBUFRvalue(ibufr, 'day', timestamp.day)
    setBUFRvalue(ibufr, 'hour', timestamp.hour)
    setBUFRvalue(ibufr, 'minute', timestamp.minute)

    setBUFRvalue(ibufr, 'relativeHumidity', row['rh_i']) # DMI wants non-corrected
    setBUFRvalue(ibufr, 'airTemperature', row['t_i'])
    setBUFRvalue(ibufr, 'pressure', row['p_i'])
    setBUFRvalue(ibufr, 'windDirection', row['wdir_i'])
    setBUFRvalue(ibufr, 'windSpeed', row['wspd_i'])

    setBUFRvalue(ibufr, 'latitude', row['gps_lat_fit'])
    setBUFRvalue(ibufr, 'longitude', row['gps_lon_fit'])
    setBUFRvalue(ibufr, 'heightOfStationGroundAboveMeanSeaLevel', row['gps_alt_fit']) # also height and heightOfStation?

    # The ## in the codes_set() indicate the position in the BUFR for the parameter.
    # e.g. #10#timePeriod will assign to the 10th occurence of "timePeriod".
    # In the case of the synopMobil template, the 10th occurence is the wind speed section.
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

    Parameters
    ----------
    df : pandas.Dataframe
        datetime-indexed df, limited to desired time length for linear fit
    column : str
        The target column for applying linear fit
    decimals : int
        How many decimals to round the output fit values
    stid: str
        The station ID to be processed. e.g. 'KPC_U'

    Returns
    -------
    df : pandas.Dataframe
        The original input df, with added column for the linear regression values

    Linear regression is following:
    https://realpython.com/linear-regression-in-python/#simple-linear-regression-with-scikit-learn
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
            # if (column == 'gps_lat') or (column == 'gps_lon') or (column == 'gps_alt'):
            #     import matplotlib.pyplot as plt
            #     plt.scatter(x,y)
            #     plt.plot(x,y_pred, color='red')
            #     plt.title('{} {}'.format(stid, column))
            #     plt.show()

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

def write_positions(s, stid):
    '''Set valid lat, lon, alt to the positions dict.
    Find recent position metadata for submitting to DMI/WMO.
    Not used in production!

    Parameters
    ----------
    s : pandas series
        The current obset we are working with (for BUFR submission)
    stid : str
        The station ID, such as NUK_L

    Returns
    -------
    None
    '''
    to_write = ['lat','lon']
    for i in to_write:
        if (f'gps_{i}_fit' in s) and (pd.isna(s[f'gps_{i}_fit']) is False):
            positions[stid][i] = s[f'gps_{i}_fit']

    # Add altitude to positions dict:
    if ('gps_alt_fit' in s) and (pd.isna(s['gps_alt_fit']) is False):
        positions[stid]['alt'] = s['gps_alt_fit']

    # Add timestamp
    positions[stid]['timestamp'] = s['time']

def fetch_old_positions(df, stid):
    '''Set valid lat, lon, alt to the positions dict.
    Used to find old GPS positions for submitting position metadata to DMI/WMO.
    Not used in production!

    Parameters
    ----------
    df : pandas dataframe
        The full tx dataframe
    stid : str
        The station ID, such as NUK_L

    Returns
    -------
    None
    '''
    # Find valid GPS data
    valid_gps = df.dropna(subset=['gps_lat','gps_lon','gps_alt'])

    if valid_gps.empty is False:
        valid_gps_limited = valid_gps.last(args.time_limit)

        valid_gps_limited = linear_fit(valid_gps_limited, 'gps_alt', 1, stid)
        valid_gps_limited = linear_fit(valid_gps_limited, 'gps_lat', 6, stid)
        valid_gps_limited = linear_fit(valid_gps_limited, 'gps_lon', 6, stid)

        s = valid_gps_limited.loc[valid_gps_limited.index.max()]

        to_write = ['lat','lon']
        for i in to_write:
            if (f'gps_{i}_fit' in s) and (pd.isna(s[f'gps_{i}_fit']) is False):
                positions[stid][i] = s[f'gps_{i}_fit']
                positions[stid][f'{i}_s'] = 'OLD' #source flag

        # Add altitude to positions dict:
        if ('gps_alt_fit' in s) and (pd.isna(s['gps_alt_fit']) is False):
            positions[stid]['alt'] = s['gps_alt_fit']

        # Add timestamp
        positions[stid]['timestamp'] = s['time']


def min_data_check(s, stid):
    '''Check that we have minimum required fields to proceed with writing to BUFR

    Parameters
    ----------
    s : pandas series
        The current obset we are working with (for BUFR submission)
    stid : str
        The station ID, such as NUK_L

    Returns
    -------
    result : bool
        True (default), the test passed. False, the test failed.
    '''
    result = True

    # Can use pd.isna() or math.isnan()
    # Must have valid air temp and pressure
    if (pd.isna(s['t_i']) is False) and (pd.isna(s['p_i']) is False):
        pass
    else:
        print('----> Failed min_data_check for air temp and pressure!')
        failed_min_data_wx.append(stid)
        result = False

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
        failed_min_data_pos.append(stid)
        result = False

    return result
#------------------------------------------------------------------------------

if __name__ == '__main__':
    # Get station names and file paths
    if args.dev is True:
        # l3path = args.l3_path_dev
        fpaths = glob.glob(args.l3_files_dev)
    else:
        sys.exit('Prod paths not yet defined. Need to pass --dev.')

    # stns = [name for name in os.listdir(l3path) if os.path.isdir(l3path)]

    # Make out dir
    outFiles = args.bufr_out
    if os.path.exists(outFiles) is False:
        os.mkdir(outFiles)

    # Read existing timestamps pickle to dictionary
    if os.path.isfile('latest_timestamps.pickle'):
        with open('latest_timestamps.pickle', 'rb') as handle:
            latest_timestamps = pickle.load(handle)
    else:
        print('latest_timestamps.pickle not found!')
        latest_timestamps = {}

    # Initiate a new dict for current timestamps
    current_timestamps = {}

    if args.dev is True:
        # Initiate a dict to store station positions
        # Used to retrieve a static set of positions to register with WMO
        positions = {}

    # Define stations to skip
    to_skip = []
    for k, v in stid_to_skip.items():
        to_skip.extend(v)
    to_skip = set(to_skip) # Get rid of any duplicates

    # Setup diagnostic lists (print at end)
    skipped = []
    no_recent_data = []
    no_valid_data = []
    partial_data = []
    no_entry_latest_timestamps = []
    failed_min_data_wx = []
    failed_min_data_pos = []

    # Iterate through csv files
    for f in fpaths:
        last_index = f.rfind('_')
        first_index = f.rfind('/')
        stid = f[first_index+1:last_index]
        # stid = f.split('/')[-1].split('.csv')[0][:-5]

        if ('Roof' not in f) and (stid not in to_skip):
        # if ('v3' not in f) and ('Roof' not in f) and (stid not in to_skip):
            print('####### Processing {} #######'.format(stid))
            bufrname = stid + '.bufr'
            print(f'Generating {bufrname} from {f}')

            if args.dev is True:
                positions[stid] = {}
                positions[stid]['lat_s'] = ''
                positions[stid]['lon_s'] = ''

            # Read csv file
            df1 = pd.read_csv(f, delimiter=',')
            df1.set_index(pd.to_datetime(df1['time']), inplace=True)
            df1.sort_index(inplace=True) # make sure we are time-sorted

            # Check that the last valid index for all instaneous values match
            # Note: we cannot always use the single most-recent timestamp in the dataframe
            # e.g. for 6-hr transmissions, *_u will have hourly data while *_i is nan
            # Need to check for last valid (non-nan) index instead
            lvi = {'t_i': df1['t_i'].last_valid_index(),
                   'p_i': df1['p_i'].last_valid_index(),
                   'rh_i': df1['rh_i'].last_valid_index(),
                   'wspd_i': df1['wspd_i'].last_valid_index(),
                   'wdir_i': df1['wdir_i'].last_valid_index()
                   }

            two_days_ago = datetime.utcnow() - timedelta(days=2)

            if len(set(lvi.values())) != 1:
                # instantaneous vars have different timestamps
                recent = {}
                for k,v in lvi.items():
                    if (v is not None) and (v >= two_days_ago):
                        recent[k] = v
                if len(recent) == 0:
                    print('No recent instantaneous timestamps!')
                    no_recent_data.append(stid)
                    if args.dev is True:
                        fetch_old_positions(df1,stid)
                    continue
                else:
                    # we have partial data, just use the most recent row
                    partial_data.append(stid)
                    current_timestamp = max(recent.values())
                    # We will throw this obset down the line, and there is a final min_data_check
                    # to make sure we have minimum data requirements before writing to BUFR
            else:
                if all(i is None for i in lvi.values()) is True:
                    print('All instantaneous timestamps are None!')
                    no_valid_data.append(stid)
                    if args.dev is True:
                        fetch_old_positions(df1,stid)
                    continue
                else:
                    # all values are present, with matching timestamps, so just use t_i
                    current_timestamp = df1['t_i'].last_valid_index()

            print(f'TIMESTAMP: {current_timestamp}')

            # set in dict, will be written to disk at end
            current_timestamps[stid] = current_timestamp

            if stid in latest_timestamps:
                latest_timestamp = latest_timestamps[stid]

                if current_timestamp > two_days_ago: # dev bypass
                # if (current_timestamp > latest_timestamp) and (current_timestamp > two_days_ago):
                    print('Time checks passed.')
                    # limit the dataframe for linear regression (e.g. previous 3 months)
                    df1_limited = df1.last(args.time_limit)

                    # Combine gps and msg lat and lon using combine_first()
                    # If any GPS positions are missing, we will fill the missing GPS positions with modem
                    # positions (if they are present). Important to do this first, and then apply linear fit
                    # to the resulting single array. Otherwise, we can have jumps when the GPS data goes out
                    # or comes back. The message coordinates can sometimes all be 0.0, so check for this.
                    if ('msg_lat' in df1_limited) and ('msg_lon' in df1_limited):
                        if (0.0 not in df1_limited['msg_lat'].values):
                            df1_limited['gps_lat'] = df1_limited['gps_lat'].combine_first(df1_limited['msg_lat'])
                        if (0.0 not in df1_limited['msg_lon'].values):
                            df1_limited['gps_lon'] = df1_limited['gps_lon'].combine_first(df1_limited['msg_lon'])

                    # Add '{}_fit' to df (linear fit of alt, lat, lon)
                    df1_limited = linear_fit(df1_limited, 'gps_alt', 1, stid)
                    df1_limited = linear_fit(df1_limited, 'gps_lat', 6, stid)
                    df1_limited = linear_fit(df1_limited, 'gps_lon', 6, stid)

                    # Apply smoothing to z_boom_u
                    # require at least 2 hourly obs? Sometimes seeing once/day data for z_boom_u
                    df1_limited = rolling_window(df1_limited, 'z_boom_u', '72H', 2, 1)

                    # limit to single most recent valid row (convert to series)
                    s1_current = df1_limited.loc[current_timestamp]

                    # Convert air temp, C to Kelvin
                    s1_current.t_i = s1_current.t_i + 273.15

                    # Convert pressure, correct the -1000 offset, then hPa to Pa
                    # note that instantaneous pressure has 0.1 hPa precision
                    s1_current.p_i = (s1_current.p_i+1000.) * 100.

                    s1_current = round_values(s1_current)

                    if args.dev is True:
                        write_positions(s1_current, stid)

                    # Check that we have minimum required valid data
                    # This should really only apply to the case of partial data (from the timestamp checks above)
                    # However, good sanity check to just run this once here for all stids
                    min_data_result = min_data_check(s1_current, stid)
                    if min_data_result is False:
                        continue

                    # Construct and export BUFR file
                    getBUFR(s1_current, outFiles+bufrname, stid)
                    print(f'Successfully exported bufr file to {outFiles+bufrname}')
                else:
                    print('Time checks failed for {}'.format(stid))
                    print('current:', current_timestamp)
                    print('latest:', latest_timestamp)
                    no_recent_data.append(stid)
                    if args.dev is True:
                        fetch_old_positions(df1,stid)
            else:
                print('{} not found in latest_timestamps'.format(stid))
                no_entry_latest_timestamps.append(stid)
        else:
            skipped.append(stid)
    # Write the most recent timestamps back to the pickle on disk
    print('writing latest_timestamps.pickle')
    with open('latest_timestamps.pickle', 'wb') as handle:
        pickle.dump(current_timestamps, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.dev is True:
        positions_df = pd.DataFrame.from_dict(
            positions,
            orient='index',
            columns=['timestamp','lat','lon','alt','lat_s','lon_s']
            )
        positions_df.sort_index(inplace=True)
        positions_df.to_csv('positions.csv')

    print('--------------------------------')
    not_processed_count = len(skipped) + len(no_recent_data) + len(no_valid_data) + len(no_entry_latest_timestamps) + len(failed_min_data_wx) + len(failed_min_data_pos)
    print('Finished processing {} of {} fpaths.'.format((len(fpaths) - not_processed_count),len(fpaths)))
    print('')
    print('skipped: {}'.format(skipped))
    print('no_recent_data: {}'.format(no_recent_data))
    print('no_valid_data: {}'.format(no_valid_data))
    print('partial_data: {}'.format(partial_data)) # This can still be processed
    print('no_entry_latest_timestamps: {}'.format(no_entry_latest_timestamps))
    print('failed_min_data_wx: {}'.format(failed_min_data_wx))
    print('failed_min_data_pos: {}'.format(failed_min_data_pos))
    print('--------------------------------')
    embed()