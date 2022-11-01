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

from wmo_config import ibufr_settings

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
    setBUFRvalue(ibufr, 'windSpeed', row['wspd_i'])
    setBUFRvalue(ibufr, 'windDirection', row['wdir_i'])
    setBUFRvalue(ibufr, 'airTemperature', row['t_i'])
    setBUFRvalue(ibufr, 'pressure', row['p_i'])

    setBUFRvalue(ibufr, 'latitude', row['gps_lat_fit'])
    setBUFRvalue(ibufr, 'longitude', row['gps_lon_fit'])
    setBUFRvalue(ibufr, 'heightOfStationGroundAboveMeanSeaLevel', row['gps_alt_fit']) # also height and heightOfStation?
    setBUFRvalue(ibufr, 'heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform', row['z_boom_u'])

    #Set monitoring time period (-10=10 minutes)
    # if math.isnan(row['wspd_i']) is False:
    #     codes_set(ibufr, '#11#timePeriod', -10)

    #Set time significance (2=temporally averaged)
    # codes_set(ibufr, '#1#timeSignificance', 2)
    # if math.isnan(row['wspd_i']) is False:
    #     codes_set(ibufr, '#2#timeSignificance', 2)

    #Set measurement heights
    if math.isnan(row['z_boom_u']) is False:
        codes_set(ibufr,
                  '#2#heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform',
                  row['z_boom_u']-0.1) # For air temp
        codes_set(ibufr,
                  '#8#heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform',
                  row['z_boom_u']+0.4) # For wind speed
        if math.isnan(row['gps_alt_fit']) is False:
            codes_set(ibufr, 'heightOfBarometerAboveMeanSeaLevel',
                      row['gps_alt_fit']+row['z_boom_u'])


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
    df_dropna = df[df[column].notna()] # limit to only non-nan for the target column
    if len(df_dropna[column]) > 0:
        # Get datetime x values into epoch sec integers
        x_epoch = df_dropna.index.values.astype(np.int64) // 10 ** 9
        x = x_epoch.reshape(-1,1)
        y = df_dropna[column].values # can also reshape this, but not necessary
        model = LinearRegression().fit(x, y)
        y_pred = model.predict(x).round(decimals=decimals)

        # Plot data if desired
        # if column == 'gps_alt':
        #     import matplotlib.pyplot as plt
        #     plt.scatter(x,y)
        #     plt.plot(x,y_pred, color='red')
        #     plt.show()

        # Add y_pred back to original df
        df_dropna['y_pred'] = y_pred
        df['{}_fit'.format(column)] = df_dropna['y_pred']
    else:
        # There is no data! Just write NaNs using the original column
        print('----> No {} data for {}!'.format(column, stid))
        df['{}_fit'.format(column)] = df[column]
    return df


def rolling_window(df, column, window, min_periods, decimals):
    '''Apply a rolling window (smoothing) to the input column
    CURRENTLY NOT USED

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

#------------------------------------------------------------------------------

if __name__ == '__main__':
    # Get station names and file paths
    if args.dev is True:
        l3path = args.l3_path_dev
        fpaths = glob.glob(args.l3_files_dev)
    else:
        print('Prod paths not yet defined. Need to pass --dev.')
        sys.exit()

    stns = [name for name in os.listdir(l3path) if os.path.isdir(l3path)]
    # Note that stn count includes Roof_PROMICE, Roof_GEUS and XXX
    print('Processing {} stations'.format(len(stns)))

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

    # Iterate through csv files
    for fname in fpaths:
        # Generate output BUFR filename
        last_index = fname.rfind('_')
        first_index = fname.rfind('/')
        stid = fname[first_index+1:last_index]
        # stid = fname.split('/')[-1].split('.csv')[0][:-5]
        print('####### Processing {} #######'.format(stid))
        bufrname = stid + '.bufr'
        print(f'Generating {bufrname} from {fname}')

        # Read csv file
        df1 = pd.read_csv(fname, delimiter=',')
        df1.set_index(pd.to_datetime(df1['time']), inplace=True)
        df1.sort_index(inplace=True) # make sure we are time-sorted

        current_timestamp = df1.index.max()
        # set in dict, will be written back to disk at end
        current_timestamps[stid] = current_timestamp

        if stid in latest_timestamps:
            latest_timestamp = latest_timestamps[stid]
            two_days_ago = datetime.utcnow() - timedelta(days=2)

            if 1 == 1: # dev bypass
            # if (current_timestamp > latest_timestamp) and (current_timestamp > two_days_ago):
                print('Time checks passed.')
                # limit the dataframe for linear regression
                df1_limited = df1.last(args.time_limit)
                # Add '{}_fit' to df (linear fit of alt, lat, lon)
                df1_limited = linear_fit(df1_limited, 'gps_alt', 1, stid)
                df1_limited = linear_fit(df1_limited, 'gps_lat', 6, stid)
                df1_limited = linear_fit(df1_limited, 'gps_lon', 6, stid)

                s1_current = df1_limited.loc[current_timestamp] # limit to single most recent row (series)

                # Convert air temp, C to Kelvin
                s1_current.t_i = s1_current.t_i + 273.15

                # Convert pressure, correct the -1000 offset, then hPa to Pa
                # note that instantaneous pressure has 0.1 hPa precision
                s1_current.p_i = (s1_current.p_i+1000.) * 100.

                # Construct and export BUFR file
                getBUFR(s1_current, outFiles+bufrname, stid)
                print(f'Successfully exported bufr file to {outFiles+bufrname}')
                # if stid == 'KPC_U':
                #     embed()
            else:
                print('Current ob not processed for {}'.format(stid))
                print('current:', current_timestamp)
                print('latest:', latest_timestamp)
        else:
            print('{} not found in latest_timestamps'.format(stid))
    # Write the most recent timestamps back to the pickle on disk
    print('writing latest_timestamps.pickle')
    with open('latest_timestamps.pickle', 'wb') as handle:
        pickle.dump(current_timestamps, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Finished processing {} stations.'.format(len(stns)))
