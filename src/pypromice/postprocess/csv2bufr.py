#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 13:44:49 2021

@author: pho

Playground script for converting PROMICE .csv files to WMO-compliant BUFR files

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

According to DMI, the BUFR messages should adhere to Common Code Table 13:
https://confluence.ecmwf.int/display/ECC/WMO%3D13+element+table#WMO=13elementtable-CL_1
"""
import pandas as pd
import glob, os, sys
from datetime import datetime
from eccodes import codes_set, codes_write, codes_release, \
                    codes_bufr_new_from_samples, CodesInternalError
import math
import datetime as dt

from args import args

from IPython import embed

#------------------------------------------------------------------------------

def setBUFRvalue(ibufr, b_name, value):
    '''Set variable in BUFR message
    Maybe investigate codes_set_array?? PJW

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
    #else:
    # do we need to specifically set a nan in the ibufr? PJW


def setTemplate(ibufr, timestamp, ed=4, master=0, vers=13, 
                template=307080, key='unexpandedDescriptors'):
    '''Set bufr message template.

    Parameters
    ----------
    ibufr : bufr.msg
        Bufr message object
    timestamp: datetime.Datetime
        Timestamp of observation
    ed : int
        Edition. The default is 4.
    master : int
        Master table number. The default is 0.
    vers : int
        Master table version number. The default is 13.
        The online example now shows 31 PJW
    template : int
        Template number. The default is 307080 ("synopLand"), which DMI uses.
        What about 307091 "surfaceObservationOneHour"?,
        or 307096 "synopOneHour"? PJW
    key : str
        Encoding type. The default is "unexpandedDescriptors".
    '''  
    #Indicator section (BUFR 4 letters, total msg size, edition number)
    #Current edition is version 4                   
    codes_set(ibufr, 'edition', ed)

    #Identification section (master table, id, sequence number, data cat number)
    codes_set(ibufr, 'masterTableNumber', master)
    codes_set(ibufr, 'masterTablesVersionNumber', vers)
    codes_set(ibufr, 'localTablesVersionNumber', 0)

    #BUFR header centre 98 = ECMF
    codes_set(ibufr, 'bufrHeaderCentre', 98)
    codes_set(ibufr, 'bufrHeaderSubCentre', 0)
    codes_set(ibufr, 'updateSequenceNumber', 0)

    #Data category 0 = surface data, land
    codes_set(ibufr, 'dataCategory', 0)

    #International data subcategory 7 = n-min obs from AWS stations
    codes_set(ibufr, 'internationalDataSubCategory', 7)
    codes_set(ibufr, 'dataSubCategory', 7)

    codes_set(ibufr, 'observedData', 1)
    codes_set(ibufr, 'compressedData', 0)

    codes_set(ibufr, 'typicalYear', timestamp.year)
    codes_set(ibufr, 'typicalMonth', timestamp.month)
    codes_set(ibufr, 'typicalDay', timestamp.day)
    codes_set(ibufr, 'typicalHour', timestamp.hour)
    codes_set(ibufr, 'typicalMinute', timestamp.minute)
    codes_set(ibufr, 'typicalSecond', timestamp.second)

    #Assign message template
    ivalues = (template)

    #Assign key name to encode sequence number
    codes_set(ibufr, key, ivalues)


def setStation(ibufr, stationNumber, blockNumber):
    '''Set station info to bufr message.

    Parameters
    ----------
    ibufr : bufr.msg
        Bufr message object
    '''
    #Data Description and Binary Data section
    #Set AWS station info

    #Need to set WMO block and station number
    codes_set(ibufr, 'stationNumber', 1)
    codes_set(ibufr, 'blockNumber', 1)
    # codes_set(ibufr, 'wmoRegionSubArea', 1)

    #Region number=7 (unknown)
    # codes_set(ibufr, 'regionNumber', 7)

    #Unset parameters
    # codes_set(ibufr, 'stationOrSiteName', CCITT IA5)
    # codes_set(ibufr, 'shortStationName', CCITT IA5)
    # codes_set(ibufr, 'shipOrMobileLandStationIdentifier', CCITT IA5)
    # codes_set(ibufr, 'directionOfMotionOfMovingObservingPlatform', deg)
    # codes_set(ibufr, 'movingObservingPlatformSpeed', m/s)

    codes_set(ibufr, 'stationType', 0)
    codes_set(ibufr, 'instrumentationForWindMeasurement', 6)
    # codes_set(ibufr, 'measuringEquipmentType', 0)
    # codes_set(ibufr, 'temperatureObservationPrecision', 0.1)
    # codes_set(ibufr, 'pressureSensorType', 30)
    # codes_set(ibufr, 'temperatureSensorType', 30)
    # codes_set(ibufr, 'humiditySensorType', 30)   


def setAWSvariables(ibufr, row, timestamp):
    '''Set AWS measurements to bufr message.
    
    Parameters
    ----------
    ibufr : bufr.msg
        Bufr message object
    row : pandas.DataFrame
        DataFrame row with AWS info
    timestamp: datetime.datetime
        timestamp for this row
    '''
    #Set baseline AWS info
    setBUFRvalue(ibufr, 'year', timestamp.year)
    setBUFRvalue(ibufr, 'month', timestamp.month)
    setBUFRvalue(ibufr, 'day', timestamp.day)
    setBUFRvalue(ibufr, 'hour', timestamp.hour)
    setBUFRvalue(ibufr, 'minute', timestamp.minute)

    setBUFRvalue(ibufr, 'relativeHumidity', row['rh_i']) # rh_i vs rh_i_cor? PJW
    setBUFRvalue(ibufr, 'windSpeed', row['wspd_i'])
    setBUFRvalue(ibufr, 'windDirection', row['wdir_i'])
    setBUFRvalue(ibufr, 'airTemperature', row['t_i'])
    setBUFRvalue(ibufr, 'pressure', row['p_i'])

    setBUFRvalue(ibufr, 'latitude', row['gps_lat'])
    setBUFRvalue(ibufr, 'longitude', row['gps_lon'])
    setBUFRvalue(ibufr, 'heightOfStationGroundAboveMeanSeaLevel', row['gps_alt_smooth'])
    setBUFRvalue(ibufr, 'heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform', row['z_boom_u'])

    #Set monitoring time period (-10=10 minutes)
    if math.isnan(row['wspd_i']) is False:
        codes_set(ibufr, '#11#timePeriod', -10)

    #Set time significance (2=temporally averaged)
    codes_set(ibufr, '#1#timeSignificance', 2)
    if math.isnan(row['wspd_i']) is False:
        codes_set(ibufr, '#2#timeSignificance', 2)

    #Set measurement heights
    if math.isnan(row['z_boom_u']) is False:
        codes_set(ibufr,
                  '#2#heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform',
                  row['z_boom_u']-0.1) # For air temp
        codes_set(ibufr,
                  '#8#heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform',
                  row['z_boom_u']+0.4) # For wind speed
        if math.isnan(row['gps_alt']) is False:
            codes_set(ibufr, 'heightOfBarometerAboveMeanSeaLevel',
                      row['gps_alt_smooth']+row['z_boom_u'])


def getBUFR(df1, outBUFR, ed=4, master=0, vers=13,
            template=307080, key='unexpandedDescriptors'):
    '''Construct and export .bufr messages to file from DataFrame.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Pandas dataframe of weather station observations
    outBUFR : str
        File path that .bufr file will be exported to
    ed : int
        BUFR table edition. The default is 4.
    master : int
        Master table number. The default is 0, standard WMO FM 94 BUFR tables
    vers : int
        Master table version number. The default is 13.
    template : int
        Template table number. The default is 307080.
    key : str
        Encoding key name. The default is "unexpandedDescriptors".
    '''
    #Open bufr file
    fout = open(outBUFR, 'wb')

    #Iterate over rows in weather observations dataframe
    for i1, r1 in df1.iterrows():

        #Create new bufr message to write to
        ibufr = codes_bufr_new_from_samples('BUFR4')

        try:
            #Get timestamp
            timestamp = datetime.strptime(r1['time'], '%Y-%m-%d %H:%M:%S')

            #Set table formatting and templating
            #Strange to set this for every row... PJW
            setTemplate(ibufr, timestamp)

            #Set station info !!!! CHANGE !!!!
            stationNumber=1
            blockNumber=1
            setStation(ibufr, stationNumber, blockNumber)

            #Set AWS measurments
            setAWSvariables(ibufr, r1, timestamp)

            #Encode keys in data section
            codes_set(ibufr, 'pack', 1)                     

            #Write bufr message to bufr file
            codes_write(ibufr, fout)

        except CodesInternalError as ec:
            print(ec)

        codes_release(ibufr)

    fout.close()


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

    # Get lookup table NOT USED!
    # lookup = pd.read_csv('./variables_bufr.csv', delimiter=',')

    # Make out dir
    outFiles = args.bufr_out
    if os.path.exists(outFiles) is False:
        os.mkdir(outFiles)

    # Iterate through csv files
    for fname in fpaths:
        # Generate output BUFR filename
        last_index = fname.rfind('_')
        first_index = fname.rfind('/')
        bufrname = fname[first_index+1:last_index]+'.bufr'
        # bufrname = fname.split('/')[-1].split('.csv')[0][:-5]+'.bufr'
        print(f'Generating {bufrname} from {fname}')

        # Read csv file
        df1 = pd.read_csv(fname, delimiter=',')
        # df1.index = pd.to_datetime(df1['time'])
        df1.set_index(pd.to_datetime(df1['time']), inplace=True)

        # Convert air temp, C to Kelvin
        df1.t_i = df1.t_i + 273.15

        # Convert pressure, correct the -1000 offset, then hPa to Pa
        # note that instantaneous pressure has 0.01 hPa precision
        df1.p_i = (df1.p_i+1000.) * 100.

        # Smooth altitude
        df1['gps_alt_smooth'] = df1['gps_alt'].rolling(
            '24H',
            min_periods=1,
            center=True, # set the window labels as the center of the window
            closed='both' # no points in the window are excluded (first or last)
            ).mean().round(decimals=1) # could also round to whole meters (decimals=0)?

        df1_limited = df1.last(args.time_limit) # limit to previous 2 weeks

        # Construct and export BUFR file
        getBUFR(df1_limited, outFiles+bufrname)
        print(f'Successfully exported bufr file to {outFiles+bufrname}')  

    print('Finished')
