#!/usr/bin/env python3
'''
Config file to store wmo-related reference objects
Imported by csv2bufr.py
Patrick Wright, GEUS
Nov, 2022

see documentation here:
https://confluence.ecmwf.int/display/ECC/Documentation

BUFR element table for WMO master table version 32
https://confluence.ecmwf.int/display/ECC/WMO%3D32+element+table
'''
stid_to_skip = { # All the following IDS will not be processed or submitted
    'test': ['XXX'],
    'not_registered': ['UWN','ZAK_A','WEG_L'], # Need to register UWN with Norwegion met
    'discontinued': ['CEN1','TAS_U','QAS_A','NUK_N','THU_U','JAR','SWC'],
    'no_instantaneous': ['ZAK_L','ZAK_U','KAN_B'], # currently not transmitting instantaneous values
    'suspect_data': [], # instantaneous data is suspect
    'use_v3': ['NUK_U', 'ZAK_L', 'ZAK_U', 'QAS_U', 'QAS_L', 'QAS_M', 'KAN_L'], # use v3 versions instead (but registered IDs are non-v3 names)
    'v3_bad': ['KPC_L', 'KPC_U', ]
}
# NOTE: Use both THU_L and THU_L2; use ONLY THU_U2, but register it as THU_U (this is dealt with in csv2bufr.py)
# NOTE: JAR_O and SWC_O are used, but registered as JAR and SWC
# NOTE: CEN2 data is registered as CEN

vars_to_skip = { # skip specific variables for stations
    # If a variable has known bad data, use this dict to skip the var
    # Note that if a station is not reporting both air temp and pressure it will be skipped,
    # as currently implemented in csv2bufr.min_data_check().
    # 'CP1': ['p_i'], # EXAMPLE
}

positions_seed = { # discontinued stations that are not in aws-l3/tx but still present in aws-l3/level_3
    # enter last known positions and timestamp of last transmission
    'TAS_U': {'lat':65.6978, 'lon':-38.8668, 'alt':570.0,  'timestamp':'2015-08-13 14:00:00'},
    'QAS_A': {'lat':61.243,  'lon':-46.7328, 'alt':1000.0, 'timestamp':'2015-08-24 17:00:00'},
    'NUK_N': {'lat':64.9452, 'lon':-49.885,  'alt':920.0,  'timestamp':'2014-07-25 11:00:00'},
    'KAN_B': {'lat':67.1252, 'lon':-50.1832, 'alt':350.0,  'timestamp':'2023-01-01 00:00:00'}, # bedrock station, not transmitting coordinates (placeholder timestamp)
}

positions_update_timestamp_only = ('KAN_B',)

ibufr_settings = {
    'mobile': { # mobile stations (on moving ice)
        'template': {
            'unexpandedDescriptors': (307090), #message template, "synopMobil"
            'edition': 4, #latest edition
            'masterTableNumber': 0,
            'masterTablesVersionNumber': 32, #DMI recommends any table version between 28-32
            'localTablesVersionNumber': 0,
            'bufrHeaderCentre': 94, #originating centre 98=ECMWF, 94=DMI
            # 'bufrHeaderSubCentre': 0,
            'updateSequenceNumber': 0, #0 is original message, incremented by 1 for updates
            'dataCategory': 0, #surface data - land
            'internationalDataSubCategory': 3, #hourly synoptic observations from mobile-land stations (SYNOP MOBIL)
            # 'dataSubCategory': 0,
            'observedData': 1,
            'compressedData': 0,
        },
        'station': {
            'shipOrMobileLandStationIdentifier': {
                # use str; fails with "gribapi.errors.InvalidArgumentError: Invalid argument" if passed as int
                'QAS_L':  '04401',
                'QAS_U':  '04402',
                'NUK_L':  '04403',
                'TAS_L':  '04404',
                'CEN':    '04407',
                'TAS_A':  '04408',
                'KAN_U':  '04409',
                'KAN_M':  '04411',
                'KAN_L':  '04412',
                'SCO_L':  '04413',
                'NAE':    '04420',
                'SCO_U':  '04421',
                'UPE_U':  '04422',
                'UPE_L':  '04423',
                'THU_L':  '04424',
                'TUN':    '04425',
                'KPC_U':  '04427',
                'KPC_L':  '04428',
                'LYN_T':  '04429',
                'MIT':    '04430',
                'HUM':    '04432',
                'NEM':    '04436',
                'NUK_K':  '04437',
                'NUK_U':  '04439',
                'QAS_M':  '04441',
                'CP1':    '04442',
                'NAU':    '04443',
                'LYN_L':  '04450',
                'EGP':    '04451',
                'JAR':    '04452',
                'THU_L2': '04453',
                'THU_U':  '04454',
                'SWC':    '04458',
                'ZAK_L':  '04461',
                'ZAK_U':  '04462',
                'DY2':    '04464',
                'SDL':    '04485',
                'NSE':    '04488',
                'SDM':    '04492'
            },
            # 'blockNumber': 4, #4 is Greenland, 6 is Denmark; not valid with synopMobil template
            'regionNumber': 6, #6 is Europe, 7 is MISSING VALUE; not valid with synopLand template
            'centre': 94, #94 is Copenhagen
            # 'agencyInChargeOfOperatingObservingPlatform': , #nothing for DMI or GEUS in code table
            # 'wmoRegionSubArea': 1,
            # 'stationOrSiteName': , #not valid with synopMobil template
            # 'shortStationName': , #not valid with synopMobil template
            # 'longStationName': , #not valid with synopMobil template
            # 'directionOfMotionOfMovingObservingPlatform': ,
            # 'movingObservingPlatformSpeed': ,
            'stationType': 0, #automatic station
            'instrumentationForWindMeasurement': 8, #certified instruments
            'stationElevationQualityMarkForMobileStations': 1, #Excellent - within 3m; not valid with synopLand template
        }
    },
    'land': { # land-based (non-mobile) stations
        'template': {
            'unexpandedDescriptors': (307080), #message template, "synopLand"
            'edition': 4, #latest edition
            'masterTableNumber': 0,
            'masterTablesVersionNumber': 32, #DMI recommends any table version between 28-32
            'localTablesVersionNumber': 0,
            'bufrHeaderCentre': 94, #originating centre 98=ECMWF, 94=DMI
            # 'bufrHeaderSubCentre': 0,
            'updateSequenceNumber': 0, #0 is original message, incremented by 1 for updates
            'dataCategory': 0, #surface data - land
            'internationalDataSubCategory': 0, #Hourly synoptic observations from fixed-land stations (SYNOP)
            # 'dataSubCategory': 0,
            'observedData': 1,
            'compressedData': 0,
        },
        'station': {
            'stationNumber': {
                # use int; fails with "Segmentation fault (core dumped)" if passed as string
                # This is the last three digits of the DMI Station ID, e.g. for "04401" use 401.
                # 'blockNumber' is used to register the first part of the ID ("04")
                'WEG_B': 460,
                'KAN_B': 445
                },
            'blockNumber': 4, #4 is Greenland, 6 is Denmark; not valid with synopMobil template
            # 'regionNumber': 6, #6 is Europe, 7 is MISSING VALUE; not valid with synopLand template
            'centre': 94, #94 is Copenhagen
            # 'agencyInChargeOfOperatingObservingPlatform': , #nothing for DMI or GEUS in code table
            # 'wmoRegionSubArea': 1,
            # 'stationOrSiteName': , #not valid with synopMobil template
            # 'shortStationName': , #not valid with synopMobil template
            # 'longStationName': , #not valid with synopMobil template
            'stationType': 0, #automatic station
            'instrumentationForWindMeasurement': 8, #certified instruments
            # 'stationElevationQualityMarkForMobileStations': 1, #Excellent - within 3m; not valid with synopLand template
        }
    },
}

'''
The following are not valid with either synopMobil or synopLand templates:
'measuringEquipmentType': 0, #Pressure instrument associated with wind-measuring equipment;
'temperatureObservationPrecision': 0.1, #Kelvin;
'pressureSensorType': 0, #capacitance aneroid;
'temperatureSensorType': 2, #capacitance bead;
'humiditySensorType': 4, #capacitance sensor;
'anemometerType': 1, #propeller rotor;
'methodOfPrecipitationMeasurement': 1, #tipping bucket method;
'''

# Optionally export to file on disk

# Export .json
# import json
# with open('ibufr_settings.json', 'w') as f:
# 	json.dump(ibufr_settings, f)

# Export .pickle
# import pickle
# with open('ibufr_settings.pickle', 'wb') as handle:
# 	pickle.dump(ibufr_settings, handle, protocol=pickle.HIGHEST_PROTOCOL)
