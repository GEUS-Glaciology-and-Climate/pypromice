#!/usr/bin/env python3
'''
Config file to store wmo-related reference objects
Imported by bufr_utilities.py
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
# NOTE: Use both THU_L and THU_L2; use ONLY THU_U2, but register it as THU_U (this is dealt with in bufr_utilities.py)
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
