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
