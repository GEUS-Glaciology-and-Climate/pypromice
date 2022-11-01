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
ibufr_settings = {
    'template': {
        'unexpandedDescriptors': (307090), # message template, 307090 is "synopMobil"
        'edition': 4, # latest edition
        'masterTableNumber': 0,
        'masterTablesVersionNumber': 32, # DMI recommends any table version between 28-32
        'localTablesVersionNumber': 0,
        'bufrHeaderCentre': 94, # originating centre 98=ECMWF, 94=DMI
        # 'bufrHeaderSubCentre': 0,
        'updateSequenceNumber': 0, # 0 is original message, incremented by 1 for updates
        'dataCategory': 0, # surface data - land
        'internationalDataSubCategory': 3, # hourly synoptic observations from mobile-land stations (SYNOP MOBIL)
        # 'dataSubCategory': 0,
        'observedData': 1,
        'compressedData': 0,
    },
    'station': {
        'shipOrMobileLandStationIdentifier': {
            'CEN_A': 4204207,
            'CEN_T': 4204208,
            'CEN_i': 4204229, #CEN1 & CEN2?
            'DIS_A': 4204209,
            'EGP_A': 4204210,
            'EGP_i': 4204210,
            'GEUS_': 4204211,
            'KAN_B': 4204213,
            'KAN_L': 4204214,
            'KAN_M': 4204215,
            'KAN_U': 4204216,
            'KPC_L': 4204217,
            'KPC_U': 4204218,
            'MIT_A': 4204212,
            'MIT_i': 4204219,
            'NUK_K': 4204228,
            'NUK_L': 4204204,
            'NUK_U': 4204206,
            'QAS_A': 4204220,
            'QAS_L': 4204201,
            'QAS_M': 4204221,
            'QAS_U': 4204203,
            'SCO_L': 4204222,
            'SCO_U': 4204223,
            'TAS_A': 4204205,
            'TAS_L': 4204202,
            'THU_L': 4204224,
            'THU_U': 4204225,
            'THU_U2': 4204230,
            'UPE_L': 4204226,
            'UPE_U': 4204227
        },
        # 'blockNumber': 4, # 4 is Greenland, 6 is Denmark; not valid if using synopMobil template
        'regionNumber': 6, # 6 is Europe, 7 is MISSING VALUE
        'centre': 94, # 94 is Copenhagen
        # 'agencyInChargeOfOperatingObservingPlatform': , #CODE TABLE?
        # 'wmoRegionSubArea': 1,
        # 'stationOrSiteName': , # not valid if using synopMobil template
        # 'shortStationName': , # not valid if using synopMobil template
        # 'longStationName': , # not valid if using synopMobil template
        # 'directionOfMotionOfMovingObservingPlatform': ,
        # 'movingObservingPlatformSpeed': ,
        'stationType': 0, # automatic station
        # 'instrumentationForWindMeasurement': 6, # Unclear in docs how to set this
        # 'measuringEquipmentType': 0, # Pressure instrument associated with wind-measuring equipment; not valid if using synopMobil template
        # 'temperatureObservationPrecision': 0.1, # Kelvin; not valid if using synopMobil template
        # 'pressureSensorType': 0, # capacitance aneroid; not valid if using synopMobil template
        # 'temperatureSensorType': 2, # capacitance bead; not valid if using synopMobil template
        # 'humiditySensorType': 4, # capacitance sensor; not valid if using synopMobil template
        # 'anemometerType': 1, # propeller rotor; not valid if using synopMobil template
        # 'methodOfPrecipitationMeasurement': 1, # tipping bucket method; not valid if using synopMobil template
    }
}

# Optionally export to file on disk

# Export .json
# import json
# with open('ibufr_settings.json', 'w') as f:
# 	json.dump(ibufr_settings, f)

# Export .pickle
# import pickle
# with open('ibufr_settings.pickle', 'wb') as handle:
# 	pickle.dump(ibufr_settings, handle, protocol=pickle.HIGHEST_PROTOCOL)