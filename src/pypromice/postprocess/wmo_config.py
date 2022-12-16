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
    'discontinued': ['CEN1','TAS_U','QAS_A','NUK_N','THU_U','JAR','SWC'],
    'no_instantaneous': ['ZAK_L','ZAK_U','KAN_B'], # currently not transmitting instantaneous values
    'suspect_data': [], # instantaneous data is suspect
    'use_v3': ['KPC_L','NUK_U','ZAK_U'], # use v3 versions instead (but registered IDs will be non-v3)
    'v3_bad': ['KPC_Uv3','QAS_Uv3','QAS_Lv3'] # KPC_Uv3 years are 2056, QAS_Uv3 stops at 2022-10-31, QAS_Lv3 new ablation sensor w/different txt fields?
}
# NOTE: Use both THU_L and THU_L2; use ONLY THU_U2, but register it as THU_U
# NOTE: Use JAR_O and SWC_O, but register them as JAR and SWC
# NOTE: CEN2 data is registered as CEN

ibufr_settings = {
    'template': {
        'unexpandedDescriptors': {
            'mobile': (307090), #message template, "synopMobil"
            'land': (307080), #message template, "synopLand"
        },
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
        'station_identifiers': {
            'shipOrMobileLandStationIdentifier': {
                # use string; synopMobil fails with "gribapi.errors.InvalidArgumentError: Invalid argument" if passed as int
                'CEN2':   '04411',
                'CP1':    '04411',
                'DY2':    '04411',
                'EGP':    '04411',
                'HUM':    '04411',
                'JAR':    '04411',
                'JAR_O':  '04411',
                'KAN_L':  '04411',
                'KAN_M':  '04411',
                'KAN_U':  '04411',
                'KPC_L':  '04411',
                'KPC_U':  '04411',
                'LYN_L':  '04411',
                'LYN_T':  '04411',
                'MIT':    '04411',
                'NAE':    '04411',
                'NAU':    '04411',
                'NEM':    '04411',
                'NSE':    '04411',
                'NUK_K':  '04411',
                'NUK_L':  '04411',
                'NUK_U':  '04411',
                'QAS_L':  '04411',
                'QAS_M':  '04411',
                'QAS_U':  '04411',
                'SCO_L':  '04411',
                'SCO_U':  '04411',
                'SDL':    '04411',
                'SDM':    '04411',
                'SWC':    '04411',
                'SWC_O':  '04411',
                'TAS_A':  '04411',
                'TAS_L':  '04411',
                'THU_L':  '04411',
                'THU_L2': '04411',
                'THU_U':  '04411',
                'TUN':    '04411',
                'UPE_L':  '04411',
                'UPE_U':  '04411',
                'UWN':    '04411',
                'ZAK_L':  '04411',
                'ZAK_U':  '04411'
            },
            'stationNumber': {
                # land-based (non-mobile) stations
                # use int; synopLand fails with "Segmentation fault (core dumped)" if stationNumber is passed as string
                # stationNumber cannot handle more than 3 characters?!
                'WEG_B':  '04411',
                'KAN_B':  '04411'
            },
        },
        # 'blockNumber': 4, #4 is Greenland, 6 is Denmark; not valid if using synopMobil template
        'regionNumber': 6, #6 is Europe, 7 is MISSING VALUE
        'centre': 94, #94 is Copenhagen
        # 'agencyInChargeOfOperatingObservingPlatform': , #nothing for DMI or GEUS in code table
        # 'wmoRegionSubArea': 1,
        # 'stationOrSiteName': , #not valid if using synopMobil template
        # 'shortStationName': , #not valid if using synopMobil template
        # 'longStationName': , #not valid if using synopMobil template
        # 'directionOfMotionOfMovingObservingPlatform': ,
        # 'movingObservingPlatformSpeed': ,
        'stationType': 0, #automatic station
        'instrumentationForWindMeasurement': 8, #certified instruments
        # 'measuringEquipmentType': 0, #Pressure instrument associated with wind-measuring equipment; not valid if using synopMobil template
        # 'temperatureObservationPrecision': 0.1, #Kelvin; not valid if using synopMobil template
        # 'pressureSensorType': 0, #capacitance aneroid; not valid if using synopMobil template
        # 'temperatureSensorType': 2, #capacitance bead; not valid if using synopMobil template
        # 'humiditySensorType': 4, #capacitance sensor; not valid if using synopMobil template
        # 'anemometerType': 1, #propeller rotor; not valid if using synopMobil template
        # 'methodOfPrecipitationMeasurement': 1, #tipping bucket method; not valid if using synopMobil template
        'stationElevationQualityMarkForMobileStations': 1, #Excellent - within 3m
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