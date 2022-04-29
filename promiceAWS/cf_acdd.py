#!/usr/bin/env python

import numpy as np
import xarray as xr
import pandas as pd
import datetime

xr.set_options(keep_attrs=True)

def cf_and_acdd(ds):
    # ds.attrs = attrs
    # load CSV to NetCDF lookup variable lookup table
    vf = pd.read_csv('variables.csv', index_col=0)
    # print(v.columns)
    # # return ds
    # vf = v.set_index('field')

    ds.attrs["featureType"] = "timeSeries"
    
    # ds['time'].encoding['units'] = 'hours since 2016-05-01 00:00:00'
    # ds['time'] = ds['time'].astype('datetime64[D]')

    # Add CF metdata
    for k in ds.keys():
        if k not in vf.index: continue
        ds[k].attrs['standard_name'] = vf.loc[k]['standard_name']
        ds[k].attrs['long_name'] = vf.loc[k]['long_name']
        ds[k].attrs['units'] = vf.loc[k]['units']
    
    # Also add metadata for 'time' variable'
    ds['time'].attrs['standard_name'] = 'time'
    ds['time'].attrs['long_name'] = 'time'
    
    a = ds['gps_lon'].attrs
    ds['gps_lon'] = -1 * ds['gps_lon']
    ds['gps_lon'].attrs = a
    ds['gps_lon'].attrs['units'] = 'degrees_east'
    
    ds['lon'] = ds['gps_lon'].mean()
    ds['lon'].attrs = a
    ds['lon'].attrs['units'] = 'degrees_east'
    
    ds['lat'] = ds['gps_lat'].mean()
    ds['lat'].attrs = ds['gps_lat'].attrs
    
    ds['alt'] = ds['gps_alt'].mean()
    ds['alt'].attrs = ds['gps_alt'].attrs
    ds['alt'].attrs['positive'] = 'up'
    ds['gps_alt'].attrs['positive'] = 'up'
    
    # ds = ds.drop(['gps_lon','gps_lat','gps_alt'])
    
    # ds['station_name'] = (('name_strlen'), [fname.split('hour')[0].split('/')[2][:-1]])
    # # ds['station_name'].attrs['long_name'] = 'station name'
    # ds['station_name'].attrs['cf_role'] = 'timeseries_id'
    
    ds['albedo'].attrs['units'] = '-'
    # for k in ds.keys(): # for each var
    #     if 'units' in ds[k].attrs:        
    #         if ds[k].attrs['units'] == 'C':
    #             attrs = ds[k].attrs
    #             ds[k] = ds[k] - 273.15
    #             attrs['units'] = 'K'
    #             ds[k].attrs = attrs
    for k in ds.keys(): # for each var
        if 'units' in ds[k].attrs:        
            if ds[k].attrs['units'] == 'C':
                ds[k].attrs['units'] = 'degrees_C'
    
    # https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3#geospatial_bounds
    # highly recommended
    ds.attrs['title'] = 'PROMICE AWS data (' + ds.attrs['station_id'] + ')'
    
    ds.attrs['summary'] = 'The Programme for Monitoring of the Greenland Ice Sheet (PROMICE) has been measuring climate and ice sheet properties since 2007. Currently the PROMICE automatic weather station network includes 25 instrumented sites in Greenland. Accurate measurements of the surface and near-surface atmospheric conditions in a changing climate is important for reliable present and future assessment of changes to the Greenland ice sheet. Here we present the PROMICE vision, methodology, and each link in the production chain for obtaining and sharing quality-checked data. In this paper we mainly focus on the critical components for calculating the surface energy balance and surface mass balance. A user-contributable dynamic webbased database of known data quality issues is associated with the data products at (https://github.com/GEUS-PROMICE/ PROMICE-AWS-data-issues/). As part of the living data option, the datasets presented and described here are available at DOI: 10.22008/promice/data/aws, https://doi.org/10.22008/promice/data/aws'
    
    kw = ['GCMDSK:EARTH SCIENCE > CRYOSPHERE > GLACIERS/ICE SHEETS > ICE SHEETS > ICE SHEET MEASUREMENTS',
          'GCMDSK:EARTH SCIENCE > CRYOSPHERE > GLACIERS/ICE SHEETS > GLACIER MASS BALANCE/ICE SHEET MASS BALANCE',
          'GCMDSK:EARTH SCIENCE > CRYOSPHERE > SNOW/ICE > SNOW/ICE TEMPERATURE',
          'GCMDSK:EARTH SCIENCE > CRYOSPHERE > SNOW/ICE',
          'GCMDSK:EARTH SCIENCE > CRYOSPHERE > SNOW/ICE > SNOW MELT',
          'GCMDSK:EARTH SCIENCE > CRYOSPHERE > SNOW/ICE > SNOW DEPTH',
          'GCMDSK:EARTH SCIENCE > CRYOSPHERE > SNOW/ICE > ICE VELOCITY',
          'GCMDSK:EARTH SCIENCE > CRYOSPHERE > SNOW/ICE > ALBEDO',
          'GCMDSK:EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SNOW/ICE > ALBEDO',
          'GCMDSK:EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SNOW/ICE > ICE GROWTH/MELT',
          'GCMDSK:EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SNOW/ICE > ICE VELOCITY',
          'GCMDSK:EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SNOW/ICE > SNOW DEPTH',
          'GCMDSK:EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SNOW/ICE > SNOW MELT',
          'GCMDSK:EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SNOW/ICE > SNOW/ICE TEMPERATURE',
          'GCMDSK:EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SNOW/ICE',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC PRESSURE',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > ALBEDO',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > INCOMING SOLAR RADIATION',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > LONGWAVE RADIATION > DOWNWELLING LONGWAVE RADIATION',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > LONGWAVE RADIATION > UPWELLING LONGWAVE RADIATION',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > LONGWAVE RADIATION',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > NET RADIATION',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > OUTGOING LONGWAVE RADIATION',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > RADIATIVE FLUX',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > RADIATIVE FORCING',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > SHORTWAVE RADIATION > DOWNWELLING SHORTWAVE RADIATION',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > SHORTWAVE RADIATION',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION > SUNSHINE',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC RADIATION',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC TEMPERATURE > SURFACE TEMPERATURE > AIR TEMPERATURE',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC WATER VAPOR > WATER VAPOR INDICATORS > HUMIDITY > ABSOLUTE HUMIDITY',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC WATER VAPOR > WATER VAPOR INDICATORS > HUMIDITY > RELATIVE HUMIDITY',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC WINDS > LOCAL WINDS',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC WINDS > SURFACE WINDS > U/V WIND COMPONENTS',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC WINDS > SURFACE WINDS > WIND DIRECTION',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC WINDS > SURFACE WINDS > WIND SPEED',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC WINDS > SURFACE WINDS',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > CLOUDS',
          'GCMDSK:EARTH SCIENCE > ATMOSPHERE > PRECIPITATION']
    
    ds.attrs['keywords'] = ', '.join(kw)

    ds.attrs['Conventions'] = 'ACDD-1.3, CF-1.7'
    
    # recommended
    import uuid
    ds.attrs['id'] = 'dk.geus.promice:' + str(uuid.uuid3(uuid.NAMESPACE_DNS, ds.attrs['station_id']))
    ds.attrs['naming_authority'] = 'dk.geus.promice'
    ds.attrs['history'] = 'Generated on ' + datetime.datetime.utcnow().isoformat()
    ds.attrs['source'] = 'PROMICE AWS processing scripts'
    ds.attrs['processing_level'] = 'Level 3'
    ds.attrs['acknowledgement'] = 'The Programme for Monitoring of the Greenland Ice Sheet (PROMICE)'
    ds.attrs['license'] = 'Creative Commons Attribution 4.0 International (CC-BY-4.0) https://creativecommons.org/licenses/by/4.0'
    ds.attrs['standard_name_vocabulary'] = 'CF Standard Name Table (v77, 19 January 2021)'
    ds.attrs['date_created'] = str(datetime.datetime.now().isoformat())
    ds.attrs['creator_name'] = 'Ken Mankoff'
    ds.attrs['creator_email'] = 'kdm@geus'
    ds.attrs['creator_url'] = 'http://promice.org'
    ds.attrs['institution'] = 'GEUS'
    ds.attrs['publisher_name'] = 'GEUS'
    ds.attrs['publisher_email'] = 'info@promice.dk'
    ds.attrs['publisher_url'] = 'http://promice.dk'
    
    ds.attrs['geospatial_bounds'] = "POLYGON((" + \
        f"{ds['lat'].min().values} {ds['lon'].min().values}, " + \
        f"{ds['lat'].min().values} {ds['lon'].max().values}, " + \
        f"{ds['lat'].max().values} {ds['lon'].max().values}, " + \
        f"{ds['lat'].max().values} {ds['lon'].min().values}, " + \
        f"{ds['lat'].min().values} {ds['lon'].min().values}))"
    ds.attrs['geospatial_bounds_crs'] = 'EPSG:4326'
    ds.attrs['geospatial_bounds_vertical_crs'] = 'EPSG:4979'
    ds.attrs['geospatial_lat_min'] = ds['lat'].min().values
    ds.attrs['geospatial_lat_max'] = ds['lat'].max().values
    ds.attrs['geospatial_lon_min'] = ds['lon'].min().values
    ds.attrs['geospatial_lon_max'] = ds['lon'].max().values
    ds.attrs['geospatial_vertical_min'] = ds['alt'].min().values
    ds.attrs['geospatial_vertical_max'] = ds['alt'].max().values
    ds.attrs['geospatial_vertical_positive'] = 'up'
    ds.attrs['time_coverage_start'] = str(ds['time'][0].values)
    ds.attrs['time_coverage_end'] = str(ds['time'][-1].values)
    # https://www.digi.com/resources/documentation/digidocs/90001437-13/reference/r_iso_8601_duration_format.htm
    ds.attrs['time_coverage_duration'] = pd.Timedelta((ds['time'][-1] - ds['time'][0]).values).isoformat()
    ds.attrs['time_coverage_resolution'] = pd.Timedelta((ds['time'][1] - ds['time'][0]).values).isoformat()
    
    # suggested
    ds.attrs['creator_type'] = 'person'
    ds.attrs['creator_institution'] = 'GEUS'
    ds.attrs['publisher_type'] = 'institution'
    ds.attrs['publisher_institution'] = 'GEUS'
    ds.attrs['program'] = 'PROMICE'
    ds.attrs['contributor_name'] = ''
    ds.attrs['contributor_role'] = ''
    ds.attrs['geospatial_lat_units'] = 'degrees_north'
    # ds.attrs['geospatial_lat_resolution'] = ''
    ds.attrs['geospatial_lon_units'] = 'degrees_east'
    # ds.attrs['geospatial_lon_resolution'] = ''
    ds.attrs['geospatial_vertical_units'] = 'EPSG:4979 CHECKME'
    # ds.attrs['geospatial_vertical_resolution'] = ''
    # ds.attrs['date_modified'] = ds.attrs['date_created']
    # ds.attrs['date_issued'] = ds.attrs['date_created']
    # ds.attrs['date_metadata_modified'] = ''
    ds.attrs['product_version'] = 3
    ds.attrs['keywords_vocabulary'] = 'GCMDSK:GCMD Science Keywords:https://gcmd.earthdata.nasa.gov/kms/concepts/concept_scheme/sciencekeywords'
    # ds.attrs['platform'] = ''
    # ds.attrs['platform_vocabulary'] = 'GCMD:GCMD Keywords'
    ds.attrs['instrument'] = 'See https://doi.org/10.5194/essd-13-3819-2021'
    # ds.attrs['instrument_vocabulary'] = 'GCMD:GCMD Keywords'
    # ds.attrs['cdm_data_type'] = ''
    # ds.attrs['metadata_link'] = ''
    ds.attrs['references'] = 'Fausto, R. S., van As, D., Mankoff, K. D., Vandecrux, B., Citterio, M., Ahlstrom, A. P., Andersen, S. B., Colgan, W., Karlsson, N. B., Kjeldsen, K. K., Korsgaard, N. J., Larsen, S. H., Nielsen, S., Pedersen, A. O., Shields, C. L., Solgaard, A. M., and Box, J. E.: Programme for Monitoring of the Greenland Ice Sheet (PROMICE) automatic weather station data, Earth Syst. Sci. Data, 13, 3819–3845, https://doi.org/10.5194/essd-13-3819-2021, 2021.'
    
    
    bib = """@article{Fausto2021,
    doi = {10.5194/essd-13-3819-2021},
    url = {https://doi.org/10.5194/essd-13-3819-2021},
    year = {2021},
    month = 8,
    publisher = {Copernicus {GmbH}},
    volume = {13},
    number = {8},
    pages = {3819--3845},
    author = {Robert S. Fausto and Dirk van As and Kenneth D. Mankoff and Baptiste Vandecrux and Michele Citterio and Andreas P. Ahlstr{\o}m and Signe B. Andersen and William Colgan and Nanna B. Karlsson and Kristian K. Kjeldsen and Niels J. Korsgaard and Signe H. Larsen and S{\o}ren Nielsen and Allan {\O}. Pedersen and Christopher L. Shields and Anne M. Solgaard and Jason E. Box},
    title = {Programme for Monitoring of the Greenland Ice Sheet ({PROMICE}) automatic weather station data},
    journal = {Earth System Science Data}
    }"""
    ds.attrs['references_bib'] = bib
    
    
    ds.attrs['comment'] = 'Data source: https://doi.org/10.22008/promice/data/aws'
    
    # ds.attrs['geospatial_lat_extents_match'] = 'gps_lat'
    # ds.attrs['geospatial_lon_extents_match'] = 'gps_lon'
    
    # from shapely.geometry import Polygon
    # geom = Polygon(zip(ds['lat'].values, ds['lon'].values))
    # # print(geom.bounds)
    # ds.attrs['geospatial_bounds'] = geom.bounds
    
    ds.attrs['project'] = 'PROMICE'
    
    for vv in ['p', 't_1', 't_2', 'rh', 'rh_cor', 'wspd', 'wdir', 'z_boom', 'z_stake', 'z_pt',
               't_i_1', 't_i_2', 't_i_3', 't_i_4', 't_i_5', 't_i_6', 't_i_7', 't_i_8',
               'tilt_x', 'tilt_y', 't_log']:
        ds[vv].attrs['coverage_content_type'] = 'physicalMeasurement'
        ds[vv].attrs['coordinates'] = "time lat lon alt"
    
    for vv in ['dshf', 'dlhf', 'dsr', 'dsr_cor', 'usr', 'usr_cor',
               'albedo', 'dlr', 'ulr', 'cc', 't_surf']:
        ds[vv].attrs['coverage_content_type'] = 'modelResult'
        ds[vv].attrs['coordinates'] = "time lat lon alt"
    
    for vv in ['fan_dc', 'batt_v']:
        ds[vv].attrs['coverage_content_type'] = 'auxiliaryInformation'
        ds[vv].attrs['coordinates'] = "time lat lon alt"
    
    for vv in ['gps_hdop']:
        ds[vv].attrs['coverage_content_type'] = 'qualityInformation'
        ds[vv].attrs['coordinates'] = "time lat lon alt"
    
    for vv in ['gps_time', 'lon', 'lat', 'alt']:
        ds[vv].attrs['coverage_content_type'] = 'coordinate'
    
    
    ds['lon'].attrs['long_name'] = 'station longitude'
    ds['lat'].attrs['long_name'] = 'station latitude'
    ds['alt'].attrs['long_name'] = 'station altitude'
    
    ds['lon'].attrs['axis'] = 'X'
    ds['lat'].attrs['axis'] = 'Y'
    ds['alt'].attrs['axis'] = 'Z'
    
    for vv in ['lon', 'lat', 'alt']:
        ds[vv].attrs['coverage_content_type'] = 'coordinate'
    
    # for vv in []: ds[vv].attrs['coverage_content_type'] = 'referenceInformation'
    
    ds.time.encoding["dtype"] = "int32" # CF standard requires time as int not int64
    return ds
