#!/usr/bin/env python

import xarray as xr
import pandas as pd
import datetime
import os, uuid, unittest
from datetime import timedelta

xr.set_options(keep_attrs=True)                                                #TODO remove this?

def addAllInfo(ds, v_file='variables.csv', m_file='metadata.csv'):
    '''Add variable attributes and metadata to dataset
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to add variable attributes to
    
    Returns
    -------
    ds : xarray.Dataset
        Dataset with metadata
    '''
    varcsv = os.path.join(os.path.dirname(__file__), v_file)                   #TODO move this to function input                  
    metcsv = os.path.join(os.path.dirname(__file__), m_file)                   
    ds = addVars(ds, varcsv)
    ds = addMeta(ds, metcsv)
    return ds

def mergeVars(ds_list, v_file='variables.csv', cols=['lo','hi','OOL']):        #TODO find way to make this one line as described
    '''Merge dataset by variable attributes from lookup table file. 

    Parameters
    ----------
    ds_list : list
        List of xarray.Dataset objects
    v_file : str, optional
        Variable look up table file path. The default is 'variables.csv'.
    vold : str, optional
        Variable column names to merge by. The default is ['lo','hi','OOL'].
        
    Returns
    -------
    ds : xarray.Dataset
        Dataset with merged attributes
    '''
    # This could be as simple as:
    # ds = xr.open_mfdataset(infile_list, combine='by_coords', mask_and_scale=False).load()     
    # Except that some files have overlapping times.
    
    # Combine Dataset objects
    ds = ds_list[0]
    if len(ds_list) > 1:
        for d in ds_list[1:]:
            ds = ds.combine_first(d)
    
    # Get variables
    df = getVars(v_file)[cols]
    df = df.dropna(how='all')
    
    # Merge where variable name matches
    for var in df.index:
        if var not in list(ds.variables): continue
        if var == 'rh_cor':
             ds[var] = ds[var].where(ds[var] >= df.loc[var, 'lo'], other = 0)
             ds[var] = ds[var].where(ds[var] <= df.loc[var, 'hi'], other = 100)
        else:
            ds[var] = ds[var].where(ds[var] >= df.loc[var, 'lo'])
            ds[var] = ds[var].where(ds[var] <= df.loc[var, 'hi'])
        other_vars = df.loc[var]['OOL'] # either NaN or "foo" or "foo bar baz ..."
        if isinstance(other_vars, str): 
            for o in other_vars.split():
                if o not in list(ds.variables): continue
                ds[o] = ds[o].where(ds[var] >= df.loc[var, 'lo'])
                ds[o] = ds[o].where(ds[var] <= df.loc[var, 'hi'])

    # Clean up metadata
    for k in ['format', 'hygroclip_t_offset', 'dsr_eng_coef', 'usr_eng_coef',
              'dlr_eng_coef', 'ulr_eng_coef', 'pt_z_coef', 'pt_z_p_coef',
              'pt_z_factor', 'pt_antifreeze', 'boom_azimuth', 'nodata',
              'conf', 'file']:
        if k in ds.attrs.keys():
            ds.attrs.pop(k)
    return ds

def addVars(ds, v_file):
    '''Add variable attributes from file to dataset
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to add variable attributes to
    v_file : str
        Variables lookup table file
    
    Returns
    -------
    ds : xarray.Dataset
        Dataset with metadata
   '''
    # load CSV to NetCDF lookup variable lookup table
    vf = getVars(v_file)
    
    # Add CF metdata
    for k in ds.keys():
        if k not in vf.index: continue
        ds[k].attrs['standard_name'] = vf.loc[k]['standard_name']
        ds[k].attrs['long_name'] = vf.loc[k]['long_name']
        ds[k].attrs['units'] = vf.loc[k]['units']
    return ds

def addMeta(ds, m_file):
    '''Add metadata attributes from file to dataset
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to add metadata attributes to
    m_file : str
        Metadata file
        
    Returns
    -------
    ds : xarray.Dataset
        Dataset with metadata
   '''    
    a = ds['gps_lon'].attrs
    ds['gps_lon'] = -1 * ds['gps_lon']
    ds['gps_lon'].attrs = a
    
    ds['lon'] = ds['gps_lon'].mean()
    ds['lon'].attrs = a
    
    ds['lat'] = ds['gps_lat'].mean()
    ds['lat'].attrs = ds['gps_lat'].attrs
    
    ds['alt'] = ds['gps_alt'].mean()
    ds['alt'].attrs = ds['gps_alt'].attrs
       
    # ds['station_name'] = (('name_strlen'), [fname.split('hour')[0].split('/')[2][:-1]])
    # # ds['station_name'].attrs['long_name'] = 'station name'
    # ds['station_name'].attrs['cf_role'] = 'timeseries_id'

    for k in ds.keys(): # for each var
        if 'units' in ds[k].attrs:        
            if ds[k].attrs['units'] == 'C':
                ds[k].attrs['units'] = 'degrees_C'

    # https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3#geospatial_bounds
    # highly recommended
    ds.attrs['id'] = 'dk.geus.promice:' + str(uuid.uuid3(uuid.NAMESPACE_DNS, ds.attrs['station_id']))
    ds.attrs['history'] = 'Generated on ' + datetime.datetime.utcnow().isoformat()
    ds.attrs['date_created'] = str(datetime.datetime.now().isoformat())
    ds.attrs['date_modified'] = ds.attrs['date_created']
    ds.attrs['date_issued'] = ds.attrs['date_created']
    ds.attrs['date_metadata_modified'] = ds.attrs['date_created'] 
    
    ds.attrs['geospatial_bounds'] = "POLYGON((" + \
        f"{ds['lat'].min().values} {ds['lon'].min().values}, " + \
        f"{ds['lat'].min().values} {ds['lon'].max().values}, " + \
        f"{ds['lat'].max().values} {ds['lon'].max().values}, " + \
        f"{ds['lat'].max().values} {ds['lon'].min().values}, " + \
        f"{ds['lat'].min().values} {ds['lon'].min().values}))"

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
        
    ds.time.encoding["dtype"] = "int32" # CF standard requires time as int not int64
    # ds['time'].encoding['units'] = 'hours since 2016-05-01 00:00:00'
    # ds['time'] = ds['time'].astype('datetime64[D]')
    
    # Load metadata attributes and add to Dataset   
    meta = getMeta(m_file)
    [_addAttr(ds, key, value) for key,value in meta.items()]
    return ds

def getVars(v_file):
   '''Load variables.csv file
   
   Parameters
   ----------
   v_file : str
       Variable lookup table file path

   Returns
   -------
   pandas.DataFrame
       Variables dataframe
   '''
   return pd.read_csv(v_file, index_col=0, comment="#")

def getMeta(m_file, delimiter=','):                                            #TODO change to DataFrame output to match variables.csv
    '''Load metadata table
    
    Parameters
    ----------
    v_file : str
        Metadata file path

    Returns
    -------
    meta : dict
        Metadata dictionary
    '''
    meta={}
    with open(m_file, 'r') as f:
        lines = f.readlines()
    for l in lines[1:]:
        meta[l.split(',')[0]] = l.split(delimiter)[1].split('\n')[0].replace(';',',')        
    return meta

def _addAttr(ds, key, value):
    '''Add attribute to xarray dataset'''
    if len(key.split('.')) == 2:
        ds[key.split('.')[0]].attrs[key.split('.')[1]] = value
    else:
        ds.attrs[key] = value    

#------------------------------------------------------------------------------
        
class TestMeta(unittest.TestCase): 

    def testgetVars(self):
        '''Test variable table lookup retrieval'''
        v = getVars('variables.csv')
        self.assertIsInstance(v, pd.DataFrame)
        self.assertTrue(v.columns[0] in 'standard_name')
        self.assertTrue(v.columns[2] in 'units')
        
    def testgetMeta(self):  
        '''Test AWS names retrieval'''
        m = getMeta('metadata.csv')
        self.assertIsInstance(m, dict)
        self.assertTrue('references' in m)
    
    def testAddAll(self):
        '''Test variable and metadata attributes added to Dataset'''
        d = xr.Dataset()
    
        v = getVars('variables.csv')    
        att = list(v.index)
        att1 = ['gps_lon', 'gps_lat', 'gps_alt', 'albedo', 'p']
        for a in att:
            d[a]=[0,1]
        for a in att1:
            d[a]=[0,1]
        d['time'] = [datetime.datetime.now(), 
                     datetime.datetime.now()-timedelta(days=365)]
        d.attrs['station_id']='TEST'
        
        ds = addAllInfo(d, v_file='variables.csv', m_file='metadata.csv')
        self.assertTrue(ds.attrs['station_id']=='TEST')
        self.assertIsInstance(ds.attrs['references'], str)
    
    def testMerge(self):                                                       #TODO
        pass
              
if __name__ == "__main__": 
    unittest.main() 