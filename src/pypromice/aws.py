#!/usr/bin/env python
"""
pypromice AWS processing module
"""
import os, unittest, toml, datetime, uuid, glob 
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta

try:
    from L0toL1 import toL1
    from L1toL2 import toL2
    from L2toL3 import toL3
except:
    from pypromice.L0toL1 import toL1
    from pypromice.L1toL2 import toL2
    from pypromice.L2toL3 import toL3

pd.set_option('display.precision', 2)
xr.set_options(keep_attrs=True)

#------------------------------------------------------------------------------

class AWS(object):
    '''AWS object to load and process PROMICE AWS data'''
    
    def __init__(self, config_file, inpath, outpath=None, 
                 var_file='./variables.csv', meta_file='./metadata.csv'):
        '''Object initialisation

        Parameters
        ----------
        config_file : str
            Configuration file path
        inpath : str
            Input file path
        outpath : str, optional
            Output file path. The default is None.
        '''
        assert(os.path.isfile(config_file))
        assert(os.path.isdir(inpath))
        print('\nAWS object initialising...')
        
        # Load config, variables CSF standards, and L0 files
        self.config = self.loadConfig(config_file, inpath)
        self.vars = getVars(var_file)
        self.meta = getMeta(meta_file)
        
        L0 = self.loadL0(config_file, inpath)
        self.L0=[]
        for l in L0:
            n = getColNames(self.vars, l.attrs['number_of_booms'], l.attrs['format'])
            self.L0.append(popCols(l, n))
        try:
            print(f'Processing data from {self.L0.attrs["station_id"]}...') 
        except:
            print(f'Processing data from {self.L0[0].attrs["station_id"]}...')         
        
        # Proces L0 to L3 product
        self.process()
        self.L3 = self.addAttributes(self.L3)
        
        # Resample to hourly, daily and monthly products
        self.L3_h = self.resample('60min')
        self.L3_d = self.resample('1D')
        self.L3_m = self.resample('M')
        
        # Round all values to specified decimals places
        self.L3_h = roundValues(self.L3_h, self.vars)
        self.L3_d = roundValues(self.L3_d, self.vars)
        self.L3_m = roundValues(self.L3_m, self.vars)
        
        # Save to file if outpath given
        if outpath is not None:
            if os.path.isdir(outpath):
                self.write(outpath)
            else:
                print(f'Outpath f{outpath} does not exist. Unable to save to file')
                pass
    
    def process(self):
        '''Perform L0 to L3 data processing'''
        try:
            print(f'Commencing {self.L0.attrs["number_of_booms"]}-boom processing...')
        except:
            print(f'Commencing {self.L0[0].attrs["number_of_booms"]}-boom processing...')        
        
        print('Level 1 processing...')
        self.L0 = [addBasicMeta(item, self.vars) for item in self.L0]
        self.L1 = [toL1(item) for item in self.L0]
        self.L1A = mergeVars(self.L1, self.vars)
        
        # L1 to L2 processing
        print('Level 2 processing...')
        self.L2 = toL2(self.L1A)
        self.L2 = clipValues(self.L2, self.vars)

        # L2 to L3 processing
        print('Level 3 processing...')        
        self.L3 = toL3(self.L2)
        
    def resample(self, resample_factor):  
        '''Resample L3 data'''
        r = resampleL3(self.L3, resample_factor)
        if resample_factor in 'M':
            print('Level 3 successfully resampled to monthly product')
        elif resample_factor in '1D':
            print('Level 3 successfully resampled to daily product')           
        elif resample_factor in '60min':
            print('Level 3 successfully resampled to hourly product')  
        else:
            print('Level 3 successfully resampled')
        return r
    
    def addAttributes(self, L3):
        '''Add variable and attribute metadata'''
        L3 = addVars(L3, self.vars)
        L3 = addMeta(L3, self.meta)
        return L3

    def write(self, outpath):
        '''Write L3 data to .nc and .csv hourly and daily files'''
        outdir = os.path.join(outpath, self.L3_h.attrs['station_id']) 
        
        f = [l.attrs['format'] for l in self.L0]
        if all(f):
            col_names = getColNames(self.vars, self.L3_h.attrs['number_of_booms'], 
                                    self.L0[0].attrs['format'])
        else:
            col_names = getColNames(self.vars, self.L3_h.attrs['number_of_booms'], 
                                    None)
        
        print('Writing to files...')
        writeL3(outdir, self.L3_h.attrs['station_id'], 
                self.L3_h, self.L3_d, self.L3_m, col_names)
        print(f'Out files successfully written to {outdir}')

    def loadConfig(self, config_file, inpath):
        '''Load configuration from .toml file'''
        conf = getConfig(config_file, inpath)
        return conf
        
    def loadL0(self, conf, L0_path):
        '''Load PROMICE AWS level 0 (L0) data from associated TOML-formatted 
        config file and L0 data file'''
        c = self.config
        if len(c.keys()) == 1: # one file in this config
            ds = self._readL0file(c[list(c.keys())[0]])
            print(f'L0 data successfully loaded from {list(c.keys())[0]}')
            return [ds]
        else:
            ds_list = []
            for k in c.keys():
                ds_list.append(self._readL0file(c[k]))
                print(f'L0 data successfully loaded from {k}')
            return ds_list    
        
    def _readL0file(self, conf):
        ''' Read L0 .txt file to Dataset object using config dictionary and
        populate with initial metadata'''
        file_version = conf.get('file_version', -1)  
        ds = getL0(conf['file'], conf['nodata'], conf['columns'], 
                   conf["skiprows"], file_version)
        ds = populateMeta(ds, conf, ["columns", "skiprows", "modem"])
        return ds

#------------------------------------------------------------------------------

def getConfig(config_file, inpath):
    '''Load configuration from .toml file. PROMICE .toml files support defining 
    features at the top level which apply to all nested properties, but do not 
    overwrite nested properties if they are defined
    
    Parameters
    ----------
    config_file : str
        TOML file path
    
    Returns
    -------
    conf : dict
        Configuration dictionary
    '''
    conf = toml.load(config_file)                                              # Move all top level keys to nested properties,
    top = [_ for _ in conf.keys() if not type(conf[_]) is dict]                # if they are not already defined in the nested properties
    subs = [_ for _ in conf.keys() if type(conf[_]) is dict]                   # Insert the section name (config_file) as a file property and config file
    for s in subs:
        for t in top:
            if t not in conf[s].keys():
                conf[s][t] = conf[t]

        conf[s]['conf'] = config_file
        conf[s]['file'] = os.path.join(inpath, s)

    for t in top: conf.pop(t)                                                  # Delete all top level keys beause each file
                                                                               # should carry all properties with it
    for k in conf.keys():                                                      # Check required fields are present
        print(k)
        for field in ["columns", "station_id", "format", "skiprows"]:
            assert(field in conf[k].keys())
    return conf

def getL0(infile, nodata, cols, skiprows, file_version, 
          delimiter=',', comment='#'):
    ''' Read L0 data file into pandas DataFrame object
    
    Parameters
    ----------
    infile : str
        L0 file path
    nodata : list
        List containing value for nan values and reassigned value
    cols : list
        List of columns in file
    skiprows : int
        Skip rows value
    file_version : int
        Version of L0 file
    delimiter : str
        String delimiter for L0 file
    comment : str
        Notifier of commented sections in L0 file
    
    Returns
    -------
    ds : xarray.Dataset
        L0 Dataset
    '''    
    if file_version == 1:        
        df = pd.read_csv(infile, comment=comment, index_col=0, 
                         na_values=nodata, names=cols, 
                         parse_dates={'time': ['year', 'doy', 'hhmm']}, 
                         date_parser=_getDateParserV1, sep=delimiter,
                         skiprows=skiprows, skip_blank_lines=True,
                         usecols=range(len(cols)))
    else:
        df = pd.read_csv(infile, comment=comment, index_col=0,
                         na_values=nodata, names=cols, parse_dates=True,
                         sep=delimiter, skiprows=skiprows,
                         skip_blank_lines=True, usecols=range(len(cols)))        
    
    # Drop SKIP columns
    for c in df.columns:
        if c[0:4] == 'SKIP':
            df.drop(columns=c, inplace=True)

    # Carry relevant metadata with ds
    ds = xr.Dataset.from_dataframe(df)    
    return ds

def addBasicMeta( ds, vars_df):
    ''' Use a variable lookup table DataFrame to add the basic metadata 
    to the xarray dataset. This is later amended to finalise L3'''
    for v in vars_df.index:
        if v == 'time': continue # coordinate variable, not normal var
        if v not in list(ds.variables): continue
        for c in ['standard_name', 'long_name', 'units']:
            if isinstance(vars_df[c][v], float) and np.isnan(vars_df[c][v]): continue
            ds[v].attrs[c] = vars_df[c][v]
    return ds

def populateMeta(ds, conf, skip):
    '''Populate L0 Dataset with metadata dictionary
    
    Parameters
    ----------
    ds : xarray.Dataset
        L0 dataset
    conf : dict
        Metadata dictionary
    skip : list
        List of column names to skip parsing to metadata 
    
    Returns
    -------
    ds : xarray.Dataset
        L0 dataset with metadata populated as Dataset attributes
    '''
    meta = {}
    # skip = ["columns", "skiprows"]
    for k in conf.keys():
        if k not in skip: meta[k] = conf[k]
    ds.attrs = meta
    return ds

def writeCSV(outfile, Lx, csv_order):
    '''Write data product to CSV file'''
    Lcsv = Lx.to_dataframe().dropna(how='all')
    if csv_order is not None:   
        names = [c for c in csv_order if c in list(Lcsv.columns)]
        Lcsv = Lcsv[names]
    Lcsv.to_csv(outfile)
    
def writeNC(outfile, Lx):
    '''Write data product to NetCDF file'''
    if os.path.isfile(outfile): 
        os.remove(outfile)
    Lx.to_netcdf(outfile, mode='w', format='NETCDF4', compute=True)    

# def writeLx(outfile, Lx):
#     '''Write Lx Dataset to .nc and .csv files'''
#     Lx.to_dataframe().dropna(how='all').to_csv(outfile+'.csv')
#     if os.path.exists(outfile+'.nc'): 
#         os.remove(outfile+'.nc')
#     Lx.to_netcdf(outfile+'.nc', mode='w', format='NETCDF4', compute=True)
    
def writeL3(outpath, station_id, l3_h, l3_d, l3_m, csv_order=None):
    '''Write L3 Dataset to .nc and .csv hourly and daily files'''
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    outfile_h = os.path.join(outpath, station_id + '_hour')
    outfile_d = os.path.join(outpath, station_id + '_day')
    outfile_m = os.path.join(outpath, station_id + '_month')
    for o,l in zip([outfile_h, outfile_d, outfile_m], [l3_h ,l3_d, l3_m]):
        writeCSV(o+'.csv',l, csv_order)
        writeNC(o+'.nc',l)        

def mergeVars(ds_list, variables, cols=['lo','hi','OOL']):                     #TODO find way to make this one line as described
    '''Merge dataset by variable attributes from lookup table file. 

    Parameters
    ----------
    ds_list : list
        List of xarray.Dataset objects
    varaibles : pandas.DataFrame
        Variable look up table
    cols : str, optional
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
    df = variables[cols]
    df = df.dropna(how='all')
    
    # Remove outliers
    ds = clipValues(ds, df, cols)
                        
    # Clean up metadata
    for k in ['format', 'hygroclip_t_offset', 'dsr_eng_coef', 'usr_eng_coef',
              'dlr_eng_coef', 'ulr_eng_coef', 'pt_z_coef', 'pt_z_p_coef',
              'pt_z_factor', 'pt_antifreeze', 'boom_azimuth', 'nodata',
              'conf', 'file']:
        if k in ds.attrs.keys():
            ds.attrs.pop(k) 
    return ds

def clipValues(ds, df, cols=['lo','hi','OOL']):
    '''Clip values in dataset to defined "hi" and "lo" variables from dataframe.
    Related issues:
    
    https://github.com/GEUS-Glaciology-and-Climate/pypromice/issues/23 - Just 
    adding special treatment here in service of replication. rh_cor is clipped 
    not NaN'd
    
    https://github.com/GEUS-Glaciology-and-Climate/pypromice/issues/20
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to clip hi-lo range to
    df : pandas.DataFrame
        Dataframe to retrieve attribute hi-lo values from
    
    Returns
    -------
    ds : xarray.Dataset
        Dataset with clipped data
    '''
    df = df[cols] 
    df = df.dropna(how='all')
    lo = cols[0]
    hi = cols[1]
    ool = cols[2]
    for var in df.index:
        if var not in list(ds.variables): 
            continue
        
        if var in ['rh_u_cor', 'rh_l_cor']:
             ds[var] = ds[var].where(ds[var] >= df.loc[var, lo], other = 0)
             ds[var] = ds[var].where(ds[var] <= df.loc[var, hi], other = 100)
        else:
            if ~np.isnan(df.loc[var, lo]):
                ds[var] = ds[var].where(ds[var] >= df.loc[var, lo])
            if ~np.isnan(df.loc[var, hi]):                
                ds[var] = ds[var].where(ds[var] <= df.loc[var, hi])
                
        other_vars = df.loc[var][ool]
        if isinstance(other_vars, str) and ~ds[var].isnull().all():            # TODO change this to accomodate for instances where all values are flagged and nan'd prior
            for o in other_vars.split():
                if o not in list(ds.variables): 
                    continue
                else:
                    if ~np.isnan(df.loc[var, lo]):
                        ds[o] = ds[o].where(ds[var] >= df.loc[var, lo])
                    if ~np.isnan(df.loc[var, hi]):  
                        ds[o] = ds[o].where(ds[var] <= df.loc[var, hi])  
    return ds

def popCols(ds, names):       
    for v in names:
        if v not in list(ds.variables):
            ds[v] = (('time'), np.arange(ds['time'].size)*np.nan)      
    return ds

def getColNames(vars_df, booms=None, data_type=None, 
                cols=['station_type', 'data_type']):
    if booms==1:
        vars_df = vars_df.loc[vars_df['station_type'].isin(['one-boom','all'])]
    elif booms==2:
        vars_df = vars_df.loc[vars_df['station_type'].isin(['two-boom','all'])]
    
    if data_type=='TX':
        vars_df = vars_df.loc[vars_df['data_type'].isin(['TX','all'])]
    elif data_type=='STM' or data_type=='raw':
        vars_df = vars_df.loc[vars_df['data_type'].isin(['raw','all'])]

    return list(vars_df.index)

def roundValues(ds, df, col='max_decimals'):
    '''Round all variable values in data array based on pre-defined rounding 
    value in variables look-up table DataFrame'''
    df = df[col]
    df = df.dropna(how='all')
    for var in df.index:
        if var not in list(ds.variables): 
            continue
        if df[var] is not np.nan:
            ds[var] = ds[var].round(decimals=int(df[var]))
    return ds
        
def addVars(ds, variables):
    '''Add variable attributes from file to dataset
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to add variable attributes to
    v_file : pandas.DataFrame
        Variables lookup table file
    
    Returns
    -------
    ds : xarray.Dataset
        Dataset with metadata
   '''
    for k in ds.keys():
        if k not in variables.index: continue
        ds[k].attrs['standard_name'] = variables.loc[k]['standard_name']
        ds[k].attrs['long_name'] = variables.loc[k]['long_name']
        ds[k].attrs['units'] = variables.loc[k]['units']
    return ds

def addMeta(ds, meta):
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

    try:
        ds.attrs['time_coverage_duration'] = pd.Timedelta((ds['time'][-1] - ds['time'][0]).values).isoformat()
        ds.attrs['time_coverage_resolution'] = pd.Timedelta((ds['time'][1] - ds['time'][0]).values).isoformat()
    except:
        ds.attrs['time_coverage_duration'] = pd.Timedelta(0).isoformat()
        ds.attrs['time_coverage_resolution'] = pd.Timedelta(0).isoformat()   
        
    ds.time.encoding["dtype"] = "int32" # CF standard requires time as int not int64
    # ds['time'].encoding['units'] = 'hours since 2016-05-01 00:00:00'
    # ds['time'] = ds['time'].astype('datetime64[D]')
    
    # Load metadata attributes and add to Dataset   
    [_addAttr(ds, key, value) for key,value in meta.items()]
    return ds

def getVars(v_file):
   '''Load variables.csv file
   
   Parameters
   ----------
   v_file : str I've moved back here  I've moved back here 
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

def resampleL3(ds_h, t):
    '''Resample L3 AWS data, e.g. hourly to daily average. This uses pandas 
    DataFrame resampling at the moment as a work-around to the xarray Dataset
    resampling. As stated, xarray resampling is a lengthy process that takes
    ~2-3 minutes per operation:

    ds_d = ds_h.resample({'time':"1D"}).mean()
    https://github.com/pydata/xarray/issues/4498 & https://stackoverflow.com/questions/64282393/
    
    This has now been fixed i I've moved back here n the latest pandas, so needs implementing:
    https://github.com/pydata/xarray/issues/4498#event-6610799698
    
    Parameters
    ----------
    ds_h : xarray.Dataset
        L3 AWS daily dataset
    t : str
        Resample factor, same variable definition as in 
        pandas.DataFrame.resample()
    
    Returns
    -------
    ds_d : xarray.Dataset
        L3 AWS hourly dataset
    '''
    df_d = ds_h.to_dataframe().resample(t).mean()
    vals = [xr.DataArray(data=df_d[c], dims=['time'], 
           coords={'time':df_d.index}, attrs=ds_h[c].attrs) for c in df_d.columns]
    ds_d = xr.Dataset(dict(zip(df_d.columns,vals)), attrs=ds_h.attrs)  
    return ds_d

def _addAttr(ds, key, value):
    '''Add attribute to xarray dataset'''
    if len(key.split('.')) == 2:
        try:
            ds[key.split('.')[0]].attrs[key.split('.')[1]] = value
        except:
            pass
            # print(f'Unable to add metadata to {key.split(".")[0]}')
    else:
        ds.attrs[key] = value    
        
def _getDateParserV1(y, doy, t):                                               #TODO fix deprecation warning
    '''Convert for yyyy,doy,hhmm (without leading 0s) to a pandas datetime.
    Example: "2007,90,430" to "2007-03-31 04:30:00"
    
    This may produce the following deprecation warning:
    FutureWarning: Use pd.to_datetime instead.
    '''
    return pd.to_datetime(f'{y}-{str(doy).zfill(3)}:{str(t).zfill(4)}',
                          format='%Y-%j:%H%M')

#------------------------------------------------------------------------------

class TestProcess(unittest.TestCase): 

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
        
        meta = getMeta('metadata.csv')
        d = addVars(d, v)
        d = addMeta(d, meta)
        self.assertTrue(d.attrs['station_id']=='TEST')
        self.assertIsInstance(d.attrs['references'], str)
    
    def testMerge(self):                                                       #TODO
        pass

    def testL0toL3(self):
        '''Test L0 to L3 processing'''
        config_file = '../test/test_config1.toml'
        inpath= '../test/'
        pAWS = AWS(config_file, inpath, None)
        self.assertIsInstance(pAWS.L3_h, xr.Dataset)
        self.assertTrue(pAWS.L3_h.attrs['station_id']=='TEST1')

#------------------------------------------------------------------------------

if __name__ == "__main__": 
    config_file = 'test/test_config1.toml'
    inpath= 'test/'
    outpath = 'test/'  
    pAWS_gc = AWS(config_file, inpath, outpath)
    
    config_file = 'test/test_config2.toml'
    inpath= 'test/'
    outpath = 'test/'  
    pAWS_gc = AWS(config_file, inpath, outpath)

