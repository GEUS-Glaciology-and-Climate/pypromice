#!/usr/bin/env python
"""
pypromice AWS processing module
"""
from importlib import metadata
import os, unittest, toml, datetime, uuid
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
    
    def __init__(self, config_file, inpath, var_file='./variables.csv', 
                 meta_file='./metadata.csv'):
        '''Object initialisation
        Parameters
        ----------
        config_file : str
            Configuration file path
        inpath : str
            Input file path
        var_file: str, optional
            Variables look-up table file path. The default is "./variables.csv".
        meta_file: str, optional
            Metadata info file path. The default is "./metadata.csv"
        '''
        assert(os.path.isfile(config_file))
        assert(os.path.isdir(inpath))
        print('\nAWS object initialising...')
        
        # Load config, variables CSF standards, and L0 files
        self.config = self.loadConfig(config_file, inpath)
        self.vars = getVars(var_file)
        self.meta = getMeta(meta_file)

        # Hard-wire the msg_lat and msg_lon here
        # Prevents having to list these vars in the individual station toml files
        config_keys = list(self.config.keys())
        for i in config_keys:
            self.config[i]['columns'].extend(['msg_lat', 'msg_lon'])

        # Load config file
        L0 = self.loadL0()
        self.L0=[]
        for l in L0:
            n = getColNames(self.vars, l.attrs['number_of_booms'], l.attrs['format'])
            self.L0.append(popCols(l, n))
      
    def process(self):
        '''Perform L0 to L3 data processing'''
        try:
            print(f'Commencing {self.L0.attrs["number_of_booms"]}-boom processing...')
        except:
            print(f'Commencing {self.L0[0].attrs["number_of_booms"]}-boom processing...')        
        self.getL1()
        self.getL2()
        self.getL3()

    def write(self, outpath):
        '''Write L3 data to .csv and .nc file'''
        # Save to file if outpath given
        if self.outpath is not None:
            if os.path.isdir(outpath):
                self.writeArr(outpath)
            else:
                print(f'Outpath f{outpath} does not exist. Unable to save to file')
                pass
        else:
            print('No outpath given. Unable to save to file')
            
    def getL1(self):
        '''Perform L0 to L1 data processing'''
        print('Level 1 processing...')
        self.L0 = [addBasicMeta(item, self.vars) for item in self.L0]
        self.L1 = [toL1(item, self.vars) for item in self.L0]
        self.L1A = mergeVars(self.L1, self.vars)

    def getL2(self):
        '''Perform L1 to L2 data processing'''
        print('Level 2 processing...')
        self.L2 = toL2(self.L1A)
        self.L2 = clipValues(self.L2, self.vars)

    def getL3(self):
        '''Perform L2 to L3 data processing, including resampling and metadata
        and attribute population'''
        print('Level 3 processing...')        
        self.L3 = toL3(self.L2)
        
        # Resample L3 product
        f = [l.attrs['format'] for l in self.L0]
        if 'raw' in f or 'STM' in f:
            print('Resampling to 10 minute')
            self.L3 = resampleL3(self.L3, '10min')
        else:
            self.L3 = resampleL3(self.L3, '60min') 
            print('Resampling to hour')
        
        # Re-format time 
        t = self.L3['time'].values
        self.L3['time'] = list(t)
        
        # Switch gps_lon to negative (degrees_east)
        # Do this here, and NOT in addMeta, otherwise we switch back to positive
        # when calling getMeta in joinL3! PJW
        self.L3['gps_lon'] = self.L3['gps_lon'] * -1

        # Add variable attributes and metadata
        self.L3 = self.addAttributes(self.L3)
        
        # Round all values to specified decimals places
        self.L3 = roundValues(self.L3, self.vars) 
               
    def addAttributes(self, L3):
        '''Add variable and attribute metadata
        
        Parameters
        ----------
        L3 : xr.Dataset
            Level-3 data object
        
        Returns
        -------
        L3 : xr.Dataset
            Level-3 data object with attributes
        '''
        L3 = addVars(L3, self.vars)
        L3 = addMeta(L3, self.meta)
        return L3

    def writeArr(self, outpath):
        '''Write L3 data to .nc and .csv hourly and daily files
        
        Parameters
        ----------
        outpath : str
            Output directory
        L3 : AWS.L3
            Level-3 data object
        '''
        outdir = os.path.join(outpath, self.L3.attrs['station_id']) 
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        
        f = [l.attrs['format'] for l in self.L0]
        if all(f):
            col_names = getColNames(self.vars, self.L3.attrs['number_of_booms'], 
                                    self.L0[0].attrs['format'])
        else:
            col_names = getColNames(self.vars, self.L3.attrs['number_of_booms'], 
                                    None)
        
        t = int(pd.Timedelta((self.L3['time'][1] - self.L3['time'][0]).values).total_seconds())
        print('Writing to files...')
        if t == 600:
            out_csv = os.path.join(outdir, self.L3.attrs['station_id']+'_10min.csv')
            out_nc = os.path.join(outdir, self.L3.attrs['station_id']+'_10min.nc')
        else:
            out_csv = os.path.join(outdir, self.L3.attrs['station_id']+'_hour.csv')
            out_nc = os.path.join(outdir, self.L3.attrs['station_id']+'_hour.nc')            
        writeCSV(out_csv, self.L3, col_names)
        writeNC(out_nc, self.L3) 
        print(f'Written to {out_csv}')           
        print(f'Written to {out_nc}') 
        
    def loadConfig(self, config_file, inpath):
        '''Load configuration from .toml file
        
        Parameters
        ----------
        config_file : str
            TOML file path
        inpath : str
            Input folder directory where L0 files can be found  
        
        Returns
        -------
        conf : dict
            Configuration parameters
        '''
        conf = getConfig(config_file, inpath)
        return conf
        
    def loadL0(self):
        '''Load level 0 (L0) data from associated TOML-formatted 
        config file and L0 data file
        Try readL0file() using the config with msg_lat & msg_lon appended. The 
        specific ParserError except will occur when the number of columns in 
        the tx file does not match the expected columns. In this case, remove 
        msg_lat & msg_lon from the config and call readL0file() again. These 
        station files either have no data after Nov 2022 (when msg_lat & 
        msg_lon were added to processing), or for whatever reason these fields 
        did not exist in the modem message and were not added.
        
        Returns
        -------
        ds_list : list
            List of L0 xr.Dataset objects
        '''
        c = self.config
        if len(c.keys()) == 1: # one file in this config
            target = c[list(c.keys())[0]]
            try:
                ds = self.readL0file(target)
            except pd.errors.ParserError as e:
                
                # ParserError: Too many columns specified: expected 40 and found 38
                print(f'-----> No msg_lat or msg_lon for {list(c.keys())[0]}')
                for item in ['msg_lat', 'msg_lon']:
                    target['columns'].remove(item)                             # Also removes from self.config
                ds = self.readL0file(target)
            print(f'L0 data successfully loaded from {list(c.keys())[0]}')
            return [ds]
        else:
            ds_list = []
            for k in c.keys():
                try:
                    ds_list.append(self.readL0file(c[k]))
                except pd.errors.ParserError as e:
                    
                    # ParserError: Too many columns specified: expected 40 and found 38
                    print(f'-----> No msg_lat or msg_lon for {k}')
                    for item in ['msg_lat', 'msg_lon']:
                        c[k]['columns'].remove(item)                           # Also removes from self.config
                    ds_list.append(self.readL0file(c[k]))
                print(f'L0 data successfully loaded from {k}')
            return ds_list

    def readL0file(self, conf):
        '''Read L0 .txt file to Dataset object using config dictionary and
        populate with initial metadata
        
        Parameters
        ----------
        conf : dict
            Configuration parameters  
        
        Returns
        -------
        ds : xr.Dataset
            L0 data
        '''
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
    inpath : str
        Input folder directory where L0 files can be found
    
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