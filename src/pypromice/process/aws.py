#!/usr/bin/env python
"""
AWS data processing module
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging, os
import pandas as pd
import xarray as xr
from functools import reduce

from pypromice.process.L0toL1 import toL1
from pypromice.process.L1toL2 import toL2
from pypromice.process.L2toL3 import toL3
from pypromice.process import write, load, utilities
from pypromice.process.resample import resample_dataset

pd.set_option('display.precision', 2)
xr.set_options(keep_attrs=True)
logger = logging.getLogger(__name__)

class AWS(object):
    '''AWS object to load and process PROMICE AWS data'''

    def __init__(self, config_file, inpath, var_file=None, meta_file=None):
        '''Object initialisation

        Parameters
        ----------
        config_file : str
            Configuration file path
        inpath : str
            Input file path
        var_file: str, optional
            Variables look-up table file path. If not given then pypromice's
            variables file is used. The default is None.
        meta_file: str, optional
            Metadata info file path. If not given then pypromice's
            metadata file is used. The default is None.
        '''
        assert(os.path.isfile(config_file)), "cannot find "+config_file
        assert(os.path.isdir(inpath)), "cannot find "+inpath
        logger.info('AWS object initialising...')

        # Load config, variables CSF standards, and L0 files
        self.config = self.loadConfig(config_file, inpath)
        self.vars = load.getVars(var_file)
        self.meta = load.getMeta(meta_file)

        # Load config file
        L0 = self.loadL0()
        self.L0=[]
        for l in L0:
            n = write.getColNames(self.vars, 
                                  l.attrs['number_of_booms'], 
                                  l.attrs['format'])
            self.L0.append(utilities.popCols(l, n))

        self.L1 = None
        self.L1A = None
        self.L2 = None
        self.L3 = None

    def process(self):
        '''Perform L0 to L3 data processing'''
        try:
            logger.info(f'Commencing {self.L0.attrs["number_of_booms"]}-boom processing...')
        except:
            logger.info(f'Commencing {self.L0[0].attrs["number_of_booms"]}-boom processing...')
        self.getL1()
        self.getL2()
        self.getL3()

    def writeL2(self, outpath):
        '''Write L2 data to .csv and .nc file'''
        if os.path.isdir(outpath):
            self.writeArr(self.L2, outpath)
        else:
            logger.info(f'Outpath f{outpath} does not exist. Unable to save to file')
            pass

    def writeL3(self, outpath):
        '''Write L3 data to .csv and .nc file'''
        if os.path.isdir(outpath):
            self.writeArr(self.L3, outpath)
        else:
            logger.info(f'Outpath f{outpath} does not exist. Unable to save to file')
            pass
        
    def getL1(self):
        '''Perform L0 to L1 data processing'''
        logger.info('Level 1 processing...')
        self.L0 = [utilities.addBasicMeta(item, self.vars) for item in self.L0]
        self.L1 = [toL1(item, self.vars) for item in self.L0]
        self.L1A = reduce(xr.Dataset.combine_first, self.L1)

    def getL2(self):
        '''Perform L1 to L2 data processing'''
        logger.info('Level 2 processing...')
        self.L2 = toL2(self.L1A, vars_df=self.vars)

    def getL3(self):
        '''Perform L2 to L3 data processing, including resampling and metadata
        and attribute population'''
        logger.info('Level 3 processing...')
        self.L3 = toL3(self.L2)

    def resample(self, dataset):       
        '''Resample dataset to specific temporal resolution (based on input
        data type)'''
        f = [l.attrs['format'] for l in self.L0]
        if 'raw' in f or 'STM' in f:
            logger.info('Resampling to 10 minute')
            resampled = resample_dataset(dataset, '10min')
        else:
            resampled = resample_dataset(dataset, '60min')
            logger.info('Resampling to hour')
        return resampled

    def writeArr(self, dataset, outpath):
        '''Write L3 data to .nc and .csv hourly and daily files

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset to write to file
        outpath : str
            Output directory
        '''
        # Resample dataset based on data type (tx/raw)
        d2 = self.resample(dataset)
        
        # Reformat time
        d2 = utilities.reformat_time(d2)
        
        # Reformat longitude (to negative values)
        d2 = utilities.reformat_lon(d2)
        
        # Add variable attributes and metadata
        d2 = self.addAttributes(d2)

        # Round all values to specified decimals places
        d2 = utilities.roundValues(d2, self.vars)
        
        # Create out directory
        outdir = os.path.join(outpath, d2.attrs['station_id'])
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        
        # Get variable names to write out
        col_names = write.getColNames(
            self.vars,
            d2.attrs['number_of_booms'],
            d2.attrs['format'],
            d2.attrs['bedrock'],
        )
        
        # Define filename based on resample rate
        t = int(pd.Timedelta((d2['time'][1] - d2['time'][0]).values).total_seconds())
        if t == 600:
            out_csv = os.path.join(outdir, d2.attrs['station_id']+'_10min.csv')
            out_nc = os.path.join(outdir, d2.attrs['station_id']+'_10min.nc')
        else:
            out_csv = os.path.join(outdir, d2.attrs['station_id']+'_hour.csv')
            out_nc = os.path.join(outdir, d2.attrs['station_id']+'_hour.nc')
        
        # Write to csv file
        logger.info('Writing to files...')
        write.writeCSV(out_csv, d2, col_names)
        
        # Write to netcdf file
        col_names = col_names + ['lat', 'lon', 'alt']
        write.writeNC(out_nc, d2, col_names)
        logger.info(f'Written to {out_csv}')
        logger.info(f'Written to {out_nc}')
        
    # def merge_flag(self):
    #     '''Determine if hard merging is needed, based on whether a hard 
    #     merge_type flag is defined in any of the configs'''
    #     f = [l.attrs['merge_type'] for l in self.L0]
    #     if 'hard' in f:
    #         return True
    #     else:
    #         return False
        
    # def hard_merge(self, dataset_list):
    #     '''Determine positions where hard merging should occur, combine 
    #     data and append to list of combined data chunks, then hard merge all 
    #     combined data chunks. This should be called in instances where there 
    #     needs to be a clear break between input datasets, such as when a station
    #     is moved (and we do not want the GPS position jumping)'''
    #     # Define positions where hard merging should occur
    #     m=[]
    #     f = [l.attrs['merge_type'] for l in self.L0]
    #     [m.append(i) for i, item in enumerate(f) if item=='hard']
        
    #     # Perform combine between hard merge breaks and append to list of combined data
    #     combined=[]
    #     for i in range(len(m[:-1])):        
    #         combined.append(reduce(xr.Dataset.combine_first, dataset_list[m[i]:m[i+1]]))
    #     combined.append(reduce(xr.Dataset.combine_first, dataset_list[m[-1]:]))
        
    #     # Hard merge all combined datasets together
    #     return reduce(xr.Dataset.update, combined)
                
    def addAttributes(self, dataset):
        '''Add variable and attribute metadata

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset (i.e. L2 or L3) object

        Returns
        -------
        d2 : xr.Dataset
            Data object with attributes
        '''
        d2 = utilities.addVars(dataset, self.vars)
        d2 = utilities.addMeta(dataset, self.meta)
        return d2

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
        conf = load.getConfig(config_file, inpath)
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
        ds_list = []
        for k in self.config.keys():
            target = self.config[k]
            try:
                ds_list.append(self.readL0file(target))

            except pd.errors.ParserError as e:
                # ParserError: Too many columns specified: expected 40 and found 38
                # logger.info(f'-----> No msg_lat or msg_lon for {k}')
                for item in ['msg_lat', 'msg_lon']:
                    target['columns'].remove(item)                           # Also removes from self.config
                ds_list.append(self.readL0file(target))
            logger.info(f'L0 data successfully loaded from {k}')
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
        ds = load.getL0(conf['file'], conf['nodata'], conf['columns'], 
                   conf["skiprows"], file_version, time_offset=conf.get('time_offset'))
        ds = utilities.populateMeta(ds, conf, ["columns", "skiprows", "modem"])
        return ds
