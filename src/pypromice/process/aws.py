#!/usr/bin/env python
"""
AWS data processing module
"""
import logging
from functools import reduce
from importlib import metadata
import os, unittest, toml, datetime, uuid, pkg_resources
from typing import Sequence, Optional

import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import xarray as xr
from datetime import timedelta

from pypromice.process.L0toL1 import toL1
from pypromice.process.L1toL2 import toL2
from pypromice.process.L2toL3 import toL3

pd.set_option('display.precision', 2)
xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------


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
        self.vars = getVars(var_file)
        self.meta = getMeta(meta_file)

        # Load config file
        L0 = self.loadL0()
        self.L0=[]
        for l in L0:
            n = getColNames(self.vars, l.attrs['number_of_booms'], l.attrs['format'])
            self.L0.append(popCols(l, n))

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

    def write(self, outpath):
        '''Write L3 data to .csv and .nc file'''
        if os.path.isdir(outpath):
            self.writeArr(outpath)
        else:
            logger.info(f'Outpath f{outpath} does not exist. Unable to save to file')
            pass

    def getL1(self):
        '''Perform L0 to L1 data processing'''
        logger.info('Level 1 processing...')
        self.L0 = [addBasicMeta(item, self.vars) for item in self.L0]
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

        # Resample L3 product
        f = [l.attrs['format'] for l in self.L0]
        if 'raw' in f or 'STM' in f:
            logger.info('Resampling to 10 minute')
            self.L3 = resampleL3(self.L3, '10min')
        else:
            self.L3 = resampleL3(self.L3, '60min')
            logger.info('Resampling to hour')

        # Re-format time
        t = self.L3['time'].values
        self.L3['time'] = list(t)

        # Switch gps_lon to negative (degrees_east)
        # Do this here, and NOT in addMeta, otherwise we switch back to positive
        # when calling getMeta in joinL3! PJW
        if self.L3.attrs['station_id'] not in ['UWN', 'Roof_GEUS', 'Roof_PROMICE']:
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

        col_names = getColNames(
            self.vars,
            self.L3.attrs['number_of_booms'],
            self.L3.attrs['format'],
            self.L3.attrs['bedrock'],
        )

        t = int(pd.Timedelta((self.L3['time'][1] - self.L3['time'][0]).values).total_seconds())
        logger.info('Writing to files...')
        if t == 600:
            out_csv = os.path.join(outdir, self.L3.attrs['station_id']+'_10min.csv')
            out_nc = os.path.join(outdir, self.L3.attrs['station_id']+'_10min.nc')
        else:
            out_csv = os.path.join(outdir, self.L3.attrs['station_id']+'_hour.csv')
            out_nc = os.path.join(outdir, self.L3.attrs['station_id']+'_hour.nc')
        writeCSV(out_csv, self.L3, col_names)
        col_names = col_names + ['lat', 'lon', 'alt']
        writeNC(out_nc, self.L3, col_names)
        logger.info(f'Written to {out_csv}')
        logger.info(f'Written to {out_nc}')

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
        ds_list = []
        for k in self.config.keys():
            target = self.config[k]
            try:
                ds_list.append(self.readL0file(target))

            except pd.errors.ParserError as e:
                # ParserError: Too many columns specified: expected 40 and found 38
                logger.info(f'-----> No msg_lat or msg_lon for {k}')
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
        ds = getL0(conf['file'], conf['nodata'], conf['columns'], 
                   conf["skiprows"], file_version, time_offset=conf.get('time_offset'))
        ds = populateMeta(ds, conf, ["columns", "skiprows", "modem"])
        return ds

#------------------------------------------------------------------------------

def getConfig(config_file, inpath, default_columns: Sequence[str] = ('msg_lat', 'msg_lon')):
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
        conf[s]["columns"].extend(default_columns)

    for t in top: conf.pop(t)                                                  # Delete all top level keys beause each file
                                                                               # should carry all properties with it
    for k in conf.keys():                                                      # Check required fields are present
        for field in ["columns", "station_id", "format", "skiprows"]:
            assert(field in conf[k].keys()), field+" not in config keys"
    return conf

def getL0(infile, nodata, cols, skiprows, file_version,
          delimiter=',', comment='#', time_offset: Optional[float] = None) -> xr.Dataset:
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
    time_offset : Optional[float]
        Time offset in hours for correcting for non utc time data.
    Returns
    -------
    ds : xarray.Dataset
        L0 Dataset
    '''
    if file_version == 1:
        df = pd.read_csv(infile, comment=comment, index_col=0,
                         na_values=nodata, names=cols,
                         sep=delimiter,
                         skiprows=skiprows, skip_blank_lines=True,
                         usecols=range(len(cols)),
                         low_memory=False)
        df['time'] = pd.to_datetime(
                                    df.year.astype(str) \
                                        + df.doy.astype(str).str.zfill(3) \
                                            + df.hhmm.astype(str).str.zfill(4),
                                    format='%Y%j%H%M'
                                    )
        df = df.set_index('time')

    else:
        df = pd.read_csv(infile, comment=comment, index_col=0,
                         na_values=nodata, names=cols, parse_dates=True,
                         sep=delimiter, skiprows=skiprows,
                         skip_blank_lines=True,
                         usecols=range(len(cols)),
                         low_memory=False)
        try:
            df.index = pd.to_datetime(df.index)
        except  ValueError as e:
            logger.info("\n", infile)
            logger.info("\nValueError:")
            logger.info(e)
            logger.info('\t\t> Trying pd.to_datetime with format=mixed')
            try:
                df.index = pd.to_datetime(df.index, format='mixed')
            except Exception as e:
                logger.info("\nDateParseError:")
                logger.info(e)
                logger.info('\t\t> Trying again removing apostrophes in timestamp (old files format)')
                df.index = pd.to_datetime(df.index.str.replace("\"",""))

    if time_offset is not None:
        df.index = df.index + timedelta(hours=time_offset)

    # Drop SKIP columns
    for c in df.columns:
        if c[0:4] == 'SKIP':
            df.drop(columns=c, inplace=True)

    # Carry relevant metadata with ds
    ds = xr.Dataset.from_dataframe(df)
    return ds

def addBasicMeta(ds, vars_df):
    ''' Use a variable lookup table DataFrame to add the basic metadata
    to the xarray dataset. This is later amended to finalise L3

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to add metadata to
    vars_df : pd.DataFrame
        Metadata dataframe

    Returns
    -------
    ds : xr.Dataset
        Dataset with added metadata
    '''
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
    '''Write data product to CSV file

    Parameters
    ----------
    outfile : str
        Output file path
    Lx : xr.Dataset
        Dataset to write to file
    csv_order : list
        List order of variables
    '''
    Lcsv = Lx.to_dataframe().dropna(how='all')
    if csv_order is not None:
        names = [c for c in csv_order if c in list(Lcsv.columns)]
        Lcsv = Lcsv[names]
    Lcsv.to_csv(outfile)

def writeNC(outfile, Lx, col_names=None):
    '''Write data product to NetCDF file

    Parameters
    ----------
    outfile : str
        Output file path
    Lx : xr.Dataset
        Dataset to write to file
    '''
    if os.path.isfile(outfile):
        os.remove(outfile)
    if col_names is not None:
        names = [c for c in col_names if c in list(Lx.keys())]
    else:
        names = list(Lx.keys())
    Lx[names].to_netcdf(outfile, mode='w', format='NETCDF4', compute=True)

def writeAll(outpath, station_id, l3_h, l3_d, l3_m, csv_order=None):
    '''Write L3 hourly, daily and monthly datasets to .nc and .csv
    files

    outpath : str
        Output file path
    station_id : str
        Station name
    l3_h : xr.Dataset
        L3 hourly data
    l3_d : xr.Dataset
        L3 daily data
    l3_m : xr.Dataset
        L3 monthly data
    csv_order : list, optional
        List order of variables
    '''
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    outfile_h = os.path.join(outpath, station_id + '_hour')
    outfile_d = os.path.join(outpath, station_id + '_day')
    outfile_m = os.path.join(outpath, station_id + '_month')
    for o,l in zip([outfile_h, outfile_d, outfile_m], [l3_h ,l3_d, l3_m]):
        writeCSV(o+'.csv',l, csv_order)
        writeNC(o+'.nc',l)


def popCols(ds, names):
    '''Populate dataset with all given variable names

    Parammeters
    -----------
    ds : xr.Dataset
        Dataset
    names : list
        List of variable names to populate
    '''
    for v in names:
        if v not in list(ds.variables):
            ds[v] = (('time'), np.arange(ds['time'].size)*np.nan)
    return ds

def getColNames(vars_df, booms=None, data_type=None, bedrock=False):
    '''Get all variable names for a given data type, based on a variables
    look-up table

    Parameters
    ----------
    vars_df : pd.DataFrame
        Variables look-up table
    booms : int, optional
        Number of booms. If this parameter is empty then all variables
        regardless of boom type will be passed. The default is None.
    data_type : str, optional
        Data type, "tx", "STM" or "raw". If this parameter is empty then all
        variables regardless of data type will be passed. The default is None.

    Returns
    -------
    list
        Variable names
    '''
    if booms==1:
        vars_df = vars_df.loc[vars_df['station_type'].isin(['one-boom','all'])]
    elif booms==2:
        vars_df = vars_df.loc[vars_df['station_type'].isin(['two-boom','all'])]

    if data_type=='TX':
        vars_df = vars_df.loc[vars_df['data_type'].isin(['TX','all'])]
    elif data_type=='STM' or data_type=='raw':
        vars_df = vars_df.loc[vars_df['data_type'].isin(['raw','all'])]

    col_names = list(vars_df.index)
    if isinstance(bedrock, str):
        bedrock = (bedrock.lower() == 'true')
    if bedrock == True:
        col_names.remove('cc')
        for v in ['dlhf_u', 'dlhf_l', 'dshf_u', 'dshf_l']:
            try:
                col_names.remove(v)
            except:
                pass
    return col_names

def roundValues(ds, df, col='max_decimals'):
    '''Round all variable values in data array based on pre-defined rounding
    value in variables look-up table DataFrame

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to round values in
    df : pd.Dataframe
        Variable look-up table with rounding values
    col : str
        Column in variable look-up table that contains rounding values. The
        default is "max_decimals"
    '''
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
    variables : pandas.DataFrame
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
        ds[k].attrs['coverage_content_type'] = variables.loc[k]['coverage_content_type']
        ds[k].attrs['coordinates'] = variables.loc[k]['coordinates']
    return ds

def addMeta(ds, meta):
    '''Add metadata attributes from file to dataset

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to add metadata attributes to
    meta : dict
        Metadata file

    Returns
    -------
    ds : xarray.Dataset
        Dataset with metadata
   '''
    ds['lon'] = ds['gps_lon'].mean()
    ds['lon'].attrs = ds['gps_lon'].attrs

    ds['lat'] = ds['gps_lat'].mean()
    ds['lat'].attrs = ds['gps_lat'].attrs

    ds['alt'] = ds['gps_alt'].mean()
    ds['alt'].attrs = ds['gps_alt'].attrs

    # for k in ds.keys(): # for each var
    #     if 'units' in ds[k].attrs:
    #         if ds[k].attrs['units'] == 'C':
    #             ds[k].attrs['units'] = 'degrees_C'

    # https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3#geospatial_bounds
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

    ds.attrs['geospatial_lat_min'] = str(ds['lat'].min().values)
    ds.attrs['geospatial_lat_max'] = str(ds['lat'].max().values)
    ds.attrs['geospatial_lon_min'] = str(ds['lon'].min().values)
    ds.attrs['geospatial_lon_max'] = str(ds['lon'].max().values)
    ds.attrs['geospatial_vertical_min'] = str(ds['alt'].min().values)
    ds.attrs['geospatial_vertical_max'] = str(ds['alt'].max().values)
    ds.attrs['geospatial_vertical_positive'] = 'up'
    ds.attrs['time_coverage_start'] = str(ds['time'][0].values)
    ds.attrs['time_coverage_end'] = str(ds['time'][-1].values)

    try:
        ds.attrs['source']= 'pypromice v' + str(metadata.version('pypromice'))
    except:
        ds.attrs['source'] = 'pypromice'

    # https://www.digi.com/resources/documentation/digidocs/90001437-13/reference/r_iso_8601_duration_format.htm
    try:
        ds.attrs['time_coverage_duration'] = str(pd.Timedelta((ds['time'][-1] - ds['time'][0]).values).isoformat())
        ds.attrs['time_coverage_resolution'] = str(pd.Timedelta((ds['time'][1] - ds['time'][0]).values).isoformat())
    except:
        ds.attrs['time_coverage_duration'] = str(pd.Timedelta(0).isoformat())
        ds.attrs['time_coverage_resolution'] = str(pd.Timedelta(0).isoformat())

    # Note: int64 dtype (long int) is incompatible with OPeNDAP access via THREDDS for NetCDF files
    # See https://stackoverflow.com/questions/48895227/output-int32-time-dimension-in-netcdf-using-xarray
    ds.time.encoding["dtype"] = "i4" # 32-bit signed integer
    #ds.time.encoding["calendar"] = 'proleptic_gregorian' # this is default

    # Load metadata attributes and add to Dataset
    [_addAttr(ds, key, value) for key,value in meta.items()]

    # Check attribute formating
    for k,v in ds.attrs.items():
        if not isinstance(v, str) or not isinstance(v, int):
            ds.attrs[k]=str(v)
    return ds


def getVars(v_file=None):
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
   if v_file is None:
        with pkg_resources.resource_stream('pypromice', 'process/variables.csv') as stream:
            return pd.read_csv(stream, index_col=0, comment="#", encoding='utf-8')
   else:
        return pd.read_csv(v_file, index_col=0, comment="#")


def getMeta(m_file=None, delimiter=','):                                            #TODO change to DataFrame output to match variables.csv
    '''Load metadata table

    Parameters
    ----------
    m_file : str
        Metadata file path
    delimiter : str
        Metadata character delimiter. The default is ","

    Returns
    -------
    meta : dict
        Metadata dictionary
    '''
    meta={}
    if m_file is None:
        with pkg_resources.resource_stream('pypromice', 'process/metadata.csv') as stream:
            lines = stream.read().decode("utf-8")
            lines = lines.split("\n")
    else:
        with open(m_file, 'r') as f:
            lines = f.readlines()
    for l in lines[1:]:
        try:
            meta[l.split(',')[0]] = l.split(delimiter)[1].split('\n')[0].replace(';',',')
        except IndexError:
            pass
    return meta

def resampleL3(ds_h, t):
    '''Resample L3 AWS data, e.g. hourly to daily average. This uses pandas
    DataFrame resampling at the moment as a work-around to the xarray Dataset
    resampling. As stated, xarray resampling is a lengthy process that takes
    ~2-3 minutes per operation: ds_d = ds_h.resample({'time':"1D"}).mean()
    This has now been fixed, so needs implementing:
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
    # recalculating wind direction from averaged directional wind speeds
    for var in ['wdir_u','wdir_l','wdir_i']:
        if var in df_d.columns:
            if ('wspd_x_'+var.split('_')[1] in df_d.columns) & ('wspd_x_'+var.split('_')[1] in df_d.columns):
                df_d[var] = _calcWindDir(df_d['wspd_x_'+var.split('_')[1]],
                                   df_d['wspd_y_'+var.split('_')[1]])
            else:
                logger.info(var,'in dataframe but not','wspd_x_'+var.split('_')[1],'wspd_x_'+var.split('_')[1])
    vals = [xr.DataArray(data=df_d[c], dims=['time'],
           coords={'time':df_d.index}, attrs=ds_h[c].attrs) for c in df_d.columns]
    ds_d = xr.Dataset(dict(zip(df_d.columns,vals)), attrs=ds_h.attrs)
    return ds_d


def _calcWindDir(wspd_x, wspd_y):
    '''Calculate wind direction in degrees

    Parameters
    ----------
    wspd_x : xarray.DataArray
        Wind speed in X direction
    wspd_y : xarray.DataArray
        Wind speed in Y direction

    Returns
    -------
    wdir : xarray.DataArray
        Wind direction'''
    deg2rad = np.pi / 180
    rad2deg = 1 / deg2rad
    wdir = np.arctan2(wspd_x, wspd_y) * rad2deg
    wdir = (wdir + 360) % 360
    return wdir


def _addAttr(ds, key, value):
    '''Add attribute to xarray dataset

    ds : xr.Dataset
        Dataset to add attribute to
    key : str
        Attribute name, with "." denoting variable attributes
    value : str/int
        Value for attribute'''
    if len(key.split('.')) == 2:
        try:
            ds[key.split('.')[0]].attrs[key.split('.')[1]] = str(value)
        except:
            pass
            # logger.info(f'Unable to add metadata to {key.split(".")[0]}')
    else:
        ds.attrs[key] = value


#------------------------------------------------------------------------------

class TestProcess(unittest.TestCase):

    def testgetVars(self):
        '''Test variable table lookup retrieval'''
        v = getVars()
        self.assertIsInstance(v, pd.DataFrame)
        self.assertTrue(v.columns[0] in 'standard_name')
        self.assertTrue(v.columns[2] in 'units')

    def testgetMeta(self):
        '''Test AWS names retrieval'''
        m = getMeta()
        self.assertIsInstance(m, dict)
        self.assertTrue('references' in m)

    def testAddAll(self):
        '''Test variable and metadata attributes added to Dataset'''
        d = xr.Dataset()
        v = getVars()
        att = list(v.index)
        att1 = ['gps_lon', 'gps_lat', 'gps_alt', 'albedo', 'p']
        for a in att:
            d[a]=[0,1]
        for a in att1:
            d[a]=[0,1]
        d['time'] = [datetime.datetime.now(),
                     datetime.datetime.now()-timedelta(days=365)]
        d.attrs['station_id']='TEST'
        meta = getMeta()
        d = addVars(d, v)
        d = addMeta(d, meta)
        self.assertTrue(d.attrs['station_id']=='TEST')
        self.assertIsInstance(d.attrs['references'], str)

    def testL0toL3(self):
        '''Test L0 to L3 processing'''
        try:
            import pypromice
            pAWS = AWS(os.path.join(os.path.dirname(pypromice.__file__),'test/test_config1.toml'),
                       os.path.join(os.path.dirname(pypromice.__file__),'test'))
        except:
            pAWS = AWS('../test/test_config1.toml', '../test/')
        pAWS.process()
        self.assertIsInstance(pAWS.L3, xr.Dataset)
        self.assertTrue(pAWS.L3.attrs['station_id']=='TEST1')

    def testCLIgetl3(self):
        '''Test get_l3 CLI'''
        exit_status = os.system('get_l3 -h')
        self.assertEqual(exit_status, 0)

    def testCLIjoinl3(self):
        '''Test join_l3 CLI'''
        exit_status = os.system('join_l3 -h')
        self.assertEqual(exit_status, 0)
        
#------------------------------------------------------------------------------

if __name__ == "__main__":

    # # Test an individual station
    # test_station = 'xxx'
    # # config_file = '../../../../aws-l0/raw/config/{}.toml'.format(test_station)
    # config_file = '../../../../aws-l0/tx/config/{}.toml'.format(test_station)
    # # inpath= '../../../../aws-l0/raw/{}/'.format(test_station)
    # inpath= '../../../../aws-l0/tx/'
    # vari = 'variables.csv'
    # pAWS_gc = AWS(config_file, inpath, var_file=vari)
    # pAWS_gc.process()
    # pAWS_gc.getL1()
    # pAWS_gc.getL2()
    # pAWS_gc.getL3()

    # # Use test configs
    # config_files = ['test/test_config1.toml', 'test/test_config2.toml']
    # inpath= 'test/'
    # outpath = 'test/'
    # vari = 'variables.csv'
    # for cf in config_files:
    #     pAWS_gc = AWS(cf, inpath, var_file=vari)
    #     pAWS_gc.process()

    unittest.main()
