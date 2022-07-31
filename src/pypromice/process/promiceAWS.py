#!/usr/bin/env python
import os, unittest, toml
import pandas as pd
import xarray as xr
from pathlib import Path

try:
    from L0toL1 import toL1
    from L1toL2 import toL2
    from L2toL3 import toL3
    from attributes import addAllInfo, mergeVars
except:
    from promiceAWS.L0toL1 import toL1
    from promiceAWS.L1toL2 import toL2
    from promiceAWS.L2toL3 import toL3
    from promiceAWS.attributes import addAllInfo, mergeVars

pd.set_option('display.precision', 2)
xr.set_options(keep_attrs=True)

#------------------------------------------------------------------------------

class promiceAWS(object):
    '''promiceAWS object to load and process PROMICE AWS data'''
    
    def __init__(self, config_file=None, inpath=None, outpath=None):
        '''Object initialisation

        Parameters
        ----------
        config_file : str, optional
            Configuration file path. The default is None.
        inpath : str, optional
            Input file path. The default is None.
        outpath : str, optional
            Output file path. The default is None.
        '''
        assert(config_file is not None)
        assert(os.path.isfile(config_file))
        assert(inpath is not None)
        assert(os.path.isdir(inpath))
        # assert(outpath is not None)
        
        self.config_file = config_file
        self.inpath = inpath
        self.outpath = outpath
        self.config = self._load_config(config_file=self.config_file, inpath=self.inpath)
        self.L0 = self.load(conf=self.config_file, L0_path=self.inpath)
        
    def _load_config(self, config_file, inpath):
        '''Load configuration from .toml file. PROMICe .toml files support 
        defining features at the top level which apply to all nested properties, 
        but do not overwrite nested properties if they are defined
        '''
        conf = toml.load(config_file)
        # Move all top level keys to nested properties,
        # if they are not already defined in the nested properties
        # Also, insert the section name (config_file) as a file property, and configuration file.
        top = [_ for _ in conf.keys() if not type(conf[_]) is dict]
        subs = [_ for _ in conf.keys() if type(conf[_]) is dict]
        for s in subs:
            for t in top:
                if t not in conf[s].keys():
                    conf[s][t] = conf[t]

            conf[s]['conf'] = config_file
            # conf[s]['file'] = os.path.join(inpath, conf[s]['station_id'], s)
            conf[s]['file'] = os.path.join(inpath, s)

        # Delete all top level keys, because each file (sub-level) should carry all
        # properties with it.
        for t in top: conf.pop(t)

        # Check required fields are present
        for k in conf.keys():
            for field in ["columns", "station_id", "format", "latitude", "longitude", "skiprows"]:
                assert(field in conf[k].keys())
        return conf

    def _read_L0(self, conf):
        ''' Read L0 data file to DataFrame object'''
        fv = conf.get('file_version', -1)       
        if fv == 1:        
            df = pd.read_csv(conf['file'],
                             comment = "#",
                             index_col = 0,
                             na_values = conf['nodata'],
                             names = conf['columns'],
                             parse_dates = {'time': [0,1,2]},
                             date_parser = _getDatParserV1,
                             sep = ",",
                             skiprows = conf["skiprows"],
                             skip_blank_lines = True,
                             usecols=range(len(conf['columns'])))
        else:
            df = pd.read_csv(conf['file'],
                             comment = "#",
                             index_col = 0,
                             na_values = conf['nodata'],
                             names = conf['columns'],
                             parse_dates = True,
                             sep = ",",
                             skiprows = conf["skiprows"],
                             skip_blank_lines = True,
                             usecols=range(len(conf['columns'])))
            
        # Drop SKIP columns
        for c in df.columns:
            if c[0:4] == 'SKIP':
                df.drop(columns=c, inplace=True)

        # Carry relevant metadata with ds
        ds = xr.Dataset.from_dataframe(df)
        meta = {}
        skip = ["columns", "skiprows"]
        for k in conf.keys():
            if k not in skip: meta[k] = conf[k]
        ds.attrs = meta
        return ds

    def load(self, conf=None, L0_path=None):
        '''Load PROMICE AWS level 0 (L0) data. Requires associated TOML-formatted config file

        Parameters
        ----------
        conf : str    
            The path to the config file
        L0_path : str
            The path to the level 0 data folder

        Returns
        -------
        list  
            A list of xarray datasets. Each nth element of the list is an xarray 
            dataset containing the nth raw data file defined in the conf file
        '''
        c = self.config
        if len(c.keys()) == 1: # one file in this config
            ds = self._read_L0(c[list(c.keys())[0]])
            return [ds]
        else:
            ds_list = []
            for k in c.keys():
                ds_list.append(self._read_L0(c[k]))
            return ds_list
    
    def show(self):
        '''Show data attributes'''
        print(self.L0)
        print(self.L1)

    def process(self):                                                         #TODO put out-of-object instructions in example script
        '''Perform L0 to L3 data processing. This can be implemented outside
        of this object with:
        
        from promiceAWS import promiceAWS
        pap = promiceAWS(config_file='./test_data/L0/config/QAS_L.toml', inpath='./test_data/', outpath='./out')

        from promiceAWS import L0_to_L1 as L0
        pap.L1 = [toL1(item) for item in pap.L0]

        from promiceAWS import merge
        pap.L1A = mergeVars(pap.L1)

        from promiceAWS import L1_to_L2 as L1
        pap.L2 = toL2(pap.L1A)

        from promiceAWS import to_L3 as L2
        pap.L3_h, pap.L3_d = toL3(pap.L2)

        from promiceAWS import addAllInfo
        pap.L3_h = addAllInfo(pap.L3_h)
        pap.L3_d = addAllInfo(pap.L3_d)            
        '''
        self.L1 = [toL1(item) for item in self.L0]
        self.L1A = mergeVars(self.L1)
        self.L2 = toL2(self.L1A)
        self.L3_h, self.L3_d = toL3(self.L2)
        self.L3_h = addAllInfo(self.L3_h)
        self.L3_d = addAllInfo(self.L3_d)

    def write(self, outpath=None):
        '''Write L3 data to .nc and .csv hourly and daily files'''
        if outpath is None: outpath = self.outpath
        assert outpath is not None
        outpath = os.path.join(outpath, self.L3_h.attrs['station_id'])
        Path(outpath).mkdir(parents=True, exist_ok=True)
        outfile = os.path.join(outpath, self.L3_h.attrs['station_id'])

        # CSV
        self.L3_h.to_dataframe().dropna(how='all').to_csv(outfile+'_hour.csv')#,float_format='%.3f')
        self.L3_d.to_dataframe().dropna(how='all').to_csv(outfile+'_day.csv')#,float_format='%.3f')

        # NetCDF
        if os.path.exists(outfile+'_hour.nc'): os.remove(outfile+'_hour.nc')
        if os.path.exists(outfile+'_day.nc'): os.remove(outfile+'_day.nc')
        self.L3_h.to_netcdf(outfile+'_hour.nc', mode='w', format='NETCDF4', compute=True)
        self.L3_d.to_netcdf(outfile+'_day.nc', mode='w', format='NETCDF4', compute=True)

#------------------------------------------------------------------------------

def _getDatParserV1(y, doy, t):                                                #TODO fix deprecation warning
    '''Convert for yyyy,doy,hhmm (without leading 0s) to a pandas datetime.
    Example: "2007,90,430" to "2007-03-31 04:30:00"
    
    This may produce the following deprecation warning:
    FutureWarning: Use pd.to_datetime instead.
    '''
    return pd.to_datetime(f'{y}-{str(doy).zfill(3)}:{str(t).zfill(4)}',
                          format='%Y-%j:%H%M')

#------------------------------------------------------------------------------
        
class TestProcess(unittest.TestCase): 

    def testL0toL3(self):
        '''Test L0 to L3 processing'''
        config_file = '../test/test_config.toml'
        inpath= '../test/'
        pAWS = promiceAWS(config_file, inpath, None)
        pAWS.process()
        self.assertIsInstance(pAWS.L3_h, xr.Dataset)
        self.assertTrue(pAWS.L3_h.attrs['station_id']=='TEST')


if __name__ == "__main__": 
    # config_file = '../test/test_config.toml'
    # inpath= '../test/'
    # pAWS = promiceAWS(config_file, inpath, None)
    # pAWS.process()
    unittest.main()   