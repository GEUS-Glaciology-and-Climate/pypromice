import os
import toml
import pandas as pd
import xarray as xr

import promiceAWS.L0_to_L1 as L0
import promiceAWS.merge as merge
import promiceAWS.L1_to_L2 as L1
import promiceAWS.L2_to_L3 as L2
import promiceAWS.cf_acdd as cf_acdd

pd.set_option('display.precision', 2)
xr.set_options(keep_attrs=True)

class promiceAWS:
    """promiceAWS class used to load and process PROMICE AWS data
    """
    
    config_file: str # location of config file
    config: dict     # config values, once loaded from file
    data_dir: str    # location of L0, L1, L1A, L2, and L3 folders

    # Variables to hold the data
    L0: list
    L1: list
    L1A: None # xarray Dataset
    L2: None
    L3_h: None
    L3_d: None

    def __init__(self, config_file=None, data_dir=None):
        assert(config_file is not None)
        assert(os.path.isfile(config_file))
        assert(data_dir is not None)
        assert(os.path.isdir(data_dir))
        
        self.config_file = config_file
        self.data_dir = data_dir
        self.config = self._load_config(config_file=self.config_file, data_dir=self.data_dir)
        self.L0 = self.load(conf=self.config_file, L0_path=self.data_dir+'./L0')

        
    def _load_config(self, config_file, data_dir):
        # Load a TOML file
        #
        # PROMICE TOML supports defining features at the top level which apply to
        # all nested properties, but do not overwrite nested properties if they are defined.
    
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
            conf[s]['file'] = os.path.join(data_dir, 'L0', conf[s]['station_id'], s)

        # Delete all top level keys, because each file (sub-level) should carry all
        # properties with it.
        for t in top: conf.pop(t)

        # check required fields are present
        for k in conf.keys():
            for field in ["columns", "station_id", "format", "latitude", "longitude", "skiprows"]:
                assert(field in conf[k].keys())
        
        return conf


    def _read_L0(self, conf):
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
        ds = xr.Dataset.from_dataframe(df)
    
        # carry relevant metadata with ds
        meta = {}
        skip = ["columns", "skiprows"]
        for k in conf.keys():
            if k not in skip: meta[k] = conf[k]
        ds.attrs = meta
        return ds


    def load(self, conf=None, L0_path=None):
        """Load PROMICE AWS level 0 (L0) data. Requires associated TOML-formatted config file

        Args:
          conf (str):    The path to the config file.
          L0_path (str): The path to the level 0 data folder

        Returns:
          A list of xarray datasets. Each nth element of the list is an xarray dataset containing
          the nth raw data file defined in the conf file.
        """
    
        # assert(conf is not None)
        # assert(type(conf) == str)
        # assert(L0_path is not None)

        # c = _load_config(conf, L0_path)
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
        print(self.L0)
        print(self.L1)


    def process(self):
        self.L1 = [L0.to_L1(item) for item in self.L0]
        self.L1A = merge.merge(self.L1)
        self.L2 = L1.to_L2(self.L1A)
        self.L3_h, self.L3_d = L2.to_L3(self.L2)
        self.L3_h = cf_acdd.cf_and_acdd(self.L3_h)
        self.L3_d = cf_acdd.cf_and_acdd(self.L3_d)
