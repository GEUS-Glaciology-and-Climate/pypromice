import toml
import os
import numpy as np
import pandas as pd
import pathlib
pd.set_option('display.precision', 2)
import xarray as xr
xr.set_options(keep_attrs=True)


def _load_config(toml_file, path_to_L0):
    # Load a TOML file
    #
    # PROMICE TOML supports defining features at the top level which apply to all nested properties,
    # but do not overwrite nested properties if they are defined.
    
    conf = toml.load(toml_file)
    # Move all top level keys to nested properties,
    # if they are not already defined in the nested properties
    # Also, insert the section name (toml_file) as a file property, and configuration file.
    top = [_ for _ in conf.keys() if not type(conf[_]) is dict]
    subs = [_ for _ in conf.keys() if type(conf[_]) is dict]
    for s in subs:
        for t in top:
            if t not in conf[s].keys():
                conf[s][t] = conf[t]

        conf[s]['conf'] = toml_file
        conf[s]['file'] = os.path.join(path_to_L0, conf[s]['station_id'], s)

    # Delete all top level keys, because each file (sub-level) should carry all
    # properties with it.
    for t in top: conf.pop(t)

    # check required fields are present
    for k in conf.keys():
        for field in ["columns", "station_id", "format", "latitude", "longitude", "skiprows"]:
            assert(field in conf[k].keys())

    return conf


def _read_L0(conf):
    df = pd.read_csv(conf['file'],
                     comment = "#",
                     index_col = 0,
                     na_values = conf['nodata'],
                     names = conf['columns'],
                     parse_dates = True,
                     sep = ",",
                     skiprows = conf["skiprows"],
                     skip_blank_lines = True,
                     usecols=np.arange(len(conf['columns'])))
    ds = xr.Dataset.from_dataframe(df)
    
    # carry relevant metadata with ds
    meta = {}
    skip = ["columns", "skiprows"]
    for k in conf.keys():
        if k not in skip: meta[k] = conf[k]
    ds.attrs = meta
    return ds


def load(conf=None, L0_path=None):
    """Load PROMICE AWS level 0 (L0) data. Requires associated TOML-formatted config file

    Args:
        conf (str):    The path to the config file.
        L0_path (str): The path to the level 0 data folder

    Returns:
        A list of xarray datasets. Each nth element of the list is an xarray dataset containing
        the nth raw data file defined in the conf file.
    """
    
    assert(conf is not None)
    assert(type(conf) == str)
    assert(L0_path is not None)

    c = _load_config(conf, L0_path)
    if len(c.keys()) == 1: # one file in this config
        ds = _read_L0(c[list(c.keys())[0]])
        return [ds]
    else:
        ds_list = []
        for k in c.keys():
            ds_list.append(_read_L0(c[k]))
        return ds_list
