#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities module for data formatting, populating and metadata handling
"""
import numpy as np

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
    # skip = ["columns", "skiprows"]
    for k in conf.keys():
        if k not in skip: ds.attrs[k] = conf[k]
    return ds

