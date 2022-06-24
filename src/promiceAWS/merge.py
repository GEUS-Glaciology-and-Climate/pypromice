#!/usr/bin/env python

import pandas as pd
import os


def vartable():
   """load the variables.csv file"""
   varcsv = os.path.join(os.path.dirname(__file__), 'variables.csv')
   return pd.read_csv(varcsv, index_col=0, comment="#")

def merge(ds_list):
    
    # This could be as simple as:
    # ds = xr.open_mfdataset(infile_list, combine='by_coords', mask_and_scale=False).load()
    # Except that some files have overlapping times.
    
    # try:
    #     ds = xr.open_mfdataset(infile, combine='by_coords', mask_and_scale=False).load()
    # except ValueError:
    #     print("Error: files with overlapping times")
    #     print("Flag out times using flagging feature")
    #     for f in infile:
    #         print(f, xr.open_dataset(f)['time'].isel({'time':[0,-1]}).values)
    #     assert(False)

    ds = ds_list[0]
    if len(ds_list) > 1:
        for d in ds_list[1:]:
            ds = ds.combine_first(d)
        
    #df=pd.read_csv("./variables.csv", index_col=0, comment="#", usecols=('field','lo','hi','OOL'))
    df = vartable()[['lo','hi','OOL']]
    df = df.dropna(how='all')
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


    # clean up metadata
    for k in ['format', 'hygroclip_t_offset', 'dsr_eng_coef', 'usr_eng_coef',
              'dlr_eng_coef', 'ulr_eng_coef', 'pt_z_coef', 'pt_z_p_coef',
              'pt_z_factor', 'pt_antifreeze', 'boom_azimuth', 'nodata',
              'conf', 'file']:
        if k in ds.attrs.keys():
            ds.attrs.pop(k)
    
    return ds
