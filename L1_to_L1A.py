#!/usr/bin/env python

import pandas as pd
import xarray as xr
import os

def L1_to_L1A(infile=None):
    
    # This could be as simple as:
    # ds = xr.open_mfdataset(infile, combine='by_coords', mask_and_scale=False).load()
    # Except that some files have overlapping times.
    
    # try:
    #     ds = xr.open_mfdataset(infile, combine='by_coords', mask_and_scale=False).load()
    # except ValueError:
    #     print("Error: files with overlapping times")
    #     print("Flag out times using flagging feature")
    #     for f in infile:
    #         print(f, xr.open_dataset(f)['time'].isel({'time':[0,-1]}).values)
    #     assert(False)
        
    if not isinstance(infile, list):
        ds = xr.open_mfdataset(infile)
    else:
        ds = xr.open_mfdataset(infile[0]).load().dropna(dim='time', how='all')
        for f in infile[1:]:
            tmp = xr.open_mfdataset(f).load().dropna(dim='time', how='all')
            ds = ds.combine_first(tmp)
            
    df = pd.read_csv("./variables.csv", index_col=0, comment="#", usecols=('field','lo','hi','OOL'))
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
    if isinstance(infile, list): infile = infile[0]
    # infile_parts = os.path.splitext(os.path.basename(infile))[0].split('_')
    # outfile = infile_parts[0] + '-' + infile_parts[-1] + '.nc' # drop year
    outfile = ds.attrs['station_id'] + '-' + ds.attrs['format'] + '.nc'
    
    outpath = os.path.split(infile)[0].split("/")
    outpath[-2] = 'L1A'
    # outfile = os.path.splitext(os.path.basename(infile))[0] + '.nc'
    outpath = '/'.join(outpath)
    outpathfile = outpath + '/' + outfile
    if os.path.exists(outpathfile): os.remove(outpathfile)
    ds.to_netcdf(outpathfile, mode='w', format='NETCDF4', compute=True)

if __name__ == "__main__":
    import sys
    # for arg in sys.argv[1:]: L1_to_L1A(arg)
    L1_to_L1A(sys.argv[1:])
