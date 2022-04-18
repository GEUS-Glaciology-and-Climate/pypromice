#!/usr/bin/env python

def L0_to_L1(conf=None):
    import re
    import shapely
    from shapely import geometry
    import os
    import sys
    import numpy as np
    import pandas as pd
    import os
    import numpy as np
    import pandas as pd
    import pathlib
    pd.set_option('display.precision', 2)
    import xarray as xr
    xr.set_options(keep_attrs=True)
    
    def read_L0(conf):
    
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
    ds = read_L0(conf)
    
    ds['n'] = (('time'), np.arange(ds.time.size)+1)
    
    def flag_NAN(ds):
        flag_file = "./data/flags/" + ds.attrs["station_id"] + ".csv"
    
        if not pathlib.Path(flag_file).is_file(): return ds # no flag file
        
        df = pd.read_csv(flag_file, parse_dates=[0,1], comment="#")\
               .dropna(how='all', axis='rows')
    
        # check format of flags.csv. Either both or neither of t0 and t1 must be defined.
        assert(((np.isnan(df['t0'].values).astype(int) + np.isnan(df['t1'].values).astype(int)) % 2).sum() == 0)
        # for now we only process the NAN flag
        df = df[df['flag'] == "NAN"]
        if df.shape[0] == 0: return ds
    
        for i in df.index:
            t0, t1, avar = df.loc[i,['t0','t1','variable']]
            # set to all vars if var is "*"
            varlist = avar.split() if avar != '*' else list(ds.variables)
            if 'time' in varlist: varlist.remove("time")
            # set to all times if times are "n/a"
            if pd.isnull(t0): t0, t1 = ds['time'].values[[0,-1]]
            for v in varlist:
                ds[v] = ds[v].where((ds['time'] < t0) | (ds['time'] > t1))
    
            # TODO: Mark these values in the ds_flags dataset using perhaps flag_LUT.loc["NAN"]['value']
    
        return ds
    ds = flag_NAN(ds)
    
    def add_variable_metadata(ds):
        """Uses the variable DB (variables.csv) to add metadata to the xarray dataset."""
        df = pd.read_csv("./variables.csv", index_col=0, comment="#")
    
        for v in df.index:
            if v == 'time': continue # coordinate variable, not normal var
            if v not in list(ds.variables): continue
            for c in ['standard_name', 'long_name', 'units']:
                if isinstance(df[c][v], float) and np.isnan(df[c][v]): continue
                ds[v].attrs[c] = df[c][v]
                
        return ds
    ds = add_variable_metadata(ds)
    
    # create variables that are missing
    df = pd.read_csv("./variables.csv", index_col=0, comment="#", usecols=('field','lo','hi','OOL'))
    for v in df.index:
        if v not in list(ds.variables):
            ds[v] = (('time'), np.arange(ds['time'].size)*np.nan)
    
    if ~ds['z_pt'].isnull().all(): assert("pt_antifreeze" in ds.attrs.keys())
    if 't_2' in list(ds.variables): assert("hygroclip_t_offset" in ds.attrs.keys())
    
    T_0 = 273.15
    
    # Calculate pressure transducer fluid density
    if ~ds['z_pt'].isnull().all():
        if ds.attrs['pt_antifreeze'] == 50:
            rho_af = 1092
        elif ds.attrs['pt_antifreeze'] == 100:
            rho_af = 1145
        else:
            rho_af = np.nan
            print("ERROR: Incorrect metadata: 'pt_antifreeze =' ", ds.attrs['pt_antifreeze'])
            print("Antifreeze mix only supported at 50 % or 100%")
            # assert(False)
        
    
    for v in ['gps_geounit','min_y']:
        if v in list(ds.variables): ds = ds.drop_vars(v)
            
    ## adjust times based on file format.
    # raw: No adjust (timestamp is at start of period)
    # STM: Adjust timestamp from end of period to start of period
    # TX: Adjust timestamp start of period (hour/day) also depending on season
    
    def time_shift(da):
        """ Adjust times
        raw: (10 min) values are sampled instantaneously. Don't call this function
        STM: (1 hour) values are averaged and timestamp is end. Shift 1 h earlier to beginning
        TX: Some 10 min, some 1 hour, some 1 day? Shift appropriately.
        """
        # assert(ds.attrs['format'] != 'raw')
        if ds.attrs['format'] == 'raw':
            # diff = da['time'].diff(dim='time')
            # diffarr = diff.values.astype('timedelta64[h]').astype(int)
            # # assume the 1st time step (dropped via diff) is equal to the 2nd timestep
            # diffarr = np.append(diffarr[0], diffarr)
            # da['time'] = (da['time'] + pd.to_timedelta("-1 hour"))\
            #     .where((diffarr == 1) & (da['time'].dt.dayofyear <= 300) & (da['time'].dt.dayofyear >= 100), other=da['time'])
    
            ### NOTE: The following line re-implements bug: https://github.com/GEUS-PROMICE/AWS_v3/issues/2
            ### See also https://github.com/GEUS-PROMICE/PROMICE-AWS-processing/issues/20
            t = (da['time'] + pd.to_timedelta("-24 hours"))\
                .where((da['time'].dt.hour == 23) & ((da['time'].dt.dayofyear <= 300) & (da['time'].dt.dayofyear >= 100)), other=da['time'])
            for t in zip(t,da['time'].values):
                print(t)
        if ds.attrs['format'] == 'STM':
            t = da['time'] + pd.to_timedelta("-1 hour")
        if ds.attrs['format'] == 'TX':
            diff = da['time'].diff(dim='time')
            diffarr = diff.values.astype('timedelta64[h]').astype(int)
            # assume the 1st time step (dropped via diff) is equal to the 2nd timestep
            # diffarr = np.append(diffarr[0], diffarr)
            diffarr = np.append(0, diffarr) # no, don't.
            t = (da['time'] + pd.to_timedelta("-1 hour"))\
                .where(# (diffarr == 1) &
                       (da['time'].dt.dayofyear <= 300) &
                       (da['time'].dt.dayofyear >= 100),
                       other=da['time'])
    
            ### NOTE: The following line re-implements bug: https://github.com/GEUS-PROMICE/AWS_v3/issues/2
            ### See also https://github.com/GEUS-PROMICE/PROMICE-AWS-processing/issues/20
            # print(da['time'])
            # t = (da['time'] + pd.to_timedelta("+24 hours"))\
            #     .where((da['time'].dt.hour == 23) & ((da['time'].dt.dayofyear <= 300) & (da['time'].dt.dayofyear >= 100)), other=da['time'])
            # print(da['time'])
        return t
    
    
    # import pdb; pdb.set_trace()
    
    
    # print(ds.attrs['format'])
    # if ds.attrs['format'] != 'raw':
    ds['time_orig'] = ds['time']
    ds['time'] = time_shift(ds['time'].copy(deep=True))
    _, index = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index)
    
    
    ###
    ### DEBUGGING
    ### 
    import matplotlib.pyplot as plt
    ds['n'] = (('time'), np.arange(ds['time'].size)+1)
    
    # Remove HygroClip temperature offset
    ds['t_2'] = ds['t_2'] - ds.attrs['hygroclip_t_offset']
    
    # convert radiation from engineering to physical units
    ds['dsr'] = (ds['dsr'] * 10) / ds.attrs['dsr_eng_coef']
    ds['usr'] = (ds['usr'] * 10) / ds.attrs['usr_eng_coef']
    ds['dlr'] = ((ds['dlr'] * 10) / ds.attrs['dlr_eng_coef']) + 5.67E-8*(ds['t_rad'] + T_0)**4
    ds['ulr'] = ((ds['ulr'] * 10) / ds.attrs['ulr_eng_coef']) + 5.67E-8*(ds['t_rad'] + T_0)**4
    
    # Adjust sonic ranger readings for sensitivity to air temperature
    ds['z_boom'] = ds['z_boom'] * ((ds['t_1'] + T_0)/T_0)**0.5
    ds['z_stake'] = ds['z_stake'] * ((ds['t_1'] + T_0)/T_0)**0.5
    
    # Adjust pressure transducer due to fluid properties
    if ~ds['z_pt'].isnull().all():
        ds['z_pt'] = ds['z_pt'] * ds.attrs['pt_z_coef'] * ds.attrs['pt_z_factor'] * 998.0 / rho_af
    
        # Calculate pressure transducer depth
        ds['z_pt_corr'] = ds['z_pt'] * np.nan # new 'z_pt_corr' copied from 'z_pt'
        ds['z_pt_corr'].attrs['long_name'] = ds['z_pt'].long_name + " corrected"
        ds['z_pt_corr'] = ds['z_pt'] * ds.attrs['pt_z_coef'] * ds.attrs['pt_z_factor'] * 998.0 / rho_af \
            + 100 * (ds.attrs['pt_z_p_coef'] - ds['p']) / (rho_af * 9.81)
    
    
    # Decode GPS
    if ds['gps_lat'].dtype.kind == 'O': # not a float. Probably has "NH"
        assert('NH' in ds['gps_lat'].dropna(dim='time').values[0])
        for v in ['gps_lat','gps_lon','gps_time']:
            a = ds[v].attrs # store
            str2nums = [re.findall(r"[-+]?\d*\.\d+|\d+", _) if isinstance(_, str) else [np.nan] for _ in ds[v].values]
            ds[v][:] = pd.DataFrame(str2nums).astype(float).T.values[0]
            ds[v] = ds[v].astype(float)
            ds[v].attrs = a # restore
            
    if np.any((ds['gps_lat'] <= 90) & (ds['gps_lat'] > 0)):  # Some stations only recorded minutes, not degrees
        xyz = np.array(re.findall("[-+]?[\d]*[.][\d]+", ds.attrs['geometry'])).astype(float)
        x=xyz[0]; y=xyz[1]; z=xyz[2] if len(xyz) == 3 else 0
        p = shapely.geometry.Point(x,y,z)
        ds['gps_lat'] = ds['gps_lat'] + 100*p.y
    if np.any((ds['gps_lon'] <= 90) & (ds['gps_lon'] > 0)):
        ds['gps_lon'] = ds['gps_lon'] + 100*p.x
            
    for v in ['gps_lat','gps_lon']:
        a = ds[v].attrs # store
        ds[v] = np.floor(ds[v] / 100) + (ds[v] / 100 - np.floor(ds[v] / 100)) * 100 / 60
        ds[v].attrs = a # restore
    
    # tilt-o-meter voltage to degrees
    # if transmitted ne 'yes' then begin
    #    tiltX = smooth(tiltX,7,/EDGE_MIRROR,MISSING=-999) & tiltY = smooth(tiltY,7,/EDGE_MIRROR, MISSING=-999)
    # endif
    
    # Should just be
    # if ds.attrs['format'] != 'TX': dstxy = dstxy.rolling(time=7, win_type='boxcar', center=True).mean()
    # but the /EDGE_MIRROR makes it a bit more complicated...
    if ds.attrs['format'] != 'TX':
        win_size=7
        s = int(win_size/2)
        tdf = ds['tilt_x'].to_dataframe()
        ds['tilt_x'] = (('time'), tdf.iloc[:s][::-1].append(tdf).append(tdf.iloc[-s:][::-1]).rolling(win_size, win_type='boxcar', center=True).mean()[s:-s].values.flatten())
        tdf = ds['tilt_y'].to_dataframe()
        ds['tilt_y'] = (('time'), tdf.iloc[:s][::-1].append(tdf).append(tdf.iloc[-s:][::-1]).rolling(win_size, win_type='boxcar', center=True).mean()[s:-s].values.flatten())
    
    # # notOKtiltX = where(tiltX lt -100, complement=OKtiltX) & notOKtiltY = where(tiltY lt -100, complement=OKtiltY)
    notOKtiltX = (ds['tilt_x'] < -100)
    OKtiltX = (ds['tilt_x'] >= -100)
    notOKtiltY = (ds['tilt_y'] < -100)
    OKtiltY = (ds['tilt_y'] >= -100)
    
    # tiltX = tiltX/10.
    ds['tilt_x'] = ds['tilt_x'] / 10
    ds['tilt_y'] = ds['tilt_y'] / 10
    
    # tiltnonzero = where(tiltX ne 0 and tiltX gt -40 and tiltX lt 40)
    # if n_elements(tiltnonzero) ne 1 then tiltX[tiltnonzero] = tiltX[tiltnonzero]/abs(tiltX[tiltnonzero])*(-0.49*(abs(tiltX[tiltnonzero]))^4 + 3.6*(abs(tiltX[tiltnonzero]))^3 - 10.4*(abs(tiltX[tiltnonzero]))^2 +21.1*(abs(tiltX[tiltnonzero])))
    
    # tiltY = tiltY/10.
    # tiltnonzero = where(tiltY ne 0 and tiltY gt -40 and tiltY lt 40)
    # if n_elements(tiltnonzero) ne 1 then tiltY[tiltnonzero] = tiltY[tiltnonzero]/abs(tiltY[tiltnonzero])*(-0.49*(abs(tiltY[tiltnonzero]))^4 + 3.6*(abs(tiltY[tiltnonzero]))^3 - 10.4*(abs(tiltY[tiltnonzero]))^2 +21.1*(abs(tiltY[tiltnonzero])))
    
    dstx = ds['tilt_x']
    nz = (dstx != 0) & (np.abs(dstx) < 40)
    dstx = dstx.where(~nz, other = dstx / np.abs(dstx) * (-0.49 * (np.abs(dstx))**4 + 3.6 * (np.abs(dstx))**3 - 10.4 * (np.abs(dstx))**2 + 21.1 * (np.abs(dstx))))
    ds['tilt_x'] = dstx
    
    dsty = ds['tilt_y']
    nz = (dsty != 0) & (np.abs(dsty) < 40)
    dsty = dsty.where(~nz, other = dsty / np.abs(dsty) * (-0.49 * (np.abs(dsty))**4 + 3.6 * (np.abs(dsty))**3 - 10.4 * (np.abs(dsty))**2 + 21.1 * (np.abs(dsty))))
    ds['tilt_y'] = dsty
    
    # if n_elements(OKtiltX) gt 1 then tiltX[notOKtiltX] = interpol(tiltX[OKtiltX],OKtiltX,notOKtiltX) ; Interpolate over gaps for radiation correction; set to -999 again below.
    # if n_elements(OKtiltY) gt 1 then tiltY[notOKtiltY] = interpol(tiltY[OKtiltY],OKtiltY,notOKtiltY) ; Interpolate over gaps for radiation correction; set to -999 again below.
    
    ds['tilt_x'] = ds['tilt_x'].where(~notOKtiltX)
    ds['tilt_y'] = ds['tilt_y'].where(~notOKtiltY)
    ds['tilt_x'] = ds['tilt_x'].interpolate_na(dim='time')
    ds['tilt_y'] = ds['tilt_y'].interpolate_na(dim='time')
    # ds['tilt_x'] = ds['tilt_x'].ffill(dim='time')
    # ds['tilt_y'] = ds['tilt_y'].ffill(dim='time')
    
    
    deg2rad = np.pi / 180
    ds['wdir'] = ds['wdir'].where(ds['wspd'] != 0)
    ds['wspd_x'] = ds['wspd'] * np.sin(ds['wdir'] * deg2rad)
    ds['wspd_y'] = ds['wspd'] * np.cos(ds['wdir'] * deg2rad)
    infile = ds.attrs['file']
    outpath = os.path.split(infile)[0].split("/")
    outpath[-2] = 'L1'
    outpath = '/'.join(outpath)
    outfile = os.path.splitext(os.path.splitext(os.path.basename(infile))[0])[0]
    
    outpathfile = outpath + '/' + outfile + ".nc"
    if os.path.exists(outpathfile): os.remove(outpathfile)
    ds.to_netcdf(outpathfile, mode='w', format='NETCDF4', compute=True)

import toml
import os

def load_conf(filename, L0_path):
    """Load a TOML file
    PROMICE TOML supports defining features at the top level which apply to all nested properties,
    but do not overwrite nested properties if they are defined.
    """
    conf = toml.load(filename)
    # Move all top level keys to nested properties,
    # if they are not already defined in the nested properties
    # Also, insert the section name (filename) as a file property, and configuration file.
    top = [_ for _ in conf.keys() if not type(conf[_]) is dict]
    subs = [_ for _ in conf.keys() if type(conf[_]) is dict]
    for s in subs:
        for t in top:
            if t not in conf[s].keys():
                conf[s][t] = conf[t]

        conf[s]['conf'] = filename
        conf[s]['file'] = os.path.join(L0_path, conf[s]['station_id'], s)

    # Delete all top level keys, because each file (sub-level) should carry all
    # properties with it.
    for t in top: conf.pop(t)

    # check required fields are present
    for k in conf.keys():
        for field in ["columns", "station_id", "format", "latitude", "longitude", "skiprows"]:
            assert(field in conf[k].keys())

    return conf

if __name__ == "__main__":
    import sys
    assert(len(sys.argv) == 3)
    L0_path = sys.argv[1]
    conf = load_conf(sys.argv[2], L0_path)
    for k in conf.keys():
        if 'Slim' in k: continue
        # if 'transmitted' in k: continue
        L0_to_L1(conf[k])
