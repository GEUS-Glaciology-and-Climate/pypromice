#!/usr/bin/env python
import logging, os, sys, unittest, toml, pkg_resources
from argparse import ArgumentParser
from pypromice.process.write import prepare_and_write
import numpy as np
import pandas as pd
import xarray as xr

def parse_arguments_joinl3(debug_args=None):
    parser = ArgumentParser(description="AWS L3 script for the processing L3 data from L2 and merging the L3 data with its historical site. An hourly, daily and monthly L3 data product is outputted to the defined output path")
    parser.add_argument('-c', '--config_folder', type=str, required=True,
                        help='Path to folder with sites configuration (TOML) files')
    parser.add_argument('-s', '--site',  default=None, type=str, required=False,
                        help='Name of site to process (default: all sites are processed)')

    parser.add_argument('-l3', '--folder_l3', type=str, required=True, 
                        help='Path to level 3 folder')
    parser.add_argument('-gc', '--folder_gcnet', type=str, required=False, 
                        help='Path to GC-Net historical L1 folder')

    parser.add_argument('-o', '--outpath', default=os.getcwd(), type=str, required=True,
                        help='Path where to write output')
    
    parser.add_argument('-v', '--variables', default=None, type=str, required=False, 
    			 help='Path to variables look-up table .csv file for variable name retained'''),
    parser.add_argument('-m', '--metadata', default=None, type=str, required=False, 
    			 help='Path to metadata table .csv file for metadata information'''),
    parser.add_argument('-d', '--datatype', default='raw', type=str, required=False, 
    			 help='Data type to output, raw or tx')
    args = parser.parse_args(args=debug_args)
    return args

def readNead(infile):
    with open(infile) as f:
        fmt = f.readline()
        assert(fmt[0] == "#")
        assert(fmt.split("#")[1].split()[0] == "NEAD")
        assert(fmt.split("#")[1].split()[1] == "1.0")
        assert(fmt.split("#")[1].split()[2] == "UTF-8")
        
        line = f.readline()
        assert(line[0] == "#")
        assert(line.split("#")[1].strip() == '[METADATA]')
    
        meta = {}
        fields = {}
        section = 'meta'
        while True:
            line = f.readline()
            if line.strip(' ') == '#': continue
            if line == "# [DATA]\n": break # done reading header
            if line == "# [FIELDS]\n":
                section = 'fields'
                continue # done reading header
            
            if line[0] == "\n": continue   # blank line
            assert(line[0] == "#")         # if not blank, must start with "#"
            
            key_eq_val = line.split("#")[1].strip()
            if key_eq_val == '' or key_eq_val == None: continue  # Line is just "#" or "# " or "#   #"...
            assert("=" in key_eq_val), print(line, key_eq_val)
            key = key_eq_val.split("=")[0].strip()
            val = key_eq_val.split("=")[1].strip()
    
            # Convert from string to number if it is a number
            if val.strip('-').strip('+').replace('.','').isdigit():
                val = float(val)
                if val == int(val):
                    val = int(val)
    
            if section == 'meta': meta[key] = val
            if section == 'fields': fields[key] = val
        # done reading header
    
        # Find delimiter and fields for reading NEAD as simple CSV
        assert("field_delimiter" in meta.keys())
        assert("fields" in fields.keys())
        FD = meta["field_delimiter"]
        names = [_.strip() for _ in fields.pop('fields').split(FD)]
    
        df = pd.read_csv(infile,
                         comment = "#",
                         names = names,
                         sep = FD,
                         usecols=np.arange(len(names)),
                         skip_blank_lines = True)
        df['timestamp'] = pd.to_datetime(df.timestamp).dt.tz_localize(None)
        df = df.set_index('timestamp')
        ds = df.to_xarray()
        ds.attrs = meta   
    return ds


def loadArr(infile):
    if infile.split('.')[-1].lower() in 'csv':
        with open(infile) as f:
            text_splitted = f.read().splitlines()
            first_char =  text_splitted[0][0]
            
        if first_char != '#':
            df = pd.read_csv(infile)
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
            df = df.set_index('time')
            ds = xr.Dataset.from_dataframe(df)
        else:
            ds = readNead(infile)
        f.close()
    elif infile.split('.')[-1].lower() in 'nc':
        ds = xr.open_dataset(infile)
        for varname in ds.variables:
            if 'encoding' in ds[varname].attrs:
                del ds[varname].attrs['encoding']
    
    try:
        name = ds.attrs['station_name'] 
    except:
        name = infile.split('/')[-1].split('.')[0].split('_hour')[0].split('_10min')[0]
        
    print(f'{name} array loaded from {infile}')
    return ds, name


def gcnet_postprocessing(l3):
    file_path = pkg_resources.resource_stream('pypromice', 'ressources/variable_aliases_GC-Net.csv')
     
    var_name = pd.read_csv(file_path)
    var_name = var_name.set_index('old_name').GEUS_name
    msk = [v for v in var_name.index if v in l3.data_vars]
    var_name = var_name.loc[msk].to_dict()
    
    l3['TA1'] =  l3['TA1'].combine_first(l3['TA3'])
    l3['TA2'] =  l3['TA2'].combine_first(l3['TA4'])
    
    l3=l3.rename(var_name)
    l3=l3.rename({'timestamp':'time'})
    return l3

# will be used in the future
# def aligning_surface_heights(l3m, l3):
    # df_aux['z_surf_combined'] = \
    #     df_aux['z_surf_combined'] \
    #         - df_aux.loc[df_aux.z_surf_combined.last_valid_index(), 'z_surf_combined'] \
    #          + df_v6.loc[df_v6.z_surf_combined.first_valid_index(), 'z_surf_combined'] 
    
    # if s == 'Swiss Camp 10m':        
    #     df.loc[:df.HS_combined.first_valid_index(), 'HS_combined'] = \
    #         df2.loc[:df.HS_combined.first_valid_index(), 'HS_combined'] \
    #             - df2.loc[df2.HS_combined.last_valid_index(), 'HS_combined'] \
    #              + df.loc[df.HS_combined.first_valid_index(), 'HS_combined'] 

                    
    # df.loc[df.HS_combined.diff()==0,'HS_combined'] = np.nan
    
    # fit = np.polyfit(df.loc[df.HS_combined.notnull(),:].index.astype('int64'),  
    #                    df.loc[df.HS_combined.notnull(),'HS_combined'], 1)
    # fit_fn = np.poly1d(fit)
    
    # df['HS_combined'] = df['HS_combined'].values \
    #     - fit_fn(
    #         df_in.loc[[df_in.z_surf_combined.first_valid_index()],:].index.astype('int64')[0]
    #     )  +  df_in.loc[df_in.z_surf_combined.first_valid_index(), 'z_surf_combined']
    # return l3m

def join_l3():
    args = parse_arguments_joinl3()
                  
                      
    config_file = os.path.join(args.config_folder, args.site+'.toml')
    conf = toml.load(config_file)

    l3m = xr.Dataset()
    l3m.attrs['level'] = 'L3'
    for stid in conf['stations']:
        print(stid)
        
        is_promice = False
        is_gcnet = False
        filepath = os.path.join(args.folder_l3, stid, stid+'_hour.nc')
        if os.path.isfile(filepath):
            is_promice = True
        else:
            filepath = os.path.join(args.folder_gcnet, stid+'.csv')
            if os.path.isfile(filepath):
                is_gcnet = True
        if not is_promice and not is_gcnet:            
            print(stid, 'not found either in', args.folder_l3, 'or', args.folder_gcnet)
            continue

        l3, _ = loadArr(filepath)
        
        if is_gcnet:
            l3 = gcnet_postprocessing(l3)
        
        # lat, lon and alt should be just variables, not coordinates
        if 'lat' in l3.keys():
            l3 = l3.reset_coords(['lat', 'lon', 'alt'])

        if len(l3m)==0:
            # saving attributes of station under an attribute called $stid
            l3m.attrs[stid] = l3.attrs.copy()
            # then stripping attributes
            attrs_list = list(l3.attrs.keys())
            for k in attrs_list:
                del l3.attrs[k]
                
            l3m = l3.copy()
        else:
            # if l3 (older data) is missing variables compared to l3m (newer data)
            # , then we fill them with nan
            for v in l3m.data_vars:
                if  v not in l3.data_vars:
                    l3[v] = l3.t_u*np.nan
                    
            # if l3 (older data) has variables that does not have l3m (newer data)
            # then they are removed from l3
            print('dropping')
            for v in l3.data_vars:
                if v not in l3m.data_vars:
                    if v != 'z_stake':
                       print(v)
                       l3 = l3.drop(v)
                    else:
                       l3m[v] = ('time', l3m.t_u.data*np.nan)
                        
            # saving attributes of station under an attribute called $stid
            l3m = l3m.assign_attrs({stid : l3.attrs.copy()})
            # then stripping attributes
            attrs_list = list(l3.attrs.keys())
            for k in attrs_list:
                del l3.attrs[k]
            
            l3m.attrs[stid]['first_timestamp'] = l3.time.isel(time=0).dt.strftime( date_format='%Y-%m-%d %H:%M:%S').item()
            l3m.attrs[stid]['last_timestamp'] = l3m.time.isel(time=0).dt.strftime( date_format='%Y-%m-%d %H:%M:%S').item()

            # merging by time block
            l3m = xr.concat((l3.sel(
                        time=slice(l3.time.isel(time=0),
                                   l3m.time.isel(time=0))
                        ), l3m), dim='time')
            

    # Assign site id
    l3m.attrs['site_id'] = args.site
    l3m.attrs['stations'] = conf['stations']

    if args.outpath is not None:
        prepare_and_write(l3m, args.outpath, args.variables, args.metadata, '60min')
        prepare_and_write(l3m, args.outpath, args.variables, args.metadata, '1D')
        prepare_and_write(l3m, args.outpath, args.variables, args.metadata, 'M')
        
if __name__ == "__main__":  
    join_l3()
        
