#!/usr/bin/env python
import logging, os, sys, unittest
from argparse import ArgumentParser
import pypromice
from pypromice.process.aws import AWS

old_name = {
                'CEN2': ['GITS'],
                'CP1': ['CrawfordPoint1'],
                'DY2': ['DYE-2'],
                'JAR': ['JAR1'],
                'HUM': ['Humboldt'],
                'NAU': ['NASA-U'],
                'NAE': ['NASA-E'],
                'NEM': ['NEEM'],
                'NSE': ['NASA-SE'],
                'EGP': ['EastGRIP'],
                'SDL': ['Saddle'],
                'SDM': ['SouthDome'],
                'SWC': ['SwissCamp', 'SwissCamp10m'],
                'TUN': ['Tunu-N'],
                }

def parse_arguments_getl3(debug_args=None):
    parser = ArgumentParser(description="AWS L3 script for the processing L3 data from L2 and merging the L3 data with its historical site. An hourly, daily and monthly L3 data product is outputted to the defined output path")
    parser.add_argument('-s', '--file1', type=str, required=True, nargs='+',
                        help='Path to source L2 file')
    parser.add_argument('-g', '--folder_gcnet', type=str, required=True, nargs='+',
                        help='Path to GC-Net historical L1 folder')
    parser.add_argument('-p', '--folder_l2', type=str, required=True, nargs='+',
                        help='Path to level 2 folder')
    parser.add_argument('-o', '--outpath', default=os.getcwd(), type=str, required=True,
                        help='Path where to write output')
    parser.add_argument('-v', '--variables', default=None, type=str, required=False, 
    			 help='Path to variables look-up table .csv file for variable name retained'''),
    parser.add_argument('-m', '--metadata', default=None, type=str, required=False, 
    			 help='Path to metadata table .csv file for metadata information'''),
    parser.add_argument('-d', '--datatype', default='raw', type=str, required=False, 
    			 help='Data type to output, raw or tx')
    args = parser.parse_args(args=debug_args)
    args.file1 = ' '.join(args.file1)
    args.folder_gcnet = ' '.join(args.folder_gcnet)
    args.folder_l2 = ' '.join(args.folder_l2)
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
    
    try:
        name = ds.attrs['station_name'] 
    except:
        name = infile.split('/')[-1].split('.')[0].split('_hour')[0].split('_10min')[0]
        
    print(f'{name} array loaded from {infile}')
    return ds, name


def gcnet_postprocessing(ds2):
    file_path = pkg_resources.resource_stream('pypromice', 'process/variable_aliases_GC-Net.csv')
     
    var_name = pd.read_csv(file_path)
    var_name = var_name.set_index('old_name').GEUS_name
    msk = [v for v in var_name.index if v in ds2.data_vars]
    var_name = var_name.loc[msk].to_dict()
    
    ds2['TA1'] =  ds2['TA1'].combine_first(ds2['TA3'])
    ds2['TA2'] =  ds2['TA2'].combine_first(ds2['TA4'])
    
    ds2=ds2.rename(var_name)
    ds2=ds2.rename({'timestamp':'time'})
    return ds2

# will be used in the future
# def aligning_surface_heights(ds1, ds2):
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
    # return ds1

def get_l3():
    args = parse_arguments_getl3()
                    
    # Check files    
    if os.path.isfile(args.file1): 
        # Load L2 data arrays
        ds1, n1 = loadArr(args.file1)
        
        # converts to L3:
        # - derives sensible heat fluxes
        # - appends historical data
        
        ds1 = toL3(ds1)
        
        if n1 in old_name.keys():
            n2 = old_name[n1]
            # for all secondary stations
            for n in n2:
                # loading secondary file, either from GC-Net or PROMICE folders
                file2 = args.folder_gcnet+n+'.csv'
                is_gcnet = True
                if not os.path.isfile(file2):
                    file2 = args.folder_l2+'/'+n+'/'+n+'_hour.csv'
                    if not os.path.isfile(file2):
                        print('could not find',n, 'in',args.folder_gcnet,'nor',args.folder_l2)
                        continue
                    else:
                        is_gcnet = False
    
                ds2, n2 = loadArr(file2)
                
                if is_gcnet:
                    ds2 = gcnet_postprocessing(ds2)
                    ds2 = ds2[[v for v in ds1.data_vars if v in ds2.data_vars]]
                else:
                    # then it is a GEUS L2 file that needs to be processed into L3
                    # converts to L3
                    ds2 = toL3(ds2)
                
                # prepairing for the merging
                for v in ds1.data_vars:
                    if  v not in ds2.data_vars:
                        ds2[v] = ds2.t_u*np.nan
                        
                print('dropping')
                for v in ds2.data_vars:
                    if  v not in ds1.data_vars:
                        print(v)
                        ds2 = ds2.drop(v)

                # merging by time block
                ds1 = xr.concat((ds2.sel(
                            time=slice(ds2.time.isel(time=0),
                                       ds1.time.isel(time=0))
                            ), ds1), dim='time')

    elif os.path.isfile(args.file1):  
        ds1, name = loadArr(args.file1)
        print(f'No historical station for {n1}...')
    
    else:
        print(f'Invalid file {args.file1}')
        exit()
    all_values = []
    for l in old_name.values():
        for ll in l:
            all_values.append(ll)
    if n1 in all_values:
        print(n1, 'is used as auxilary at another site')
        exit()
    # Get hourly, daily and monthly datasets
    print('Resampling L3 data to hourly, daily and monthly resolutions...')
    l3_h = resampleL2(ds1, '60min')
    l3_d = resampleL2(ds1, '1D')
    l3_m = resampleL2(ds1, 'M')
    
    print(f'Adding variable information from {args.variables}...')
        
    # Load variables look-up table
    var = getVars(args.variables)
        	
    # Round all values to specified decimals places
    l3_h = roundValues(l3_h, var)
    l3_d = roundValues(l3_d, var)
    l3_m = roundValues(l3_m, var)
        
    # Get columns to keep
    if hasattr(ds1, 'p_l'):
        col_names = getColNames(var, 2, args.datatype.lower())  
    else:
        col_names = getColNames(var, 1, args.datatype.lower())    

    # Assign site id
    site_id = n1.replace('v3','').replace('CEN2','CEN')
    for l in [l3_h, l3_d, l3_m]:
        l.attrs['site_id'] = site_id
        l.attrs['station_id'] = site_id
        if n1 in old_name.keys():
            l.attrs['list_station_id'] = '('+n1+', '+', '.join(old_name[n1])+')'
        else:
            l.attrs['list_station_id'] = '('+n1+')'
    
    # Define input path
    station_name = args.config_file.split('/')[-1].split('.')[0] 
    station_path = os.path.join(args.inpath, station_name)
    if os.path.exists(station_path):
        aws = AWS(args.config_file, station_path, v, m)
    else:
        aws = AWS(args.config_file, args.inpath, v, m)

    # Perform level 1 to 3 processing
    aws.getL1()
    aws.getL2()
    aws.getL3()
    
    # Write out Level 3
    if args.outpath is not None:
    	aws.writeL3(args.outpath)
        
if __name__ == "__main__":  
    get_l3()
        
