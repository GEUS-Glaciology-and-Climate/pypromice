#!/usr/bin/env python
import logging, sys, toml
from pathlib import Path

import xarray as xr
from argparse import ArgumentParser
import pypromice
from pypromice.process.L2toL3 import toL3
import pypromice.resources
from pypromice.process.write import prepare_and_write
logger = logging.getLogger(__name__)

def parse_arguments_l2tol3(debug_args=None):
    parser = ArgumentParser(description="AWS L3 script for the processing L3 "+
                            "data from L2. An hourly, daily and monthly L3 "+
                            "data product is outputted to the defined output path")
    parser.add_argument('-c', '--config_folder', type=str, required=True,
                        default='../aws-l0/metadata/station_configurations/',
                        help='Path to folder with sites configuration (TOML) files')
    parser.add_argument('-i', '--inpath', type=str, required=True, 
                        help='Path to Level 2 .nc data file')
    parser.add_argument('-o', '--outpath', default=None, type=str, required=False, 
                        help='Path where to write output')
    parser.add_argument('-v', '--variables', default=None, type=str, 
                        required=False, help='File path to variables look-up table')
    parser.add_argument('-m', '--metadata', default=None, type=str, 
                        required=False, help='File path to metadata')
    parser.add_argument('--data_issues_path', '--issues', default=None, help="Path to data issues repository")


    args = parser.parse_args(args=debug_args)
    return args

def get_l2tol3(config_folder: Path|str, inpath, outpath, variables, metadata, data_issues_path: Path|str):
    if isinstance(config_folder, str):
        config_folder = Path(config_folder)

    logging.basicConfig(
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
  
    # Define Level 2 dataset from file
    with xr.open_dataset(inpath) as l2:
        l2.load()
    
    # Remove encoding attributes from NetCDF
    for varname in l2.variables:
        if l2[varname].encoding!={}:
            l2[varname].encoding = {}

    if 'bedrock' in l2.attrs.keys():
        l2.attrs['bedrock'] = l2.attrs['bedrock'] == 'True'
    if 'number_of_booms' in l2.attrs.keys():
        l2.attrs['number_of_booms'] = int(l2.attrs['number_of_booms'])
    
    # importing station_config (dict) from config_folder (str path)
    config_file = config_folder / (l2.attrs['station_id']+'.toml')

    if config_file.exists():
        # File exists, load the configuration
        station_config = toml.load(config_file)
    else:
        # File does not exist, initialize with standard info 
        # this was prefered by RSF over exiting with error
        logger.error("\n***\nNo station_configuration file for %s.\nPlease create one on AWS-L0/metadata/station_configurations.\n***"%l2.attrs['station_id'])
        station_config = {"stid":l2.attrs['station_id'],
                        "station_site":l2.attrs['station_id'],
                        "project": "PROMICE",
                        "location_type": "ice sheet",
                        }
        
    # checking that the adjustement directory is properly given
    if data_issues_path is None:
        data_issues_path = Path("../PROMICE-AWS-data-issues")
        if data_issues_path.exists():
            logging.warning(f"data_issues_path is missing. Using default data issues path: {data_issues_path}")
        else:
            raise ValueError("data_issues_path is missing. Please provide a valid path to the data issues repository")
    else:
        data_issues_path = Path(data_issues_path)

    data_adjustments_dir = data_issues_path / "adjustments"
    
    # Perform Level 3 processing
    l3 = toL3(l2, data_adjustments_dir, station_config)

    # Write Level 3 dataset to file if output directory given
    v = pypromice.resources.load_variables(variables)
    m = pypromice.resources.load_metadata(metadata)
    if outpath is not None:
        prepare_and_write(l3, outpath, v, m, '60min')
        prepare_and_write(l3, outpath, v, m, '1D')
        prepare_and_write(l3, outpath, v, m, 'M')
    return l3

def main():
    args = parse_arguments_l2tol3()
    


    _ = get_l2tol3(args.config_folder, 
                   args.inpath, 
                   args.outpath,
                   args.variables, 
                   args.metadata, 
                   args.data_issues_path)
    
if __name__ == "__main__":  
    main()
