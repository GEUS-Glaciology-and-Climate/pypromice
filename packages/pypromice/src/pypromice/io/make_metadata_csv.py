#!/usr/bin/env python
import os, sys, argparse
import pandas as pd
import xarray as xr
import logging

logging.basicConfig(
    format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def extract_metadata_from_nc(file_path: str, data_type: str, label_s_id: str) -> pd.Series:
    """
    Extract metadata from a NetCDF file and return it as a pandas Series.

    Parameters:
    - file_path (str): The path to the NetCDF file.
    - data_type (str): The type of data ('station' or 'site').
    - label_s_id (str): The label for the station or site ID.

    Returns:
    - pd.Series: A pandas Series containing the extracted metadata.
    """
    try:
        with xr.open_dataset(file_path) as nc_file:
            # Extract attributes
            s_id = nc_file.attrs.get(label_s_id, 'N/A')
            location_type = nc_file.attrs.get('location_type', 'N/A')
            project = nc_file.attrs.get('project', 'N/A')
            if data_type == 'site':
                stations = nc_file.attrs.get('stations', s_id)
            if data_type == 'station':
                number_of_booms = nc_file.attrs.get('number_of_booms', 'N/A')
                
            # Extract the time variable as datetime64
            time_var = nc_file['time'].values.astype('datetime64[s]')

            # Extract the first and last timestamps
            date_installation_str = pd.Timestamp(time_var[0]).strftime('%Y-%m-%d')
            last_valid_date_str = pd.Timestamp(time_var[-1]).strftime('%Y-%m-%d')

            # Extract the first and last values of lat, lon, and alt
            lat_installation = nc_file['lat'].isel(time=0).values.item()
            lon_installation = nc_file['lon'].isel(time=0).values.item()
            alt_installation = nc_file['alt'].isel(time=0).values.item()

            lat_last_known = nc_file['lat'].isel(time=-1).values.item()
            lon_last_known = nc_file['lon'].isel(time=-1).values.item()
            alt_last_known = nc_file['alt'].isel(time=-1).values.item()

            # Create a pandas Series for the metadata
            if data_type == 'site':
                row = pd.Series({
                    'project': project.replace('\r',''),
                    'location_type': location_type,
                    'stations': stations,
                    'date_installation': date_installation_str,
                    'latitude_installation': lat_installation,
                    'longitude_installation': lon_installation,
                    'altitude_installation': alt_installation,
                    'date_last_valid': last_valid_date_str,
                    'latitude_last_valid': lat_last_known,
                    'longitude_last_valid': lon_last_known,
                    'altitude_last_valid': alt_last_known
                }, name=s_id)
            else:
                row = pd.Series({
                    'project': project.replace('\r',''),
                    'number_of_booms': number_of_booms,
                    'location_type': location_type,
                    'date_installation': date_installation_str,
                    'latitude_installation': lat_installation,
                    'longitude_installation': lon_installation,
                    'altitude_installation': alt_installation,
                    'date_last_valid': last_valid_date_str,
                    'latitude_last_valid': lat_last_known,
                    'longitude_last_valid': lon_last_known,
                    'altitude_last_valid': alt_last_known
                }, name=s_id)
            return row
    except Exception as e:
        logger.info(f"Warning: Error processing {file_path}: {str(e)}")
        return pd.Series()  # Return an empty Series in case of an error

def process_files(base_dir: str, csv_file_path: str, data_type: str) -> pd.DataFrame:
    """
    Process all files in the base directory to generate new metadata.

    Parameters:
    - base_dir (str): The base directory containing the NetCDF files.
    - csv_file_path (str): The path to the existing metadata CSV file.
    - data_type (str): The type of data ('station' or 'site').

    Returns:
    - pd.DataFrame: The combined metadata DataFrame.
    """
    label_s_id = 'station_id' if data_type == 'station' else 'site_id'

    # Initialize a list to hold the rows (Series) of DataFrame
    rows = []

    # Read existing metadata if the CSV file exists
    if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
        logger.info("Updating " + str(csv_file_path))
        existing_metadata_df = pd.read_csv(csv_file_path, index_col=label_s_id)
    else:
        logger.info("Creating " + str(csv_file_path))
        existing_metadata_df = pd.DataFrame()

    # Track updated sites or stations to avoid duplicate updates
    updated_s = []
    new_s = []

    # Traverse through all the subfolders and files in the base directory
    for subdir, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_hour.nc'):
                file_path = os.path.join(subdir, file)
                row = extract_metadata_from_nc(file_path, data_type, label_s_id)
                if not row.empty:
                    s_id = row.name
                    if s_id in existing_metadata_df.index:
                        # Compare with existing metadata
                        existing_row = existing_metadata_df.loc[s_id]
                        old_date_installation = existing_row['date_installation']
                        old_last_valid_date = existing_row['date_last_valid']

                        # Update the existing metadata
                        existing_metadata_df.loc[s_id] = row

                        # Print message if dates are updated
                        if old_last_valid_date != row['date_last_valid']:
                            logger.info(f"Updated {label_s_id}: {s_id} date_last_valid: {old_last_valid_date} --> {row['date_last_valid']}")

                        updated_s.append(s_id)
                    else:
                        new_s.append(s_id)
                        # Append new metadata row to the list
                        rows.append(row)

    # Convert the list of rows to a DataFrame
    new_metadata_df = pd.DataFrame(rows)

    # Concatenate the existing metadata with the new metadata
    combined_metadata_df = pd.concat([existing_metadata_df, new_metadata_df], ignore_index=False)

    # Exclude some sites
    sites_to_exclude = [s for s in ['XXX', 'Roof_GEUS', 'Roof_PROMICE'] if s in combined_metadata_df.index]
    excluded_metadata_df = combined_metadata_df.loc[sites_to_exclude].copy()
    combined_metadata_df.drop(sites_to_exclude, inplace=True)

    # Sort the DataFrame by index (s_id)
    combined_metadata_df.sort_index(inplace=True)

    # Print excluded lines
    if not excluded_metadata_df.empty:
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.max_colwidth', None) # Show full width of columns
        pd.set_option('display.width', None)        # Disable line wrapping
        logger.info("\nExcluded lines from combined metadata.csv:")
        print(excluded_metadata_df)

    # Drop excluded lines from combined_metadata_df
    combined_metadata_df.drop(sites_to_exclude, errors='ignore', inplace=True)

    # Save to csv
    combined_metadata_df.to_csv(csv_file_path, index_label=label_s_id)

    return combined_metadata_df, existing_metadata_df, new_s, updated_s

def compare_and_log_updates(combined_metadata_df: pd.DataFrame, existing_metadata_df: pd.DataFrame, new_s: list, updated_s: list):
    """
    Compare the combined metadata with the existing metadata and log the updates.

    Parameters:
    - combined_metadata_df (pd.DataFrame): The combined metadata DataFrame.
    - existing_metadata_df (pd.DataFrame): The existing metadata DataFrame.
    - new_s (list): List of new station/site IDs.
    - updated_s (list): List of updated station/site IDs.
    """
    # Determine which lines were not updated (reused) and which were added
    if not existing_metadata_df.empty:
        reused_s = [s_id for s_id in existing_metadata_df.index if ((s_id not in new_s) & (s_id not in updated_s))]
        reused_lines = existing_metadata_df.loc[reused_s]
        added_lines = combined_metadata_df.loc[combined_metadata_df.index.difference(existing_metadata_df.index)]
        
        logger.info("\nLines from the old metadata.csv that are reused (not updated):")
        print(reused_lines)
        
        if not added_lines.empty:
            logger.info("\nLines that were not present in the old metadata.csv and are added:")
            print(added_lines)
    else:
        logger.info("\nAll lines are added (no old metadata.csv found)")

def main():
    parser = argparse.ArgumentParser(description='Process station or site data.')
    parser.add_argument('-t', '--type', choices=['station', 'site'], 
                        required=True, 
                        help='Type of data to process: "station" or "site"')
    parser.add_argument('-r', '--root_dir', required=True, help='Root directory ' +
                        'containing the aws-l3 station or site folder')
    parser.add_argument('-m','--metadata_file', required=True,
                        help='File path to metadata csv file (existing or '+
                        'intended output path')
    
    args = parser.parse_args()
    combined_metadata_df, existing_metadata_df, new_s, updated_s = process_files(args.root_dir, args.metadata_file, args.type)
    compare_and_log_updates(combined_metadata_df, existing_metadata_df, new_s, updated_s)

if __name__ == '__main__':
    main()
