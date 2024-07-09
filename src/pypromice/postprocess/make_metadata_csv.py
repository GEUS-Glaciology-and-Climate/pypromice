#!/usr/bin/env python
import os
import argparse
import pandas as pd
import xarray as xr
import logging
logger = logging.getLogger(__name__)
        
def process_files(base_dir, csv_file_path, data_type):
    
    # Determine the CSV file path based on the data type
    if data_type == 'station':
        label_s_id = 'station_id'
    elif data_type == 'site':
        label_s_id = 'site_id'

    # Initialize a list to hold the rows (Series) of DataFrame
    rows = []

    # Read existing metadata if the CSV file exists
    if os.path.exists(csv_file_path):
        logger.info("Updating "+str(csv_file_path))
        existing_metadata_df = pd.read_csv(csv_file_path, index_col=label_s_id)
    else:
        logger.info("Creating "+str(csv_file_path))
        existing_metadata_df = pd.DataFrame()

    # Drop the 'timestamp_last_known_coordinates' column if it exists
    if 'timestamp_last_known_coordinates' in existing_metadata_df.columns:
        existing_metadata_df.drop(columns=['timestamp_last_known_coordinates'], inplace=True)

    # Track updated sites or stations to avoid duplicate updates
    updated_s = []
    new_s = []

    # Traverse through all the subfolders and files in the base directory
    for subdir, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_hour.nc'):
                file_path = os.path.join(subdir, file)
                try:
                    with xr.open_dataset(file_path) as nc_file:
                        # Extract attributes
                        s_id = nc_file.attrs.get(label_s_id, 'N/A')

                        number_of_booms = nc_file.attrs.get('number_of_booms', 'N/A')
                        if number_of_booms == '1':
                            station_type = 'one boom'
                        elif number_of_booms == '2':
                            station_type = 'two booms'
                        else:
                            station_type = 'N/A'

                        # Keep the existing location_type if it exists
                        if s_id in existing_metadata_df.index:
                            location_type = existing_metadata_df.loc[s_id, 'location_type']
                        else:
                            location_type = nc_file.attrs.get('location_type', 'N/A')

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
                        row = pd.Series({
                            'station_type': station_type,
                            'location_type': location_type,
                            'date_installation': date_installation_str,
                            'last_valid_date': last_valid_date_str,
                            'lat_installation': lat_installation,
                            'lon_installation': lon_installation,
                            'alt_installation': alt_installation,
                            'lat_last_known': lat_last_known,
                            'lon_last_known': lon_last_known,
                            'alt_last_known': alt_last_known
                        }, name=s_id)

                        # Check if this s_id is already in the existing metadata
                        if s_id in existing_metadata_df.index:
                            # Compare with existing metadata
                            existing_row = existing_metadata_df.loc[s_id]
                            old_date_installation = existing_row['date_installation']
                            old_last_valid_date = existing_row['last_valid_date']

                            # Update the existing metadata
                            existing_metadata_df.loc[s_id] = row

                            # Print message if dates are updated
                            if old_date_installation != date_installation_str or old_last_valid_date != last_valid_date_str:
                                logger.info(f"Updated {label_s_id}: {s_id}")
                                logger.info(f"  Old date_installation: {old_date_installation} --> New date_installation: {date_installation_str}")
                                logger.info(f"  Old last_valid_date: {old_last_valid_date} --> New last_valid_date: {last_valid_date_str}")

                            updated_s.append(s_id)
                        else:
                            new_s.append(s_id)
                            # Append new metadata row to the list
                            rows.append(row)

                except Exception as e:
                    logger.info(f"Warning: Error processing {file_path}: {str(e)}")
                    continue  # Continue to next file if there's an error

    # Convert the list of rows to a DataFrame
    new_metadata_df = pd.DataFrame(rows)

    # Convert the list of excluded rows to a DataFrame

    # Concatenate the existing metadata with the new metadata and excluded metadata
    combined_metadata_df = pd.concat([existing_metadata_df, new_metadata_df], ignore_index=False)

    # excluding some sites
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

    if label_s_id == 'site_id':
        combined_metadata_df.drop(columns=['station_type'], inplace=True)
        
    # saving to csv
    combined_metadata_df.to_csv(csv_file_path, index_label=label_s_id)
    
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
    process_files(args.root_dir, args.metadata_file, args.type)

if __name__ == '__main__':
    main()
