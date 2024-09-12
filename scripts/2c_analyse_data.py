#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from lipmad import base_conf
from lipmad.image import Image
from lipmad import project
from lipmad import general
from lipmad import io
from lipmad.conversions import convert_mnk_to_kmn

log = base_conf.log

from lipmad.Utils.raster_utils import *
from lipmad.Utils.new_elevation import SRTM

tilename = "S24W071"  # Tile name
cache_file = f'../../data/ANF_DEM_rasters/S24W071.SRTMGL1/{tilename}.hgt'


def main():
    """ Analyse data """
    description = """ Perform image/data analysis"""

    # Create argument parser
    parser = argparse.ArgumentParser(prog='lipmad-georef',
                                     description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional mandatory argument
    parser.add_argument("input_arg", type=str,
                        help="Directory containing sets of aerial image(s).")
    parser.add_argument('-f', '--force-overwrite', action='store_true', dest='force',
                        help='Force overwrite of an existing config file. Defaults to False.')
    # Add mutually exclusive arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", '--DSM', help="Path to DSM file (optional).",
                       default=tilename, required=False)
    group.add_argument("-m", "--elevation_service", choices=['y', 'n'], default='n',
                       help="Use elevation services APIs \"Y\" or \"N\" (optional).",
                       required=False)

    args = parser.parse_args()

    # Access the arguments
    if args.DSM:
        pass
    elif args.elevation_service == 'y':
        pass
    elif args.DSM:
        log.exception("")
    else:
        pass

    # load the project
    proj = project.ProjectMgr(input_arg=args.input_arg)
    proj.process_input(args.input_arg, image_fmt='dng')

    proj.load_images_info()

    # load the image object
    image = Image()

    elevation_service = args.elevation_service
    dsm = cache_file

    if elevation_service == 'y':
        config.update_elevation(True)
    if dsm:
        config.update_dtm(dsm)
    # if args.nodejs == 'y':
    #     config.update_nodejs(True)

    log.info("> Initializing the SRTM interpolator with a single tile")
    srtm_call = SRTM()  # Adapt lat, lon, and path as needed
    if srtm_call.parse():
        srtm_call.make_lla_interpolator()
    else:
        log.error("Failed to parse the SRTM tile.")
    config.update_srtm(srtm_call)

    # grouped data
    obs_df_grouped = proj.image_files_grouped
    group_dataframes = []  # List to store each DataFrame
    for group_name, group_data in obs_df_grouped:
        first_row = group_data.iloc[0]  # Get the first row of each group
        image_loc_name = first_row['file_loc_parent_name']

        # get the image file list
        image_file_list = general.get_image_files(images_dir=group_name,
                                                  image_fmt='dng')

        # # analyse coast view
        # analyse_coastView(image_file_list, image)

        area_df = analyse_topDown(image_file_list[:], image, image_loc_name)
        group_dataframes.append(area_df)

    # Concatenate all DataFrames in the list
    area_data = pd.concat(group_dataframes, ignore_index=True)

    # get area outliers
    all_data_df = image.get_outliers_ML(area_data, '_global')
    log.info(f'  Save globally processed data')
    all_data_file = proj.analysis_dir / f'all_data_processed.tab'
    io.save_data(all_data_df, all_data_file)


def analyse_topDown(image_file_list, image, folder_name):

    dataframes = []  # List to store each DataFrame
    features = {}  # List to store each DataFrame

    # Example loop to process multiple images
    for image_id, file in enumerate(image_file_list[:]):
        file_path = file[0]
        image_base = file[1]
        image_path_base = file[5]

        log.info(f"====> Run analysis on {image_base} <====")

        # initiate the image object
        image.init_image(file_path, image_base, image_path_base)

        # check if detection data exist, skip if not else load
        if not image.load_detection_data():
            log.warning(f"> No segmentation data found for: {image_base}. "
                        f"Skipping object.")
            continue

        data_df, feature_df = image.analyse_image(image_id, use_rgb=False)
        # data_df, feature_df = image.analyse_image(image_id, use_rgb=True)

        # Append the DataFrame to the list
        dataframes.append(data_df)
        features[image_id] = feature_df

    # Concatenate all DataFrames in the list
    all_data = pd.concat(dataframes, ignore_index=True)

    # get area outliers
    log.info(f'> Find outlier in the combined images dataset')
    all_data_df = image.get_outliers_ML(all_data, '_area')

    image.create_geojson_area(all_data_df, features)

    log.info(f'  Save processed data')
    all_data_file = image.cache_dir/f'area_data_processed_{folder_name}.tab'
    io.save_data(all_data_df, all_data_file)

    return all_data_df


def analyse_coastView(image_file_list, image):
    image_file_list = image_file_list#[:3]
    csv_file = '../../data/ANF_CoastView/ImageAnalysis/ANF_CoastView/total_flux.csv'
    poi_file = '../../data/ANF_CoastView/ImageAnalysis/ANF_CoastView/poi_locations.csv'
    # calc_and_save_data(image_file_list, image, csv_file)

    df_csv = pd.read_csv(csv_file, index_col=0)
    df = df_csv.sort_values(by='Latitude', ascending=False).reset_index()
    # Calculate distance and add it as a new column
    df['distance'] = 0.  # Initialize the column with zeros
    for i in range(1, len(df)):
        df.loc[i, 'distance'] = haversine(df.loc[i - 1, 'Latitude'],
                                          df.loc[i - 1, 'Longitude'],
                                          df.loc[i, 'Latitude'],
                                          df.loc[i, 'Longitude'])

    step_size = 5
    # Define the range of latitudes
    start_lat = df['Latitude'].min()
    end_lat = df['Latitude'].max()
    lat_step_degrees = (step_size / 111139)  # Convert step size in meters to degrees latitude
    print(start_lat, end_lat, step_size)
    # Create the new DataFrame with evenly spaced latitudes
    even_lats = np.arange(start_lat, end_lat, lat_step_degrees)
    new_df = pd.DataFrame({'Latitude': even_lats,
                           'Longitude': np.full_like(even_lats, fill_value=np.nan),
                           'R_sum_cal': np.full_like(even_lats, fill_value=np.nan),
                           'G_sum_cal': np.full_like(even_lats, fill_value=np.nan),
                           'B_sum_cal': np.full_like(even_lats, fill_value=np.nan),
                           })
    print(new_df, even_lats)
    import matplotlib.pyplot as plt
    landmarks_df = pd.read_csv(poi_file, comment='#')

    # Function to find the index of the closest point in 'df' for each landmark's latitude
    def get_closest_data_point_index(landmark_lat, data_df):
        # Calculate the absolute difference between the landmark's latitude and all data points' latitudes
        data_df['lat_diff'] = data_df['Latitude'].sub(landmark_lat).abs()
        # Find the index of the closest data point
        return data_df['lat_diff'].idxmin()

    # Populate the new DataFrame with the closest data points
    for i, lat in enumerate(new_df['Latitude']):
        closest_data_point_idx = get_closest_data_point_index(lat, df)
        closest_data_point = df.iloc[closest_data_point_idx]
        new_df.at[i, 'Longitude'] = closest_data_point['Longitude']
        new_df.at[i, 'R_sum_cal'] = closest_data_point['R_sum_cal']
        new_df.at[i, 'G_sum_cal'] = closest_data_point['G_sum_cal']
        new_df.at[i, 'B_sum_cal'] = closest_data_point['B_sum_cal']

    df = new_df.sort_values(by='Latitude', ascending=False).reset_index()

    # Mapping each landmark to the closest data point in 'df'
    landmark_to_index = {row['name']: get_closest_data_point_index(row['latitude'], df) for index, row in
                         landmarks_df.iterrows()}

    # Reverse the mapping to get a dictionary from data point index to landmark name
    index_to_landmark = {v: f'{k}' for k, v in landmark_to_index.items()}

    # Apply the mapping to create a list of x-axis labels with landmark names where applicable
    x_labels = [index_to_landmark.get(i, '') for i in df.index]
    print(x_labels)
    # Now use these labels in your plot
    plt.figure(figsize=(10, 6))  # Adjust the size as needed
    plt.plot(df.index, df['R_sum_cal'] / df['R_sum_cal'].mean(), linestyle='-', color='r')
    plt.plot(df.index, df['G_sum_cal'] / df['G_sum_cal'].mean(), linestyle='-', color='g')
    plt.plot(df.index, df['B_sum_cal'] / df['B_sum_cal'].mean(), linestyle='-', color='b')

    plt.xticks(ticks=df.index,
               labels=x_labels, rotation=45, ha='right', fontsize=12)
    # plt.ylabel('Your Y-axis Label', fontsize=14)  # Adjust the fontsize as necessary
    plt.ylabel('Normalized total flux', fontsize=14)
    plt.tick_params(axis='y', labelsize=12, direction='in')  # Adjust the fontsize of the tick labels
    plt.tick_params(axis='x', bottom=False, labelsize=12, direction='in')  # Adjust the fontsize of the tick labels
    plt.tight_layout()  # This will help prevent the labels from overlapping

    # Don't forget to show or save the plot

    plt.figure(figsize=(10, 6))
    # plt.plot(df.index, df['L_sum']/df['L_sum'].mean(), marker='o', linestyle='-', color='k')
    plt.plot(df.index, df['R_sum_cal']/df['R_sum_cal'].mean(), linestyle='-', color='r')
    plt.plot(df.index, df['G_sum_cal']/df['G_sum_cal'].mean(), linestyle='-', color='g')
    plt.plot(df.index, df['B_sum_cal']/df['B_sum_cal'].mean(), linestyle='-', color='b')

    # Abbreviating coordinates and only labeling every nth point
    n = 500  # Adjust n based on the density of your data points
    abbrev_labels = [[i, f"{lat:.2f}"] for i, (lat, lon) in
                     enumerate(zip(df['Latitude'], df['Longitude'])) if i % n == 0]
    print(abbrev_labels)
    abbrev_labels = np.array(abbrev_labels)
    print(abbrev_labels)
    plt.xticks(ticks=abbrev_labels[:, 0].astype(int), labels=abbrev_labels[:, 1],
               rotation=45, ha='right')

    plt.xlabel('Latitude (deg)', fontsize=14)
    plt.ylabel('Normalized total flux', fontsize=14)
    plt.grid(False)
    plt.tick_params(axis='y', labelsize=12, direction='in')  # Adjust the fontsize of the tick labels
    plt.tick_params(axis='x', labelsize=12, direction='in')  # Adjust the fontsize of the tick labels
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 6))
    # plt.plot(df.index, df['L_sum']/df['L_sum'].mean(), marker='o', linestyle='-', color='k')
    plt.plot(df.index, df['R_sum_cal'] / df['R_sum_cal'].mean(), linestyle='-', color='r')
    plt.plot(df.index, df['G_sum_cal'] / df['G_sum_cal'].mean(), linestyle='-', color='g')
    plt.plot(df.index, df['B_sum_cal'] / df['B_sum_cal'].mean(), linestyle='-', color='b')

    # Abbreviating coordinates and only labeling every nth point
    n = 25  # Adjust n based on the density of your data points
    abbrev_labels = [f"({lat:.2f}, {lon:.2f})" if i % n == 0 else '' for i, (lat, lon) in
                     enumerate(zip(df['Latitude'], df['Longitude']))]
    plt.xticks(ticks=df.index, labels=abbrev_labels, rotation=45, ha='right')

    plt.xlabel('GPS Position (Latitude, Longitude)')
    plt.ylabel('rel. total image flux (RGB)', fontsize=12)
    plt.grid(False)
    plt.show()

    # # # Creating the plot for values against cumulative distance
    # # df['cumulative_distance'] = df['distance'].cumsum()
    # # plt.figure(figsize=(12, 6))
    # # plt.plot(df['cumulative_distance'], df['L_sum'], marker='o', linestyle='-', color='k')
    # # plt.plot(df['cumulative_distance'], df['R_sum_cal'], marker='o', linestyle='-', color='r')
    # # plt.plot(df['cumulative_distance'], df['G_sum_cal'], marker='o', linestyle='-', color='g')
    # # plt.plot(df['cumulative_distance'], df['B_sum_cal'], marker='o', linestyle='-', color='b')
    # # plt.xlabel('Cumulative Distance (meters)')
    # # plt.ylabel('Value')
    # # plt.title('Value by Cumulative Distance')
    # # plt.grid(True)
    # #
    # # # Show both plots
    # # plt.tight_layout()  # Adjust layout to make room for label rotation


def calc_and_save_data(image_file_list, image, csv_file):

    result_arr = np.zeros((len(image_file_list), 12), dtype=np.float32)
    result_arr[result_arr == 0] = np.nan

    for i, file in enumerate(image_file_list):
        file_path = file[0]
        image_base = file[1]
        image_path_base = file[5]

        log.info(f"====> Run analysis on {image_base} <====")

        # initiate the image object
        image.init_image(file_path, image_base, image_path_base)

        # print(image.name)
        pos, ypr, _ = image.get_aircraft_pose()
        # print(pos, ypr)

        calRGB_image = image.load_image_calibrated()
        calRGB_image[calRGB_image < 0] = 0

        XYZ_image = image.load_XYZ_image()
        lum_image = XYZ_image[:, :, 1]
        lum_image[lum_image < 0] = 0
        linRGB_image = image.convert_XYZ_to_sRGB(XYZ_image)
        linRGB_image = convert_mnk_to_kmn(linRGB_image)
        linRGB_image[linRGB_image < 0] = 0
        # print(lum_image.shape, np.nansum(lum_image))
        # print(linRGB_image.shape, np.nansum(linRGB_image, axis=(1, 2)))
        # print(calRGB_image.shape, np.nansum(calRGB_image, axis=(1, 2)))

        tmp = pos
        tmp.extend(ypr[:2])
        tmp.extend(np.nansum(calRGB_image, axis=(1, 2)))
        tmp.extend(np.nansum(linRGB_image, axis=(1, 2)))
        tmp.extend([np.nansum(lum_image)])

        result_arr[i, :] = tmp

    df = pd.DataFrame(data=result_arr,
                      columns=['Latitude', 'Longitude',
                               'Altitude', 'yaw', 'pitch',
                               'R_sum_cal', 'G_sum_cal', 'B_sum_cal',
                               'R_sum', 'G_sum', 'B_sum',
                               'L_sum'])
    # Calculate distance and add it as a new column
    df['distance'] = 0.  # Initialize the column with zeros
    for i in range(1, len(df)):
        df.loc[i, 'distance'] = haversine(df.loc[i - 1, 'Latitude'],
                                          df.loc[i - 1, 'Longitude'],
                                          df.loc[i, 'Latitude'],
                                          df.loc[i, 'Longitude'])

    df.to_csv(csv_file)


    # # quit()


# Function to calculate the haversine distance between two points in meters
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Difference in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c * 1000  # Convert to meters
    return distance


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
