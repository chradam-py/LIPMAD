#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import os
import datetime
from lipmad import base_conf
from lipmad.image import Image as Image_class
from lipmad import io
from lipmad import project
from lipmad import general
from lipmad.Utils.raster_utils import *
from lipmad.Utils.new_elevation import SRTM
from lipmad.create_geojson import create_geojson_feature

log = base_conf.log

tilename = "S24W071"  # Tile name
cache_file = f'../../data/ANF_DEM_rasters/S24W071.SRTMGL1/{tilename}.hgt'


def is_valid_file(arg):
    if not os.path.isfile(arg):
        log.error("\"" + arg + "\""" is not a valid file.  Switching to Default elevation model")
        return
    else:
        return arg


def main():
    """ Georeference data """
    description = """ Georeference data """

    # Create argument parser
    parser = argparse.ArgumentParser(prog='lipmad-georef',
                                     description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional mandatory argument
    parser.add_argument("input_arg", type=str,
                        help="Directory containing sets of aerial image(s).")
    parser.add_argument('-f', '--force-overwrite', action='store_true', dest='force',
                        help='Force overwrite of an existing config file. Defaults to False.')
    parser.add_argument("-n", "--nodejs", choices=['y', 'n'], default='n',
                        help="Cmd from nodejs? \"Y\" or \"N\" (optional).",
                        required=False)

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
    image = Image_class()

    # grouped data
    obs_df_grouped = proj.image_files_grouped

    elevation_service = args.elevation_service
    dsm = cache_file
    nodejs = args.nodejs
    if elevation_service == 'y':
        config.update_elevation(True)
    if dsm:
        config.update_dtm(dsm)
    if args.nodejs == 'y':
        config.update_nodejs(True)

    log.info("> Initializing the SRTM interpolator with a single tile")
    srtm_call = SRTM()  # Adapt lat, lon, and path as needed
    if srtm_call.parse():
        srtm_call.make_lla_interpolator()
    else:
        log.error("Failed to parse the SRTM tile.")
    config.update_srtm(srtm_call)
    for group_name, group_data in obs_df_grouped:
        first_row = group_data.iloc[0]  # Get the first row of each group
        image_loc_name = first_row['file_loc_parent_name']

        image_file_list = general.get_image_files(images_dir=group_name,
                                                  image_fmt='dng')

        # ref = proj.get_ned_reference_lla(image_loc_name)
        group_feature_collection = {"type": "FeatureCollection", "features": []}
        line_coordinates = []
        line_feature = ""
        for file in image_file_list[:]:
            file_path = file[0]
            image_base = file[1]
            image_path_base = file[5]

            log.info(f"====> Run geo referencing on {image_base} <====")

            # initiate the image object
            image.init_image(file_path, image_base, image_path_base)

            log.info(f'> Georeference image bounding box')
            result = image.get_georef()

            (coord_array, polybox, fixed_polygon,
             pos, _, properties) = result
            log.info(f'> Create image geojson file')
            # Create GeoJSON features for the current image
            feature_point, feature_polygon = create_geojson_feature(polybox,
                                                                    pos[0],
                                                                    pos[1],
                                                                    properties)

            image.feature_collection["features"].append(feature_point)
            image.feature_collection["features"].append(feature_polygon)

            geojson_file = f"{image.name}_geo.json"
            io.write_geojson_file(geojson_file, image.meta_dir,
                                  image.feature_collection)

            group_feature_collection["features"].append(feature_point)
            group_feature_collection["features"].append(feature_polygon)

            # pos = image.get_image_gps()
            # Update line coordinates for potential LineString creation
            line_coordinates.append([pos[0], pos[1]])

            #
            image.georeference_image_data(coord_array, fixed_polygon)
            #
            # log.info(f'> Georeference B/R image')
            # #
            # image.georeference_image_ratio_br(coord_array, fixed_polygon)

        # Add lines to the GeoJSON feature collection if necessary
        if line_coordinates:

            now = datetime.datetime.now()
            process_date = f"{now.strftime('%Y-%m-%d %H-%M')}"
            line_geometry = dict(type="LineString", coordinates=line_coordinates)
            mission_props = dict(Process_date=process_date, epsg=config.epsg_code,
                                 cog=config.cog)
            line_feature = dict(type="Feature", geometry=line_geometry, properties=mission_props)
            group_feature_collection["features"].insert(0, line_feature)

        log.info(f'> Create area geojson file')
        geojson_file = f"M_{image_loc_name}.json"
        io.write_geojson_file(geojson_file, image.processed_dir,
                              group_feature_collection)

        # create_mosaic(str(image.processed_dir), image.processed_dir)


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
