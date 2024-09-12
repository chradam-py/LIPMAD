#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse

from lipmad import base_conf
from lipmad import general
from lipmad import pose
from lipmad import project

from props import getNode

log = base_conf.log


def main():
    """ Initialize a new camera """
    description = """Get aircraft/camera pose from image meta info and create a pix4d pose file."""

    # Create argument parser
    parser = argparse.ArgumentParser(prog='lipmad-set_poses',
                                     description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional mandatory argument
    parser.add_argument("input_arg", type=str,
                        help="Directory containing sets of aerial image(s).")
    parser.add_argument('--max-angle', type=float, default=25.0,
                        help='Maximum pitch or roll angle for image inclusion.')
    # parser.add_argument('-f', '--force-overwrite', action='store_true', dest='force',
    #                     help='Force overwrite of an existing config file. Defaults to False.')

    args = parser.parse_args()

    # load the project
    proj = project.ProjectMgr(input_arg=args.input_arg)
    proj.process_input(args.input_arg, image_fmt='dng')

    # grouped data
    obs_df_grouped = proj.image_files_grouped

    for group_name, group_data in obs_df_grouped:
        first_row = group_data.iloc[0]  # Get the first row of each group
        image_loc_name = first_row['file_loc_parent_name']

        image_file_list = general.get_image_files(images_dir=group_name, image_fmt='dng')

        pose.set_poses(proj=proj, image_list=image_file_list,
                       images_dir_base=image_loc_name,
                       max_angle=args.max_angle)

        # compute the project's NED (North-East-Distance) reference location (based on average of
        # aircraft poses)
        log.info("> Set NED reference location")
        proj.compute_ned_reference_lla(image_loc_name)

        pose.compute_camera_poses(image_file_list, image_loc_name)

        # save the poses
        proj.save_images_info(image_loc_name)

        # save change to ned reference
        proj.save()


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
