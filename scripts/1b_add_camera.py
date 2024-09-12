#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import numpy as np
import os
import pathlib

from props import getNode, PropertyNode
import props_json

from lipmad import base_conf
from lipmad import camera
from lipmad import io
from lipmad import project

log = base_conf.log


def main():
    """ Initialize a new camera """
    description = """ Add a camera to the current project. """

    # Create argument parser
    parser = argparse.ArgumentParser(prog='lipmad-add_camera',
                                     description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional mandatory argument
    parser.add_argument("input_arg", type=str,
                        help="Directory containing sets of aerial image(s).")
    parser.add_argument('--camera-config', default='../cameras', dest='config',
                        help='Camera configuration file directory. Defaults to `../cameras`')
    parser.add_argument('--yaw-deg', type=float, default=0.0,
                        help='camera yaw mounting offset from aircraft')
    parser.add_argument('--pitch-deg', type=float, default=-90.0,
                        help='camera pitch mounting offset from aircraft')
    parser.add_argument('--roll-deg', type=float, default=0.0,
                        help='camera roll mounting offset from aircraft')
    parser.add_argument('--ccd-width', type=float,
                        help='Width of CCD chip in mm.', default=17.3)
    parser.add_argument('--ccd-height', type=float,
                        help='Height of CCD chip in mm.', default=13.)
    parser.add_argument('-f', '--force-overwrite', action='store_true', dest='force',
                        help='Force overwrite of an existing config file. Defaults to False.')

    args = parser.parse_args()

    proj = project.ProjectMgr(input_arg=args.input_arg)

    proj.process_input(args.input_arg, image_fmt='dng')

    log.info("> Detect camera")
    camera_name, make, model, lens_model, focal_len_mm, image_file = proj.detect_camera()
    camera_file = os.path.join('..', 'cameras', camera_name + '.json')
    lens_model = 'N/A' if lens_model is None else lens_model
    log.info(f"  Camera auto-detected: "
             f"{', '.join([camera_name, make, model, lens_model, str(focal_len_mm)])}")
    log.info(f"  Camera file: {camera_file}")

    # get image width and height from image meta data
    raw_file = io.load_raw_file(image_file)
    height, width = raw_file.raw_image.shape
    height_px, width_px = raw_file.raw_image_visible.shape
    print(width, width_px, width_px/width)
    print(height, height_px)
    scale_width = width_px / width
    scale_height = height_px / height

    # calculate pixel size in µm
    pixel_size = np.sqrt((args.ccd_width * args.ccd_height) / (height * width)) * 1e3

    # Bit depth - find the maximum value and the corresponding bit depth
    maximum_value = raw_file.raw_image_visible.max()
    try:
        bit_depth = base_conf.BIT_DEPTH_CONVERSION[maximum_value]
    except KeyError:
        log.info(f"The provided image ({image_file}) does not have any saturated pixels "
                 f"(maximum value: {maximum_value}).")
        log.info("Please enter the bit depth manually.\n"
                 "Typical values are 8 (JPEG), 10 (Samsung), 12 (Apple), 14 (Nikon).\n")
        bit_depth = int(input())

    # get some useful properties
    bias = raw_file.black_level_per_channel
    bayer_pattern = raw_file.raw_pattern.tolist()
    n_colors = raw_file.num_colors
    colors = raw_file.color_desc.decode()
    fx, fy, cu, cv = camera.calc_K(args.ccd_width, args.ccd_height,
                                   focal_len_mm, width_px/2, height_px/2)
    # update configuration
    camera.set_defaults()
    camera.set_meta(make, model, lens_model)
    camera.set_lens_params(args.ccd_width, args.ccd_height, focal_len_mm, scale_width, scale_height)
    camera.set_image_params(pixel_size, width_px, height_px, bias, n_colors, colors, bayer_pattern, bit_depth)
    camera.set_K(fx, fy, cu, cv)
    # save the camera configuration
    cam_node = getNode('/config/camera', True)
    tmp_node = PropertyNode()

    log.info(f"> Saving: {camera_file}")
    camera_file = pathlib.Path(camera_file).expanduser().resolve()
    if camera_file.exists():
        log.info(f"  Camera config file already exists: {camera_file}")
        if args.force:
            log.info("> Overwriting configuration file ...")
        else:
            log.info("  Use [ --force-overwrite ] to overwrite configuration ...")
            quit()
    props_json.save(str(camera_file), cam_node)

    # update and save project config
    if props_json.load(camera_file, tmp_node):
        # copy/overlay camera config on the project
        props_json.overlay(cam_node, tmp_node)

        camera.set_mount_params(args.yaw_deg, args.pitch_deg, args.roll_deg)

        # save project
        proj.save()


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
