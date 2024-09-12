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
from lipmad.plot import PlotResults

log = base_conf.log
plot = PlotResults()


def main():
    """ Georeference data """
    description = """ Plot data """

    # Create argument parser
    parser = argparse.ArgumentParser(prog='lipmad-plot',
                                     description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional mandatory argument
    parser.add_argument("input_arg", type=str,
                        help="Directory containing sets of aerial image(s).")
    parser.add_argument('-f', '--force-overwrite', action='store_true', dest='force',
                        help='Force overwrite of an existing config file. Defaults to False.')

    args = parser.parse_args()

    # load the project
    proj = project.ProjectMgr(input_arg=args.input_arg)
    proj.process_input(args.input_arg, image_fmt='dng')

    proj.load_images_info()

    # load the image object
    image = Image_class()

    # grouped data
    obs_df_grouped = proj.image_files_grouped

    plot_detection(proj, image)
    #
    # plot_table_data(proj, image, suffix='_image')
    # plot_table_data(proj, image, suffix='_area')

    # plot_outlier_segment_images(proj, image)

    # plot_coastal_image_and_map(proj, image)


def plot_coastal_image_and_map(proj, image):
    file_list = proj.image_file_list[:]
    for image_idx, file in enumerate(file_list):
        file_path = file[0]
        image_base = file[1]
        image_path_base = file[5]

        # initiate the image object
        image.init_image(file_path, image_base, image_path_base)

        # plot.plot_coastal_image_and_map(image)
        plot.plot_coastal_image(image)


def plot_outlier_segment_images(proj, image):
    file_list = proj.image_file_list[:]
    for image_idx, file in enumerate(file_list):
        file_path = file[0]
        image_base = file[1]
        image_path_base = file[5]

        # initiate the image object
        image.init_image(file_path, image_base, image_path_base)

        plot.plot_segm_outlier_closeup(image)


def plot_table_data(proj, image, suffix='_image'):
    """"""

    file_list = proj.image_file_list[:] if suffix == '_image' else proj.image_file_list[:1]

    for image_idx, file in enumerate(file_list):
        file_path = file[0]
        image_base = file[1]
        image_path_base = file[5]
        plt_str = image_base if suffix == '_image' else image_path_base
        log.info(f"====> Make plots for {plt_str} <====")

        # initiate the image object
        image.init_image(file_path, image_base, image_path_base)

        plot.plot_table_data(image_idx, image, suffix=suffix)


def plot_detection(proj, image):
    """"""

    for image_idx, file in enumerate(proj.image_file_list[:]):
        file_path = file[0]
        image_base = file[1]
        image_path_base = file[5]

        log.info(f"====> Make plots for {image_base} <====")

        # initiate the image object
        image.init_image(file_path, image_base, image_path_base)

        # plot.plot_image_loc_on_map(image_idx, image)

        image.load_image_calibrated()
        # image.load_XYZ_image()
        # image.get_background(image.image_XYZ, 'gray')
        # image.load_detection_data()
        #
        # plot.plot_detection(image)
        #
        # plot.plot_as_rgb_image(image)
        # plot.plot_as_rgb_image(image, include_segm=True)

        # plot.plot_images_colored(image)
        plot.plot_g_band(image)


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
