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
from lipmad.conversions import convert_kmn_to_mnk, convert_mnk_to_kmn
from lipmad.exif import Exif
from lipmad.image import Image
from lipmad import io
from lipmad import project

log = base_conf.log


def main():
    """ Detect sources in images """
    description = """ Detect sources in image. """

    # Create argument parser
    parser = argparse.ArgumentParser(prog='lipmad-detect_sources',
                                     description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional mandatory argument
    parser.add_argument("input_arg", type=str,
                        help="Directory containing sets of aerial image(s).")
    parser.add_argument('-f', '--force-overwrite',
                        action='store_true', dest='force',
                        help='Force overwrite of an existing config file.'
                             ' Defaults to False.')

    args = parser.parse_args()

    # load the project
    proj = project.ProjectMgr(input_arg=args.input_arg)
    proj.process_input(args.input_arg, image_fmt='dng')

    proj.load_images_info()

    # load the image object
    image = Image()

    # loop images
    for file in proj.image_file_list:
        file_path = file[0]
        image_base = file[1]
        image_path_base = file[5]

        log.info(f"====> Run source detection for {image_base} <====")

        # initiate the image object
        image.init_image(file_path, image_base, image_path_base)
        # cal_data = image.load_image_calibrated()
        # print(cal_data.shape)
        # gray_image = image.calculate_physical_luminance(cal_data)

        # # gray_image = XYZ_data[2, :, :] / XYZ_data[0, :, :]
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1, sharex="all", sharey="all", figsize=(10, 6))
        # ax.imshow(gray_image, interpolation='nearest')
        # plt.show()

        # quit()
        XYZ_data = image.load_XYZ_image()
        gray_image = XYZ_data[:, :, 1]

        gray_mask = (gray_image >= 1.) | (gray_image <= 0.)
        if image.rgb_mask is not None:
            gray_mask |= image.rgb_mask

        # get background
        bkg_kwargs = {'mask': gray_mask, 'box_size': 79, 'filter_size': 5,
                      'exclude_percentile': 5.}
        image.get_background(gray_image,
                             'gray',
                             force_est=args.force,
                             **bkg_kwargs)

        # detect sources
        detect_kwargs = {'rgb_mask': gray_mask,
                         'bkg_sigma': 5.,
                         'use_convolve': False,
                         'npixels': 31,
                         'gfwhm': 1, 'gsize': 11, 'deblend': False}
        image.find_sources(gray_image, **detect_kwargs)

        if image.segment_map is None:
            continue
        # save segment map, grayscale and bkg to pkl
        image.save_detection_data()

        log.info('====> Source detection finished <====')
        # proj.save_images_info(image_path_base)

        # update folder and state

        # save proj, mask, segment map, catalog, detection image

        # clip values to be in range [0-1]
        # XYZ_data_clipped = np.clip(XYZ_data, 0., 1.)

        # convert XYZ to sRGB
        # sRGB_data = image.convert_XYZ_to_sRGB(XYZ_data_clipped)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 3, sharex="all", sharey="all", figsize=(10, 6))
        # vmin = 0
        # vmax = 1500
        # ax[0].imshow(image_raw, vmin=vmin+5000, vmax=vmax+5000, interpolation='nearest')
        # ax[1].imshow(image_raw_bias_corrected, vmin=vmin, vmax=vmax, interpolation='nearest')
        # ax[2].imshow(image_raw_flat_corrected, vmin=vmin, vmax=vmax, interpolation='nearest')
        # plt.show()

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 3, sharex="all", sharey="all", figsize=(10, 6))
        #
        # ax[0].imshow(gray_image, interpolation='nearest', vmin=0,
        #              vmax=np.nanpercentile(gray_image, 99.5))
        # ax[1].imshow(image.bkg_image, interpolation='nearest')
        # segment_map = image.segment_map
        # cmap = segment_map.make_cmap(seed=12345)
        # ax[2].imshow(segment_map.data, cmap=cmap, interpolation='nearest')
        #
        # plt.show()

        # import colour
        # colour.plotting.plot_image(np.clip(sRGB_data, 0, 1))
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2, sharex="all", sharey="all", figsize=(10, 6))
        # vmin = 0
        # vmax = 255
        # ax[0].imshow(XYZ_data_clipped[:, :, 1]*255, vmin=vmin, vmax=vmax, interpolation='nearest')
        # ax[1].imshow(sRGB_data, vmin=0, vmax=255, interpolation='nearest')
        # plt.show()


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
