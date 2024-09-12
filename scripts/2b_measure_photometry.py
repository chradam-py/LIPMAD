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
    """ Measure photometry in images """
    description = """ Measure photometry in image. """

    # Create argument parser
    parser = argparse.ArgumentParser(prog='lipmad-measure_photometry',
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
    image = Image()

    # loop images and get photometry for each channel
    for file in proj.image_file_list:
        file_path = file[0]
        image_base = file[1]
        image_path_base = file[5]

        log.info(f"====> Run photometry on {image_base} <====")

        # initiate the image object
        image.init_image(file_path, image_base, image_path_base)

        # check if detection data exist, skip if not else load
        if not image.load_detection_data():
            log.warning(f"> No segmentation data found for: {image_base}. "
                        f"Skipping object.")
            continue

        # load the image for photometry
        XYZ_image = image.load_XYZ_image()

        # log.info("> Convert XYZ image -> xy image")
        # xy_image = image.convert_XYZ_to_xy(XYZ_image)

        calRGB_image = image.load_image_calibrated()
        # import tifffile
        #
        # # Create a sample NumPy array
        # # array = convert_kmn_to_mnk(calRGB_image)
        # array = calRGB_image
        # array = array.clip(min=0)
        # # Save the array as a TIFF file
        # tifffile.imwrite(image.georef_image_file, np.asarray(array, dtype=np.float32))
        # # Load the TIFF file back into a NumPy array
        # loaded_array = tifffile.imread(image.georef_image_file)
        # print(loaded_array)

        linRGB_image = image.convert_XYZ_to_sRGB(XYZ_image)

        linRGB_image = convert_mnk_to_kmn(linRGB_image)
        # print(linRGB_image.shape)
        # print(calRGB_image.shape)
        # print(xy_image.shape)

        # get the segmentation map and catalogue
        # segment_map = image.segment_map
        # segment_cat = image.segment_src_table
        # print(segment_cat)
        # print(segment_map.labels)

        # print(xy_image.shape)
        # print(segm.bbox)

        # todo: put the georeference later
        # image.georeference_data(linRGB_image)
        # image.georeference_data(segment_map.data)

        # loop filter and get flux from each channel, save data for each segment:
        # get distribution of rgb values, median, std, as well as CCT values
        kwargs = {'mask': image.rgb_mask, 'box_size': 31, 'filter_size': 11,
                  'exclude_percentile': 5.}
        try:
            image.get_photometry(calRGB_image, False, args.force, **kwargs)
        except ZeroDivisionError as e:
            continue
        try:
            image.get_photometry(linRGB_image, True, args.force, **kwargs)
        except ZeroDivisionError as e:
            continue

        # save results: table

        # make some plots

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(2, 3, sharex="all", sharey="all", figsize=(10, 6))
        #
        # ax[0, 0].imshow(xy_image[:, :, 0], interpolation='nearest')
        # from scipy import ndimage
        # result = ndimage.median_filter(xy_image, size=9, axes=(0, 1))
        # ax[0, 1].imshow(xy_image[:, :, 1], interpolation='nearest')
        # ax[0, 2].imshow(result[:, :, 1], interpolation='nearest')
        #
        # cmap = segment_map.make_cmap(seed=12345)
        # ax[1, 0].imshow(segment_map.data, cmap=cmap, interpolation='nearest')
        # ax[1, 1].imshow(XYZ_image[:, :, 1], interpolation='nearest')
        # print(np.max(result, axis=1), np.min(result, axis=1))
        # CCT_IMAGE_masked = image.convert_xy_to_CCT(result, method="Hernandez 1999")
        #
        # ax[1, 2].imshow(CCT_IMAGE_masked, interpolation='nearest')
        #
        # plt.show()
        # apply all corrections possible, bias, flat, dark, iso normalization
        # demosaick

        # make a XYZ -> xy image from only bia and flat for CCT

        # make rgbg image in adu m-2 sr-1 s-1 for photometry

        # measure photometry for each segment

        # save results to file

        #
        log.info('====> Photometry measurement finished <====\n')


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
