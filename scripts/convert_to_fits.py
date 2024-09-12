#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse

from lipmad import base_conf
from lipmad.image import Image
from lipmad import project

log = base_conf.log


def main():
    """ Detect sources in images """
    description = """ Detect sources in image. """

    # Create argument parser
    parser = argparse.ArgumentParser(prog='lipmad-convert_to_fits',
                                     description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional mandatory argument
    parser.add_argument("input_arg", type=str,
                        help="Directory containing sets of RAW image(s).")
    # parser.add_argument('-f', '--force-overwrite', action='store_true', dest='force',
    #                     help='Force overwrite of an existing config file. Defaults to False.')

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

        log.info(f"====> Run conversion to FITS for {image_base}.DNG <====")

        # initiate the image object
        image.init_image(file_path, image_base, image_path_base)

        image.convert_raw2fits_and_save()


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
