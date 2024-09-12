#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np
import os
from pathlib import Path
import re

from . import base_conf


def get_last_valid_folder(path):

    if path.suffix != "":
        return path.parent

    # Check if the path itself is a valid directory
    if path.is_dir():
        return path

    # Iterate over the path's parents (including itself)
    for parent in path.parents:
        # The first parent that exists (closest to the original path) is the valid directory
        if parent.exists():
            # Return the name of the folder
            return parent

    # If no valid directory is found, return the empty string
    return ""


def generate_XY(shape):
    """
    Given a `shape`, generate a meshgrid of index values in both directions and a combination.
    """

    x = np.arange(shape[1])
    y = np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()])
    return X, Y, XY


def correct_bias_from_map(bias_map, data):
    """
    Apply a bias correction from a bias map `bias_map` to any number of
    elements in `data`
    """
    data_corrected = data - bias_map

    return data_corrected


def correct_dark_current_from_map(dark_current_map, exposure_time, data):
    """
    Apply a dark current correction from a dark current map `dark_current_map`,
    multiplied by an `exposure_time`, to any number of elements in `data`.
    """

    dark_current = exposure_time * dark_current_map
    data_corrected = data - dark_current

    return data_corrected


def convert_flux_from_map(calib_coeff_map, data):
    """
    Apply a bias correction from a bias map `bias_map` to any number of
    elements in `data`
    """
    data_corrected = data * calib_coeff_map[0] + calib_coeff_map[1]

    return data_corrected


def normalise_iso(lookup_table, isos, data):
    """
    Normalise data for ISO speed. `isos` can be an iterable of the same length as
    `data` or a single value, which is then used for all data elements.
    """
    # Get the normalisation from the lookup table
    normalisation = lookup_table[1, isos]

    # If `normalisation` is a single value, divide
    try:
        data_normalised = data / normalisation
    # If this does not work, assume `normalisation` is iterable
    except ValueError:
        # NumPy broadcasting works when the first axis of `data`, which has
        # the same length as `normalisation`, is at the end
        data_normalised = np.moveaxis(data.copy(), 0, -1)
        data_normalised /= normalisation
        data_normalised = np.moveaxis(data_normalised, -1, 0)

    return data_normalised


def convert_to_photoelectrons_from_map(gain_map, data):
    """
    Convert `data` from normalised ADU to photoelectrons using a map of gain
    in each pixel `gain_map`.
    """
    data_converted = data / gain_map

    return data_converted


def correct_flatfield_from_map(flatfield, data):
    """
    Apply a flat-field correction from a flat-field map `flatfield` to an
    array or iterable of arrays `data`.
    """
    data_corrected = data * flatfield

    return data_corrected


def vignette_radial(shape, XY, k0, k1, k2, k3, k4, cx_hat, cy_hat):
    """
    Vignetting function as defined in Adobe DNG standard 1.4.0.0
    Reference:
        https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/dng_spec_1.4.0.0.pdf

    Adapted to use a given image shape for conversion to relative coordinates,
    rather than deriving this from the inputs XY.

    Parameters
    ----------
    XY
        array with X and Y positions of pixels, in absolute (pixel) units
    k0, ..., k4
        polynomial coefficients
    cx_hat, cy_hat
        optical center of image, in normalized euclidean units (0-1)
        relative to the top left corner of the image
    """
    x, y = XY

    x0, y0 = 0, 0  # top left corner
    x1, y1 = shape[1], shape[0]  # bottom right corner
    cx = x0 + cx_hat * (x1 - x0)
    cy = y0 + cy_hat * (y1 - y0)
    # (cx, cy) is the optical center in absolute (pixel) units

    mx = max([abs(x0 - cx), abs(x1 - cx)])
    my = max([abs(y0 - cy), abs(y1 - cy)])
    m = np.sqrt(mx**2 + my**2)
    # m is the Euclidean distance from the optical center to the farthest corner in absolute (pixel) units

    r = 1 / m * np.sqrt((x - cx)**2 + (y - cy)**2)
    # r is the normalized Euclidean distance of every pixel from the optical center (0-1)

    p = [k4, 0, k3, 0, k2, 0, k1, 0, k0, 0, 1]
    g = np.polyval(p, r)
    # g is the normalization factor to multiply measured values with

    return g


def apply_vignette_radial(shape, parameters):
    """
    Apply a radial vignetting function to obtain a correction factor map.
    """
    X, Y, XY = generate_XY(shape)
    correction = vignette_radial(shape, XY, *parameters).reshape(shape)
    return correction


def create_folder(folder, log: logging = base_conf.log,
                  create_if_needed: bool = True) -> bool:
    """Create main and children folder for the process."""

    if not folder.exists():
        if create_if_needed:
            log.info(f"> Create folder: {folder}")
            folder.mkdir(exist_ok=True, parents=True)
        else:
            log.error(f"> Folder doesn't exist: {folder}")
            return False
    return True


def compile_regex(prefix, extensions):
    """Compile regex for file extensions."""
    pattern = f'^{prefix}.*({"|".join(extensions)})$'
    return re.compile(pattern)


def handle_directory(directory, regex):
    """Process files in a directory."""
    image_file_list = []
    for root, _, files in os.walk(directory):
        matching_files = [os.path.join(root, file) for file in files if regex.search(file)]
        image_file_list.extend(matching_files)
    return image_file_list


def check_image_format(image_fmt):

    image_fmt = image_fmt.upper() if image_fmt is not None else None

    if image_fmt is not None and not isinstance(image_fmt, str):
        raise ValueError("Image format must be a str")
    if image_fmt is not None and image_fmt not in base_conf.SUPPORTED_IMAGE_FORMATS:
        raise KeyError(f"Given image typ '{image_fmt}' NOT supported.")

    if image_fmt is None:
        image_fmt = [fmt for fmt in base_conf.SUPPORTED_IMAGE_FORMATS if fmt != 'FITS']
    else:
        image_fmt = [image_fmt]

    extensions = [base_conf.DEF_FILE_EXT[fmt] for fmt in image_fmt]

    return image_fmt, extensions


def split_file_names(image_file_list):

    paths = [
        [
            f,  # File path
            Path(f).stem,  # Stem of the file
            Path(f).suffix,  # File extension
            os.path.dirname(f),  # Directory containing the file
            str(Path(f).parents[1 if 'raw' in os.path.dirname(f) else 0]),
            # Full path one level up, or two if 'raw' is in the name
            Path(f).parents[1 if 'raw' in os.path.dirname(f) else 0].name
            # Name of the directory one level up, or two if 'raw' is in the name
        ] for f in image_file_list
    ]

    return np.array(paths)


def get_image_files(images_dir, image_fmt='dng'):
    """"""
    image_fmt, extensions = check_image_format(image_fmt=image_fmt)
    regex_ext = compile_regex(prefix='', extensions=extensions)
    image_list = handle_directory(images_dir, regex_ext)
    image_file_list = split_file_names(image_list)
    return image_file_list
