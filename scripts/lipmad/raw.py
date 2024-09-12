#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# Matrix for converting RGBG2 to RGB
M_RGBG2_to_RGB = np.array([[1, 0, 0, 0],
                           [0, 0.5, 0, 0.5],
                           [0, 0, 1, 0]])


def _generate_bayer_slices(color_pattern, colours=range(4)):
    """
    Generate the slices used to demosaick data.
    """
    # Find the positions of the first element corresponding to each colour
    positions = [np.array(np.where(color_pattern == colour)).T[0] for colour in colours]

    # Make a slice for each colour
    slices = [np.s_[..., x::2, y::2] for x, y in positions]

    return slices


def demosaick(bayer_map, data, color_desc="RGBG"):
    """
    Uses a Bayer map `bayer_map` (RGBG channel for each pixel) and any number
    of input arrays `data`.
    """
    # Cast the data to a numpy array for the following indexing tricks to work
    data = np.array(data)

    # Check that we are dealing with RGBG2 data, as only these are supported right now.
    assert color_desc in ("RGBG", b"RGBG"), f"Unknown colour description `{color_desc}"

    # Check that the data and Bayer pattern have similar shapes
    assert data.shape[-2:] == bayer_map.shape, (f"The data ({data.shape}) and "
                                                f"Bayer map ({bayer_map.shape}) have incompatible shapes")

    # Demosaick the data along their last two axes
    bayer_pattern = bayer_map[:2, :2]
    slices = _generate_bayer_slices(bayer_pattern)

    # Combine the data back into one array of shape [..., 4, x/2, y/2]
    newshape = list(data.shape[:-2]) + [4, data.shape[-2]//2, data.shape[-1]//2]
    RGBG = np.empty(newshape)
    for i, s in enumerate(slices):
        RGBG[..., i, :, :] = data[s]

    return RGBG


def convert_between_colourspaces(data, conversion_matrix, axis=0):
    """
    Convert data from one color space to another, using the given conversion matrix.
    Matrix multiplication can be done on an arbitrary axis in the data, using np.tensordot.
    Example uses include converting from RGBG2 to RGB and from RGB to XYZ.
    """
    # Use tensor multiplication to multiply along an arbitrary axis
    new_data = np.tensordot(conversion_matrix, data, axes=(1, axis))
    new_data = np.moveaxis(new_data, 0, axis)

    return new_data


def convert_RGBG2_to_RGB(RGBG2_data, axis=0):
    """
    Convert data in Bayer RGBG2 format to RGB format, by averaging the G and G2
    channels.

    The RGBG2 data are assumed to have their color axis on `axis`. For example,
    if `axis=0`, then the data are assumed to have a shape of (4, ...).
    If `axis=2`, then the data are assumed to be (:, :, 4, ...). Etc.

    The resulting array has the same shape, but with the given axis changed from
    4 to 3.
    """
    RGB_data = convert_between_colourspaces(RGBG2_data, M_RGBG2_to_RGB, axis=axis)

    return RGB_data
