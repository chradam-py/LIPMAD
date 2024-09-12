#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import lensfunpy
import numpy as np

from astropy.convolution import convolve
from astropy.stats import SigmaClip

from photutils.background import (Background2D,
                                  MMMBackground, StdBackgroundRMS, SExtractorBackground,
                                  MADStdBackgroundRMS,
                                  BkgZoomInterpolator)
from photutils.segmentation import (detect_sources, deblend_sources,
                                    make_2dgaussian_kernel, SourceCatalog)


def get_background(img, box_size=50, filter_size=11, mask=None, interp=None,
                   clipping_sigma=3.,
                   bkg_estimator=SExtractorBackground(),
                   bkgrms_estimator=MADStdBackgroundRMS(),
                   exclude_percentile=10.):
    """ Run a photutils background with SigmaClip and MedianBackground"""

    if interp is None:
        interp = BkgZoomInterpolator()

    return Background2D(img, box_size=box_size,
                        sigma_clip=SigmaClip(sigma=clipping_sigma,
                                             maxiters=10),
                        filter_size=filter_size,
                        bkg_estimator=bkg_estimator,
                        bkgrms_estimator=bkgrms_estimator,
                        exclude_percentile=exclude_percentile,
                        edge_method='pad',
                        mask=mask,
                        interpolator=interp)


def extract_sources(data, bkg, bkg_rms, bkg_sigma=3., gfwhm=3, gsize=3,
                    npixels=9, use_convolve=False, deblend=False):
    """"""

    # set the detection threshold
    threshold = bkg + (bkg_sigma * bkg_rms)

    # Before detection, smooth image with Gaussian FWHM = 3 pixels
    kernel = make_2dgaussian_kernel(fwhm=gfwhm, size=gsize)  # FWHM = 3.0
    convolved_data = convolve(data=data, kernel=kernel)
    if use_convolve:
        data = convolved_data

    # Detect
    segm_detect = detect_sources(data=data,
                                 threshold=threshold,
                                 mask=None,
                                 npixels=npixels,
                                 connectivity=8)
    # Deblend
    if deblend:
        segm_detect = deblend_sources(data=data,
                                      segment_img=segm_detect,
                                      npixels=npixels, nproc=8,
                                      nlevels=32, contrast=0.01,
                                      relabel=True,
                                      progress_bar=True)

    del data, bkg, bkg_rms

    return segm_detect


def get_catalog_from_segments(data, segment_map, bkg=None, bkg_rms=None, mask=None):
    """"""

    if bkg is not None:
        data = data - bkg

    cat = SourceCatalog(data=data,
                        segment_img=segment_map,
                        background=bkg,
                        error=bkg_rms,
                        mask=mask)

    tbl = cat.to_table()

    del data, bkg, bkg_rms, segment_map

    return tbl, cat


# noinspection PyUnresolvedReferences
def undistort_image(image, cam_maker, cam_model,
                    lens_maker, lens_model, focal_length, aperture):
    """ Undistort single image using lensfunpy """

    # Get image dimensions
    height, width = image.shape[0], image.shape[1]

    # Search the database
    db = lensfunpy.Database()
    cam = db.find_cameras(cam_maker, cam_model)[0]
    lens = db.find_lenses(cam, lens_maker, lens_model)[0]

    # Create a modifier
    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(focal=focal_length,
                   distance=0.,
                   aperture=aperture,
                   scale=1.0,
                   flags=8,
                   pixel_format=np.uint16)

    # Apply geometric and distortion correction to modifier
    # undistort_coords = mod.apply_geometry_distortion()
    maps = mod.apply_geometry_distortion()
    map_x = maps[:, :, 0]
    map_y = maps[:, :, 1]

    # Use opencv to undistort the image
    # noinspection PyTypeChecker
    image_undistorted = cv2.remap(src=image,
                                  map1=map_x,
                                  map2=map_y,
                                  interpolation=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_TRANSPARENT)

    return image_undistorted
