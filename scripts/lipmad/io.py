#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import numpy as np
import pandas as pd
import rawpy
from pathlib import Path
import photutils
from astropy.io import fits
from astropy.table import QTable
import joblib
from datetime import datetime
import geopandas as gpd

from . import base_conf


def load_raw_file(filename):
    """
    Load the complete RAW image file using rawpy's `imread` function.
    Return the image object.
    """
    return rawpy.imread(filename)


def load_raw_image(filename):
    """
    Load RAW image file using rawpy's `imread` function.
    Return only the visible image data.
    """

    raw = load_raw_file(filename)
    raw_image_visible = np.array(raw.raw_image_visible, dtype=np.uint16)
    raw.close()

    # return visible raw image data
    return raw_image_visible


def load_raw_image_processed(raw_filename, **kwargs):
    """
    Load a raw file using rawpy's `imread` function and post-process it.
    Return the post-processed image data.
    """

    # demosaic_algorithm = None, output_bps = 16,
    # no_auto_bright = False, user_wb = None, use_camera_wb = False

    # Open the RAW file
    with rawpy.imread(raw_filename) as raw:
        # post-process this file to obtain a numpy ndarray of shape (h,w,c)
        img_post = raw.postprocess(**kwargs)

    return img_post


def load_segment_map(path, file_base):
    """ Load segmentation map """
    # Segmentation map
    segm_file = Path(path) / f'{file_base}_detections_segm.fits'
    segm = fits.open(segm_file)[0].data
    segm = photutils.segmentation.SegmentationImage(segm)
    return segm


def save_segment_map(segm, outdir, filename_base):
    """ Save segmentation map of detected objects """

    segm_hdu = fits.PrimaryHDU(segm.data.astype(np.uint16))
    segm_hdu.writeto(Path(outdir) / f'{filename_base}_detections_segm.fits',
                     overwrite=True)


def save_photometry_table(tbl, filename, is_pd=False):
    """ Save photometry table"""

    if not is_pd:
        tbl.write(filename, overwrite=True)
    else:
        tbl.to_csv(filename, index=False)


def load_photometry_table(filename, as_pd=False):
    """ Load photometry table from file """
    # Catalog: the ecsv format preserves units for loading in Python notebooks
    if not as_pd:
        tbl = QTable.read(filename)
    else:
        tbl = pd.read_csv(filename)
    return tbl


def save_data(data, file_name):
    """"""
    # save map and mask in dict
    with gzip.open(file_name, 'wb', compresslevel=6) as fp:
        joblib.dump(data, fp)


def load_data(file_name):
    """"""
    with gzip.open(file_name, 'rb') as fp:
        data = joblib.load(fp)
    return data


def save_fits(data, hdr_dict, file_name):
    """"""

    N_HDUs = data.shape[0]

    # Create the FITS object and the header
    hdu = fits.PrimaryHDU()
    for k, v in hdr_dict.items():
        hdu.header.set(keyword=k, value=v[0], comment=v[1])
    hdu.header.set(keyword="NAXIS", value=2)
    hdu.header.set(keyword="IMAGETYP",
                   value='LIGHT',
                   before='FNUMBER')
    hdu.header.set(keyword='BITPIX', value=16, comment='bits per data value')
    hdu.header.set(keyword='DATE',
                   value=datetime.utcnow().isoformat('T'),
                   comment='file creation UTC timestamp')

    for i in range(N_HDUs):
        color_key = base_conf.SUPPORTED_COLORS[i]
        color_data = base_conf.DEF_COLORS[color_key]
        filt = f"_{color_data[0]}.fits"
        fname = str(file_name).replace(".fits", filt)

        hdu.header.set(keyword='FILTER', value=color_data[0])
        hdu.data = data[i, :, :]

        hdu.writeto(fname, overwrite=True, output_verify="silentfix")


def write_geojson_file(geojson_file, geojson_dir, feature_collection):
    """
    Write the GeoJSON feature collection to a file.

    Args:
        geojson_file (str): The filename for the GeoJSON output.
        geojson_dir (Path): The directory where the GeoJSON file should be saved.
        feature_collection (dict): The GeoJSON feature collection to be written.
    """
    file_path = Path(geojson_dir) / geojson_file
    import geojson
    try:
        with open(file_path, "w") as file:
            geojson.dump(feature_collection, file, indent=4)
    except Exception as e:
        base_conf.log.critical(f"Error writing GeoJSON file: {e}")


def read_geojson(geojson_file):
    return gpd.read_file(geojson_file, engine="pyogrio", encoding='utf-8')
