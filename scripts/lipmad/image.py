#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import collections

import logging
import math
from mpmath import mp, radians, sqrt
import numpy as np
from shapely.geometry import Polygon
import os

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# import numpy as np
# from pathlib import Path
import seaborn as sns
import pandas as pd
from photutils.segmentation import SegmentationImage
from astropy.table import join
from astropy.stats import sigma_clipped_stats

from transformations import (euler_from_quaternion,
                             quaternion_from_euler, quaternion_matrix, quaternion_multiply,
                             rotation_matrix)

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score

from . import base_conf
from . import camera
from . import conversions
from . import detection
from .exif import Exif
from . import general
from . import raw
from . import io
from . import project

from .new_georef import HighAccuracyFOVCalculator
from .create_geojson import create_geojson_feature
from .Utils.raster_utils import *

from props import getNode

calib_dir = Path('../../data/mavic3_calibration/calibration/')

# Empty slice that just selects all data - used as default argument
all_data = np.s_[:]
filters = ['R', 'G', 'B', 'G2']

np.seterr(divide='ignore', invalid='ignore')


class CenteredNorm(Normalize):
    def __init__(self, vcenter=0, halfrange=None, clip=False):
        self.vcenter = vcenter
        self.halfrange = halfrange
        Normalize.__init__(self, None, None, clip)

    def autoscale_None(self, A):
        super().autoscale_None(A)
        if self.vmin is None or self.vmax is None:
            self.vmin, self.vmax = A.min(), A.max()
        if self.halfrange is None:
            self.halfrange = max(abs(self.vmin - self.vcenter), abs(self.vmax - self.vcenter))
        self.vmin, self.vmax = self.vcenter - self.halfrange, self.vcenter + self.halfrange

    def __call__(self, value, clip=None):
        return np.ma.masked_array(np.interp(value, [self.vmin, self.vmax], [0, 1]), mask=np.ma.getmask(value))


# Create a custom normalization: center is white at 0
class MidpointNormalize(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self, vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax
        if clip:
            result = np.clip(result, vmin, vmax)
        result = (1 - 2 * (result - self.midpoint) / (vmax - vmin - 2 * self.midpoint + 1e-12)) / 2
        return np.ma.masked_array(result, mask=np.ma.getmask(result))


class Image(object):
    """Handle exif data of an image file"""
    calibration_data_all = ["bias_map", "readnoise", "dark_current", "iso_lookup_table",
                            "flatfield_map", "bkg_data", "correction_map"]

    def __init__(self,
                 log: logging.Logger = base_conf.log,
                 log_level: int = 20):
        """ Constructor with default values """

        # set log level
        self.bkg_image_rms = None
        log.setLevel(log_level)

        self.correction_map_c0 = None
        self.correction_map_c1 = None
        self.bkg_file_gray = None
        self.bkg_file_RGB = None
        self.bkg_image = None
        self.segment_map = None
        self.segment_map_undist = None
        self.segment_src_table = None
        self.src_mask = None
        self.xyz_data = None
        self.image_XYZ = None
        self.image_RGB = None
        self.image_RGB_red = None
        self.image_RGB_cal = None

        self.detect_catalog_file = None
        # self.detect_image_file = None
        self.exif = None
        self.fits_image_file = None
        self.jpg_image_file = None
        self.folder_name = None
        self.georef_image_file = None
        self.georef_image_br_file = None
        self.georef_map_file = None
        self.img_geojson = None
        self.area_geojson = None

        self.segm_img_geojson = None
        self.segm_area_geojson = None

        self.image_corrected = None
        self.image_file = None
        self.log = log
        self.name = None
        self.node = None
        self.rgb_mask = None
        self.phot_cat_rgb_img = None
        self.phot_cat_cal_img = None
        # self.phot_image_file = None
        self.segment_file = None

        self.geo_img_props = {}
        self.feature_collection = {}

        self.processed_dir = None
        self.cache_dir = None
        self.meta_dir = None
        self.plot_dir = None
        self.segm_plot_dir = None

        self.detect_dict = {}
        self.segment_dict = {}

        # get general image parameter
        self.cam2body = np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ],
            dtype=float)
        self.body2cam = np.linalg.inv(self.cam2body)

        image_params = camera.get_image_params()
        lens_params = camera.get_lens_params()

        self.pixel_size = image_params[0]
        self.width_px_full = image_params[1]
        self.height_px_full = image_params[2]
        self.width_px = image_params[3]
        self.height_px = image_params[4]
        self.bias = image_params[5]
        self.n_colors = image_params[6]
        self.color_description = image_params[7]
        self.bayer_pattern = image_params[8]
        self.bit_depth = image_params[9]
        self.image_shape = (self.height_px, self.width_px)
        self.image_shape_full = (self.height_px_full, self.width_px_full)

        self.saturation = 2 ** self.bit_depth - 1
        self.bands = self.color_description
        self.bayer_map = self._generate_bayer_map()

        self.ccd_width_mm = lens_params[0]
        self.ccd_height_mm = lens_params[1]
        self.focal_len_mm = lens_params[2]
        self.h_fov_deg = lens_params[3]
        self.v_fov_deg = lens_params[4]
        self.maker = lens_params[6]
        self.model = lens_params[7]
        self.scale_width = lens_params[8]
        self.scale_height = lens_params[9]

        # Calculate the angular size of a pixel
        delta_theta_x = np.deg2rad(self.h_fov_deg) / self.width_px
        delta_theta_y = np.deg2rad(self.v_fov_deg) / self.height_px
        self.omega_pixel = delta_theta_x * delta_theta_y

    def _generate_bayer_map(self, dtype='int'):
        """
        Generate a Bayer map, with the Bayer channel (RGBG2) for each pixel.
        """
        bayer_map = np.zeros(self.image_shape_full, dtype=dtype)
        bayer_map[0::2, 0::2] = self.bayer_pattern[0][0]
        bayer_map[0::2, 1::2] = self.bayer_pattern[0][1]
        bayer_map[1::2, 0::2] = self.bayer_pattern[1][0]
        bayer_map[1::2, 1::2] = self.bayer_pattern[1][1]
        return bayer_map

    def init_image(self, image_file: str,
                   image_base: str,
                   image_loc_name: str):
        """"""

        self.exif = Exif(image_file=image_file)

        self.image_file = image_file
        self.name = image_base
        self.folder_name = image_loc_name
        self.node = getNode(f"/images/{image_loc_name}/{self.name}", True)
        self.feature_collection = {"type": "FeatureCollection", "features": []}
        #
        dir_node = getNode('/config/directories', True)
        analysis_dir = dir_node.getString('analysis_dir')

        # create directories
        cache_dir = Path(analysis_dir) / image_loc_name / 'cache'
        meta_dir = Path(analysis_dir) / image_loc_name / 'meta'
        plot_dir = Path(analysis_dir) / image_loc_name / 'figures'
        segm_plot_dir = Path(analysis_dir) / image_loc_name / 'figures' / 'segments'
        processed_dir = Path(analysis_dir) / image_loc_name / 'processed'
        state_dir = Path(analysis_dir) / image_loc_name / 'state'

        general.create_folder(cache_dir, log=self.log, create_if_needed=True)
        general.create_folder(meta_dir, log=self.log, create_if_needed=True)
        general.create_folder(processed_dir, log=self.log, create_if_needed=True)
        general.create_folder(state_dir, log=self.log, create_if_needed=True)
        general.create_folder(plot_dir, log=self.log, create_if_needed=True)
        general.create_folder(segm_plot_dir, log=self.log, create_if_needed=True)

        # set file names
        self.bkg_file_gray = cache_dir / f"{image_base}_bkg_gray.bkg"
        self.bkg_file_RGB = cache_dir / f"{image_base}_bkg_RGB.bkg"

        self.segment_file = cache_dir / f"{image_base}_segments.segm"
        # self.detect_image_file = cache_dir / f"{image_base}_detect.img"
        self.detect_catalog_file = meta_dir / f"{image_base}_detect.ecsv"

        # self.phot_image_file = cache_dir / f"{image_base}_phot.img"
        self.phot_cat_rgb_img = meta_dir / f"{image_base}_phot_img_rgb.csv"
        self.phot_cat_cal_img = meta_dir / f"{image_base}_phot_img_cal.csv"

        self.georef_map_file = processed_dir / f"{image_base}_georeferenced_segment_map.tif"
        self.georef_image_file = processed_dir / f"{image_base}_georeferenced_image.tif"
        self.georef_image_br_file = processed_dir / f"image_ratio_br_georeferenced_{image_base}.tif"

        self.img_geojson = meta_dir / f"{image_base}_geo.json"
        self.area_geojson = processed_dir / f"M_{image_loc_name}.json"

        self.segm_img_geojson = meta_dir / f"segment_bbox_geojson_image_{image_base}.json"
        self.segm_area_geojson = meta_dir / f"segment_bbox_geojson_area_{image_loc_name}.json"

        self.fits_image_file = processed_dir / f"{image_base}.fits"
        self.jpg_image_file = processed_dir / f"{image_base}.jpg"

        self.processed_dir = processed_dir
        self.cache_dir = cache_dir
        self.meta_dir = meta_dir
        self.plot_dir = plot_dir
        self.segm_plot_dir = segm_plot_dir

    def convert_raw2jpg_and_save(self):
        """"""

        image_sRGB = self.load_image_calibrated()

        sRGB_IMAGE = conversions.convert_kmn_to_mnk(image_sRGB)

        p2, p98 = np.nanpercentile(sRGB_IMAGE, (2, 99.95))
        image_data_contrast_stretched = np.clip((sRGB_IMAGE - p2) / (p98 - p2), 0, 1)

        from PIL import Image as pilImage
        # Scale the float values to 0-255 and convert to uint8
        image_array_uint8 = (image_data_contrast_stretched * 255).astype(np.uint8)

        # Convert the numpy array to a PIL Image object
        image = pilImage.fromarray(image_array_uint8, 'RGB')

        # Save the image as a JPEG file
        image.save(self.jpg_image_file, 'JPEG')

    def convert_raw2fits_and_save(self):
        """"""

        RGB_data = self.load_image_processed_basic()

        self.log.info("> Create FITS file and save to disk")

        self.exif.get_all_info()
        hdr_dict = self.exif.all_info

        # order the header dict
        hdr_dict['OBJECT'] = (self.name, 'object name')
        hdr_dict['FILENAME'] = (self.name, 'input file name')
        hdr_dict['INPUTFMT'] = ('DNG', 'input file extension')
        hdr_dict['FILTER'] = (None, 'camera filter')
        hdr_dict_ordered = collections.OrderedDict([(k, hdr_dict[k])
                                                    for k in base_conf.HDR_KEYS_ORDERED
                                                    if k in hdr_dict])

        io.save_fits(data=RGB_data,
                     hdr_dict=hdr_dict_ordered,
                     file_name=self.fits_image_file)

    def load_detection_data(self):
        """"""

        if not self.segment_file.exists():
            return False

        self.log.info("> Load detection results")

        segment_dict = io.load_data(self.segment_file)

        self.rgb_mask = segment_dict['rgb_mask']
        self.segment_map = segment_dict['segment_map']
        if self.segment_map is None or self.segment_map.nlabels < 3:
            return False

        self.segment_map_undist = segment_dict['segment_map_undist']
        self.src_mask = segment_dict['src_mask']

        self.segment_src_table = io.load_photometry_table(self.detect_catalog_file)

        return True

    def save_detection_data(self):
        """"""

        self.log.info("> Save detection results")

        # save segmentation map and source map
        segm_data = {
            'gray_image_bkg': self.bkg_image,
            'rgb_mask': self.rgb_mask,
            'segment_map': self.segment_map,
            'segment_map_undist': self.segment_map_undist,
            'src_mask': self.src_mask
        }
        io.save_data(segm_data, self.segment_file)

        # save catalogue
        io.save_photometry_table(self.segment_src_table, self.detect_catalog_file)
        self.log.info("  Segment data saved ...")

    @staticmethod
    def ypr_to_quat(yaw_deg, pitch_deg, roll_deg):
        quat = quaternion_from_euler(np.deg2rad(yaw_deg),
                                     np.deg2rad(pitch_deg),
                                     np.deg2rad(roll_deg),
                                     axes='szyx')
        return quat

    def set_aircraft_pose(self, lat_deg, lon_deg, alt_m,
                          yaw_deg, pitch_deg, roll_deg, flight_time=-1.0):
        # computed from euler angles
        ned2body = self.ypr_to_quat(yaw_deg, pitch_deg, roll_deg)

        ac_pose_node = self.node.getChild('aircraft_pose', True)
        ac_pose_node.setFloat('lat_deg', lat_deg)
        ac_pose_node.setFloat('lon_deg', lon_deg)
        ac_pose_node.setFloat('alt_m', alt_m)
        ac_pose_node.setFloat('yaw_deg', yaw_deg)
        ac_pose_node.setFloat('pitch_deg', pitch_deg)
        ac_pose_node.setFloat('roll_deg', roll_deg)
        ac_pose_node.setLen('quat', 4)
        for i in range(4):
            ac_pose_node.setFloatEnum('quat', i, ned2body[i])
        if flight_time > 0.0:
            self.node.setFloat("flight_time", flight_time)

    # ned = [n_m, e_m, d_m] relative to the project ned reference point
    # ypr = [yaw_deg, pitch_deg, roll_deg] in the ned coordinate frame
    # note that the matrix derived from 'quat' is inv(R) is transpose(R)
    def set_camera_pose(self, ned, yaw_deg, pitch_deg, roll_deg, opt=False):
        # computed from euler angles
        ned2cam = self.ypr_to_quat(yaw_deg, pitch_deg, roll_deg)
        if opt:
            cam_pose_node = self.node.getChild('camera_pose_opt', True)
            cam_pose_node.setBool('valid', True)
        else:
            cam_pose_node = self.node.getChild('camera_pose', True)
        for i in range(3):
            cam_pose_node.setFloatEnum('ned', i, ned[i])
        cam_pose_node.setFloat('yaw_deg', yaw_deg)
        cam_pose_node.setFloat('pitch_deg', pitch_deg)
        cam_pose_node.setFloat('roll_deg', roll_deg)
        cam_pose_node.setLen('quat', 4)
        for i in range(4):
            cam_pose_node.setFloatEnum('quat', i, ned2cam[i])

    def set_image_exposure(self, obsdate, obsdate_iso, unixtime,
                           iso, aperture, f_number, exptime, shutter_speed):
        """"""

        self.node.setFloat('aperture', aperture)
        self.node.setFloat('exptime', exptime)
        self.node.setFloat('f_number', f_number)
        self.node.setFloat('shutter_speed', shutter_speed)

        self.node.setInt('iso', iso)
        self.node.setInt('unixtime', unixtime)

        self.node.setString('obsdate', obsdate)
        self.node.setString('obsdate_iso', obsdate_iso)

    def get_image_exposure(self):
        """"""

        exposure = [
            self.node.getFloat('exptime'),
            self.node.getFloat('f_number'),
            self.node.getInt('iso')
        ]

        return exposure

    def get_image_gps(self):

        ac_pose_node = self.node.getChild('aircraft_pose', True)

        pos = [
            ac_pose_node.getFloat('lat_deg'),
            ac_pose_node.getFloat('lon_deg'),
            ac_pose_node.getFloat('alt_m')
        ]
        return pos

    def get_aircraft_pose(self):
        pose_node = self.node.getChild('aircraft_pose', True)
        lla = [pose_node.getFloat('lat_deg'),
               pose_node.getFloat('lon_deg'),
               pose_node.getFloat('alt_m')]
        ypr = [pose_node.getFloat('yaw_deg'),
               pose_node.getFloat('pitch_deg'),
               pose_node.getFloat('roll_deg')]
        quat = []
        for i in range(4):
            quat.append(pose_node.getFloatEnum('quat', i))
        return lla, ypr, quat

    def get_camera_pose(self, opt=False):
        if opt:
            pose_node = self.node.getChild('camera_pose_opt', True)
        else:
            pose_node = self.node.getChild('camera_pose', True)
        ned = []
        for i in range(3):
            ned.append(pose_node.getFloatEnum('ned', i))
        ypr = [pose_node.getFloat('yaw_deg'),
               pose_node.getFloat('pitch_deg'),
               pose_node.getFloat('roll_deg')]
        quat = []
        for i in range(4):
            quat.append(pose_node.getFloatEnum('quat', i))
        return ned, ypr, quat

    # cam2body rotation matrix (M)
    def get_cam2body(self):
        return self.cam2body

    # body2cam rotation matrix (IM)
    def get_body2cam(self):
        return self.body2cam

    # ned2body (R) rotation matrix
    def get_ned2body(self, opt=False):
        return np.matrix(self.get_body2ned(opt)).T

    # body2ned (IR) rotation matrix
    def get_body2ned(self, opt=False):
        # ned, ypr, quat = self.get_camera_pose(opt)
        ned, ypr, quat = self.get_aircraft_pose()
        return quaternion_matrix(np.array(quat))[:3, :3]

    def demosaick(self, data, selection=all_data):
        """
        Demosaick data using this camera's Bayer pattern.
        """
        # Select the relevant data
        bayer_map = self.bayer_map[selection]

        # Demosaick the data
        RGBG_data = raw.demosaick(bayer_map, data, color_desc=self.bands)
        return RGBG_data

    def generate_bias_map(self):
        """
        Generate a Bayer-aware map of bias values from the camera information.
        """
        bayer_map = self._generate_bayer_map()
        for j, bias_value in enumerate(self.bias):
            bayer_map[bayer_map == j] = bias_value
        return bayer_map

    def generate_correction_map(self, calib_coeff):
        """
        Generate a Bayer-aware map of bias values from the camera information.
        """
        bayer_map = self._generate_bayer_map(dtype='float64')
        for j, corr_value in enumerate(calib_coeff):
            bayer_map[bayer_map == j] = corr_value
        return bayer_map

    def _load_bias_map(self):
        """
        Load a bias map from the root folder or from the camera information.
        """
        # First try using a data-based bias map from file
        try:
            filename = calib_dir / "mavic3_bias.npy"
            bias_map = np.load(filename)

        # If a data-based bias map does not exist or cannot be loaded, use camera information instead
        except (FileNotFoundError, OSError, TypeError):
            bias_map = self.generate_bias_map()

        self.bias_map = bias_map

    def _load_dark_current_map(self):
        """
        Load a dark current map from the root folder - if none is available, return 0 everywhere.
        """
        # Try to use a dark current map from file
        try:
            filename = calib_dir / "mavic3_dark_current_normalised.npy"
            dark_current = np.load(filename)

        # If a dark current map does not exist, return an empty one, and warn the user
        except (FileNotFoundError, OSError, TypeError):
            dark_current = np.zeros(self.image_shape_full)
            self.log.warning(f"Could not find a dark current map in the folder `{calib_dir}` - using all 0 instead")

        # Whatever dark current map was used, save it to this object
        self.dark_current = dark_current

    def _load_iso_normalisation(self):
        """
        Load an ISO normalisation look-up table from the root folder.
        If none is available, make an estimate from the camera's ISO range.
        """
        # Try to use a lookup table from file
        try:
            filename = calib_dir / "mavic3_iso_normalisation_lookup_table.csv"
            lookup_table = np.loadtxt(filename, delimiter=",").T

        # If a lookup table cannot be found, assume a linear relation and warn the user
        except (FileNotFoundError, OSError, TypeError):
            self.log.warning(f"No lookup table found.")
            lookup_table = None

        # Whatever method was used, save the lookup table
        self.iso_lookup_table = lookup_table

    def _load_gain_map(self):
        """
        Load a gain map from the root folder.
        """
        # Try to use a gain map from file
        try:
            filename = calib_dir / "mavic3_gain.npy"
            gain_map = np.load(filename)

        # If a gain map cannot be found, do not use any, and warn the user
        except (FileNotFoundError, OSError, TypeError):
            gain_map = None
            self.log.warning(f"No gain map found.")

        self.gain_map = gain_map

    def _load_flatfield_correction(self):
        """
        Load a flatfield correction model from the root folder, and generate a correction map.
        """
        # Try to use a flatfield model from file
        try:
            filename = calib_dir / "mavic3_flatfield_parameters.csv"
            data = np.loadtxt(filename, delimiter=",")
            parameters, errors = data[:7], data[7:]
            correction_map = general.apply_vignette_radial(self.image_shape_full,
                                                           parameters)

        # If a flatfield map cannot be found, do not use any, and warn the user
        except (FileNotFoundError, OSError, TypeError):
            correction_map = None
            self.log.warning("No flatfield model found.")

        self.flatfield_map = correction_map

    def _load_calibration_coefficient(self):
        """
        Load the calibration coefficients from the root folder
        """

        # Try to use a flatfield model from file
        try:
            filename = calib_dir / "calibration_coeff_RGB_mavic3.csv"
            data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=(1, 3)).tolist()
            data.append(data[1])
            data = np.array(data)
            correction_map_c1 = self.generate_correction_map(data[:, 0])
            correction_map_c0 = self.generate_correction_map(data[:, 1])

        # If a flatfield map cannot be found, do not use any, and warn the user
        except (FileNotFoundError, OSError, TypeError):
            correction_map_c0 = None
            correction_map_c1 = None
            self.log.warning("No calibration model found.")

        self.correction_map_c1 = correction_map_c1
        self.correction_map_c0 = correction_map_c0

    def correct_bias(self, data, selection=all_data):
        """
        Correct data for bias using this sensor's data.
        Bias data are loaded from the root folder or from the camera information.
        """
        # If a bias map has not been loaded yet, do so
        if not hasattr(self, "bias_map"):
            self._load_bias_map()

        # Select the relevant data
        bias_map = self.bias_map[selection]

        # Apply the bias correction
        data_corrected = general.correct_bias_from_map(bias_map, data)
        return data_corrected

    def correct_dark_current(self, exposure_time, data, selection=all_data):
        """
        Calibrate data for dark current using this sensor's data.
        Dark current data are loaded from the root folder or estimated 0 in all pixels,
        if no data were available.
        """
        # If a dark current map has not been loaded yet, do so
        if not hasattr(self, "dark_current"):
            self._load_dark_current_map()

        # Select the relevant data
        dark_current = self.dark_current[selection]

        # Apply the dark current correction
        data_corrected = general.correct_dark_current_from_map(dark_current, exposure_time, data)
        return data_corrected

    def normalise_iso(self, iso_values, data):
        """
        Normalise data for their ISO speed using this sensor's lookup table.
        The ISO lookup table is loaded from the root folder.
        """
        # If a lookup table has not been loaded yet, do so
        if not hasattr(self, "iso_lookup_table"):
            self._load_iso_normalisation()

        # Apply the ISO normalisation
        data_corrected = general.normalise_iso(self.iso_lookup_table, iso_values, data)
        return data_corrected

    def convert_to_photoelectrons(self, data, selection=all_data):
        """
        Convert data from ADU to photoelectrons using this sensor's gain data.
        The gain data are loaded from the root folder.
        """
        # If a gain map has not been loaded yet, do so
        if not hasattr(self, "gain_map"):
            self._load_gain_map()

        # Assert that a gain map was loaded
        assert self.gain_map is not None, "Gain map unavailable"

        # Select the relevant data
        gain_map = self.gain_map[selection]

        # If a gain map was available, apply it
        data_converted = general.convert_to_photoelectrons_from_map(gain_map, data)
        return data_converted

    def correct_flatfield(self, data, selection=all_data):
        """
        Correct data for flatfield using this sensor's flatfield data.
        The flatfield data are loaded from the root folder.
        """
        # If a flatfield map has not been loaded yet, do so
        if not hasattr(self, "flatfield_map"):
            self._load_flatfield_correction()

        # Assert that a flatfield map was loaded
        assert self.flatfield_map is not None, "Flatfield map unavailable"

        # Select the relevant data
        flatfield_map = self.flatfield_map[selection]

        # If a flatfield map was available, apply it
        data_corrected = general.correct_flatfield_from_map(flatfield_map, data)
        return data_corrected

    def correct_flux(self, data, selection=all_data):
        """

        """

        # self._load_calibration_coefficient()

        # If a flatfield map has not been loaded yet, do so
        if not hasattr(self, "correction_map"):
            self._load_calibration_coefficient()

        # Assert that a flatfield map was loaded
        assert self.correction_map_c0 is not None, "Calibration map unavailable"
        assert self.correction_map_c1 is not None, "Calibration map unavailable"

        # Select the relevant data
        correction_map = (self.correction_map_c1[selection], self.correction_map_c0[selection])

        # If a flatfield map was available, apply it
        data_corrected = general.convert_flux_from_map(correction_map, data)

        return data_corrected

    convert_RGBG2_to_RGB = staticmethod(raw.convert_RGBG2_to_RGB)
    convert_CAM_to_XYZ = staticmethod(conversions.convert_CAM_to_XYZ)
    convert_XYZ_to_sRGB = staticmethod(conversions.convert_XYZ_to_sRGB)
    convert_XYZ_to_xy = staticmethod(conversions.convert_XYZ_to_xy)
    convert_xy_to_CCT = staticmethod(conversions.convert_xy_to_CCT)

    def mask_sources_rgb(self, data, limit=65535):
        """Make a source mask to exclude saturated sources from bkg estimation and photometry."""
        mask = np.any(data > limit, axis=0)
        self.rgb_mask = mask
        return mask

    def _load_background_maps(self, bkg_file):
        """"""

        try:
            self.log.info(f"> Load 2D background from file: {bkg_file}")
            bkg_map = io.load_data(bkg_file)

        # If a gain map cannot be found, do not use any, and warn the user
        except (FileNotFoundError, OSError, TypeError):
            bkg_map = None
            self.log.warning(f"  No background data file found.")

        self.bkg_data = bkg_map

    def get_background(self, data, filt, force_est=False, **kwargs):
        """Load or estimate background from the image"""
        self.bkg_image = None

        bkg_file = self.bkg_file_gray if filt == 'gray' else self.bkg_file_RGB
        # print(bkg_file)

        # If a bias map has not been loaded yet, do so
        # if not hasattr(self, "bkg_data"):
        #     self._load_background_maps(bkg_file)
        self._load_background_maps(bkg_file)
        # print(self.bkg_data.keys())

        bkg_data = collections.OrderedDict() if self.bkg_data is None else self.bkg_data

        if filt not in bkg_data or force_est:
            self.log.info("> Estimate 2D background ")
            bkg = detection.get_background(data, **kwargs)
            bkg_data[filt] = {
                'bkg_image': bkg.background,
                'bkg_image_rms': bkg.background_rms
            }
            io.save_data(bkg_data, bkg_file)

        self.bkg_image = bkg_data[filt]['bkg_image']
        self.bkg_image_rms = bkg_data[filt]['bkg_image_rms']

        return bkg_data

    def find_sources(self, data, **kwargs):
        """"""

        # separate out the rgb mask
        rgb_mask = kwargs.pop('rgb_mask')

        self.log.info("> Extract sources from detection image")

        # detect sources
        segment_map = detection.extract_sources(data, self.bkg_image, self.bkg_image_rms, **kwargs)
        if segment_map is None or segment_map.nlabels < 3:
            return

        # create source mask from the segmentation map
        src_mask = segment_map.make_source_mask(footprint=None)
        src_mask = np.where(src_mask, 1, 0)[:, :, np.newaxis]

        data_undist = self.undistort_data(segment_map.data, is_segm=True)
        segment_map_undist = SegmentationImage(data_undist)

        self.log.info("  Make a source catalogue from segments")

        # convert the segment map to a table
        src_table, _ = detection.get_catalog_from_segments(data, segment_map,
                                                           None, None,
                                                           mask=rgb_mask)

        # undistort the segment map and source table
        src_table_undist, _ = detection.get_catalog_from_segments(data, segment_map_undist,
                                                                  None, None,
                                                                  mask=rgb_mask)

        src_table_undist_cut = src_table_undist['label', 'xcentroid', 'ycentroid',
        'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax']
        src_table_undist_cut.rename_column('xcentroid', 'xcentroid_undist')
        src_table_undist_cut.rename_column('ycentroid', 'ycentroid_undist')
        src_table_undist_cut.rename_column('bbox_xmin', 'bbox_xmin_undist')
        src_table_undist_cut.rename_column('bbox_xmax', 'bbox_xmax_undist')
        src_table_undist_cut.rename_column('bbox_ymin', 'bbox_ymin_undist')
        src_table_undist_cut.rename_column('bbox_ymax', 'bbox_ymax_undist')

        src_table = join(src_table, src_table_undist_cut, keys='label',
                         join_type='outer', metadata_conflicts='silent')
        # print(src_table)
        # fig, ax = plt.subplots(1, 2, sharex="all", sharey="all", figsize=(10, 6))
        #
        # cmap = segment_map.make_cmap(seed=12345)
        # ax[0].imshow(segment_map.data, cmap=cmap, interpolation='nearest')
        #
        # cmap = segment_map_undist.make_cmap(seed=12345)
        # ax[1].imshow(segment_map_undist.data, cmap=cmap, interpolation='nearest')
        #
        # plt.show()

        # store results
        self.segment_map = segment_map
        self.segment_map_undist = segment_map_undist
        self.segment_src_table = src_table
        self.src_mask = src_mask

        return segment_map, src_table, src_mask

    def load_image_processed_basic(self):
        """"""

        self.log.info("> Load image and apply basic corrections")

        exptime, f_number, iso = self.get_image_exposure()

        # load the raw image
        image_raw = io.load_raw_image(self.image_file)

        # correct bias
        corrected_bias = self.correct_bias(image_raw)

        # ADU
        corrected_dark = self.correct_dark_current(exptime, corrected_bias)
        self.log.info("  Corrected for bias and dark current")

        # ADU
        corrected_flat = self.correct_flatfield(corrected_dark)
        self.log.info("  Corrected for flat")

        # norm. ADU sr^-1 s^-1
        corrected_exposure = (f_number ** 2 / exptime) * self.normalise_iso(iso, corrected_flat)
        self.log.info("  Corrected for exposure parameters ADU -> norm. ADU sr^-1 s^-1")

        self.log.info("  Demosaick the image")
        # demosaick
        RGBG_data = self.demosaick(corrected_exposure)

        # map to RGB image
        RGB_data = self.convert_RGBG2_to_RGB(RGBG_data)

        self.image_RGB_red = RGB_data

        return RGB_data

    def load_XYZ_image(self):
        """"""

        self.log.info("> Load image and convert to XYZ")

        # load the raw image
        image_raw = io.load_raw_image(self.image_file)

        image_raw_demos = self.demosaick(image_raw)
        self.mask_sources_rgb(image_raw_demos, 0.98*self.saturation)

        # correct bias
        image_raw_bias_corrected = self.correct_bias(image_raw)

        # correct flat
        image_raw_flat_corrected = self.correct_flatfield(image_raw_bias_corrected)
        self.log.info("  Corrected for bias and flat")

        self.log.info("  Demosaick the image")
        # demosaick
        RGBG_data = self.demosaick(image_raw_flat_corrected)

        # map to RGB image
        RGB_data = self.convert_RGBG2_to_RGB(RGBG_data)

        # convert from (3, m, n) to (m, n, 3)
        RGB_data = conversions.convert_kmn_to_mnk(RGB_data)

        min_val = np.amin(RGB_data, axis=(0, 1)).reshape(1, 3)
        shifted_image = RGB_data - min_val
        max_val = np.amax(shifted_image, axis=(0, 1)).reshape(1, 3)

        self.log.info("  Normalize the image")

        # normalize to maximum value
        RGB_data = shifted_image / max_val

        self.log.info("> Convert Camera RGB -> XYZ image")
        # get colour matrices, white balance, and reference illuminant
        self.exif.get_colour_info()

        (as_shot,
         calibration_illuminant1,
         calibration_illuminant2,
         color_matrix_1,
         color_matrix_2
         ) = self.exif.colour_info

        # convert RGB to XYZ
        XYZ_data = self.convert_CAM_to_XYZ(RGB_data, as_shot,
                                           color_matrix_1, color_matrix_2)

        # print(XYZ_data[:, :, 1].min(), XYZ_data[:, :, 1].max(), np.argmax(XYZ_data[:, :, 1]))
        # plt.figure()
        # plt.imshow(XYZ_data[:, :, 1])
        # plt.show()

        self.image_XYZ = XYZ_data

        return XYZ_data

    def load_image_calibrated(self):
        """"""

        self.log.info("> Load image, correct and apply calibration coefficients")

        exptime, f_number, iso = self.get_image_exposure()

        # load the raw image
        image_raw = io.load_raw_image(self.image_file)

        # correct bias
        corrected_bias = self.correct_bias(image_raw)

        # ADU
        corrected_dark = self.correct_dark_current(exptime, corrected_bias)
        self.log.info("  Corrected for bias and dark current")

        # ADU
        corrected_flat = self.correct_flatfield(corrected_dark)
        self.log.info("  Corrected for flat")

        # norm. ADU sr^-1 s^-1
        # print(f_number, exptime, iso)
        corrected_exposure = (f_number ** 2 / exptime) * self.normalise_iso(iso, corrected_flat)
        self.log.info("  Corrected for exposure parameters ADU -> norm. ADU sr^-1 s^-1")

        # norm. ADU m^-2 sr^-1 s^-1
        # pixel_area_m = (self.pixel_size * 1e-6) ** 2
        # corrected_pixel_size = corrected_exposure / pixel_area_m  # norm. ADU m^-2 sr^-1 s^-1
        # self.log.info("  Corrected for pixel size norm. ADU sr^-1 s^-1 -> norm. ADU m^-2 sr^-1 s^-1")

        corrected_pixel_size = self.correct_flux(corrected_exposure)
        self.log.info("  Apply calibration coefficients norm. ADU sr^-1 s^-1 -> W/m²")

        # demosaick
        RGBG_data = self.demosaick(corrected_pixel_size)
        self.log.info("  Demosaick the image RGBG2 -> RGB")

        RGB_image_cal = self.convert_RGBG2_to_RGB(RGBG_data)

        self.image_RGB_cal = RGB_image_cal

        return RGB_image_cal

    @staticmethod
    def calculate_physical_luminance(radiance_image):
        tbl = pd.read_csv(calib_dir / 'luminance_coeff_RGB_mavic3.csv')

        r_coef, g_coef, b_coef = tbl['C_r'].values[0], tbl['C_g'].values[0], tbl['C_b'].values[0]

        # Calculate luminance
        luminance = (radiance_image[0, :, :] * r_coef +
                     radiance_image[1, :, :] * g_coef +
                     radiance_image[2, :, :] * b_coef)
        return luminance

    def undistort_data(self, data, is_segm=False):
        """"""

        _, f_number, _ = self.get_image_exposure()

        if not is_segm:
            data = conversions.convert_kmn_to_mnk(data)

        data_undist = detection.undistort_image(data, self.maker, self.model,
                                                self.maker, self.model,
                                                self.focal_len_mm, f_number)

        return data_undist

    def get_photometry(self, rgb_image, is_rgb=True, force_est=False, **kwargs):
        """"""

        tbl = self.segment_src_table
        segm = self.segment_map
        img_str = 'RGB' if is_rgb else 'calibrated'
        self.log.info(f'> Measure photometry in the {img_str} image')
        for i in range(rgb_image.shape[0]):
            filt = filters[i]
            data = rgb_image[i, :, :]
            self.log.info(f'  Process filter: {filt}')

            # bkg = self.get_background(data, filt, force_est=force_est, **kwargs)
            #
            # background = bkg[filt]['bkg_image']
            # background_rms = bkg[filt]['bkg_image_rms']
            mask = segm.make_source_mask(footprint=None)
            mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)

            data -= median
            data[data < 0] = 0

            filttbl, filtcat = detection.get_catalog_from_segments(data, segm, None,
                                                                   None,
                                                                   mask=self.rgb_mask)

            tbl[filt + '_flux'] = filttbl['segment_flux']
            tbl[filt + '_fluxerr'] = filttbl['segment_fluxerr']
            del data, filttbl, filtcat  # , bkg, background, background_rms

        # convert to pandas
        df = tbl.to_pandas()

        # sort by flux
        sorted_df = df.sort_values(by=['G_flux'], ascending=False).reset_index(drop=True)

        # calculate some parameter
        idx_list = ['B/G', 'G/R', 'R/G', 'B/R',  # ratios
                    'B-G', 'G-R', 'R-G', 'B-R',  # magnitude differences
                    'R_avg', 'G_avg', 'B_avg',  # mean flux
                    'B/G_avg', 'G/R_avg', 'R/G_avg', 'B/R_avg',  # mean of ratios
                    'B-G_avg', 'G-R_avg', 'R-G_avg', 'B-R_avg',  # mean magnitude difference
                    'R_mag', 'G_mag', 'B_mag',  # mean flux
                    'R_mag_avg', 'G_mag_avg', 'B_mag_avg',  # mean flux
                    'R_mag_surf', 'G_mag_surf', 'B_mag_surf',  # mean flux
                    'B/GR', 'B/GR_avg']
        sorted_df[idx_list] = (
            sorted_df.apply(calculate_photometric_vars, args=(rgb_image, segm, idx_list, self.rgb_mask),
                            axis=1)
        )

        # save the result
        file_name = self.phot_cat_rgb_img if is_rgb else self.phot_cat_cal_img
        io.save_photometry_table(sorted_df, file_name, True)

        # sorted_df.plot(kind='scatter', x='B/G', y='G/R')
        # sorted_df.plot(kind='scatter', x='B-G_avg', y='G-R_avg')
        # sorted_df.plot(kind='scatter', x='B/G_avg', y='G/R_avg')
        # sorted_df.plot(kind='scatter', x='B-G_avg', y='G-R_avg')
        # sorted_df.plot(kind='scatter', x='G_mag', y='R_mag')
        # sorted_df.plot(kind='scatter', x='G_mag', y='B_mag')
        # sorted_df.plot(kind='scatter', x='G_mag_avg', y='R_mag_avg')
        # sorted_df.plot(kind='scatter', x='G_mag_avg', y='B_mag_avg')
        # fig = plt.figure(figsize=(10, 10))
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(sorted_df['R_mag'], sorted_df['G_mag'], sorted_df['B_mag'])
        # fig = plt.figure(figsize=(10, 10))
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(sorted_df['R_avg'], sorted_df['G_avg'], sorted_df['B_avg'])
        # plt.show()
        # sorted_df.plot(kind='scatter', x='R_avg', y='B_avg')
        # sorted_df.plot(kind='scatter', x='xcentroid', y='ycentroid', c='R_mag', colormap='viridis')
        # sorted_df.plot(kind='scatter', x='xcentroid', y='ycentroid', c='R_mag_avg', colormap='viridis')
        # sorted_df.plot(kind='scatter', x='xcentroid', y='ycentroid', c='R_mag_surf', colormap='viridis')
        # sorted_df.plot(kind='scatter', x='xcentroid', y='ycentroid', c='G_mag', colormap='viridis')
        # sorted_df.plot(kind='scatter', x='xcentroid', y='ycentroid', c='G_mag_avg', colormap='viridis')
        # sorted_df.plot(kind='scatter', x='xcentroid', y='ycentroid', c='G_mag_surf', colormap='viridis')
        # sorted_df.plot(kind='scatter', x='xcentroid', y='ycentroid', c='B_mag', colormap='viridis')
        # sorted_df.plot(kind='scatter', x='xcentroid', y='ycentroid', c='B_mag_avg', colormap='viridis')
        # sorted_df.plot(kind='scatter', x='xcentroid', y='ycentroid', c='B_mag_surf', colormap='viridis')

        # sorted_df.plot(kind='scatter', x='area', y='R_avg')
        # sorted_df.plot(kind='scatter', x='area', y='G_avg')
        # sorted_df.plot(kind='scatter', x='area', y='B_avg')

        # fig, ax = plt.subplots(1, 2, sharex="all", sharey="all", figsize=(10, 6))
        #
        # ax[0].imshow(rgb_image[1, :, :], interpolation='nearest', vmin=0,
        #              vmax=np.nanpercentile(rgb_image[1, :, :], 99.5))
        # for i in segm.bbox:
        #     ax[0].add_patch(i.as_artist(facecolor=None, edgecolor='red', fill=False, lw=2))
        # cmap = segm.make_cmap(seed=12345)
        # ax[1].imshow(segm.data, cmap=cmap, interpolation='nearest')
        #
        # plt.show()

    def analyse_image(self, image_id, use_rgb=False):

        # load the table
        file_name = self.phot_cat_rgb_img if use_rgb else self.phot_cat_cal_img
        if not file_name.exists():
            self.log.warning(f"> No photometry data found for: {self.name}. "
                             f"Skipping object.")
            return False

        # load the table
        df = io.load_photometry_table(file_name, as_pd=True)

        # drop nan rows from the table
        bbox_columns = df.filter(regex='^bbox_.*_undist$').columns

        rows_with_nan = df[bbox_columns].isna().any(axis=1)

        df_clean = df[~rows_with_nan].copy()

        df_clean = df_clean.reset_index(drop=True)

        # create a new column for the id
        df_clean['id'] = image_id
        df_clean['imageID'] = self.name
        df_clean['areaID'] = self.folder_name

        self.log.info(f'> Find outlier in the image dataset')

        # get outliers on image level
        df_ml = self.get_outliers_ML(df_clean, suffix='_image')
        io.save_photometry_table(df_ml, file_name, is_pd=True)

        # georeference segment bbox and position
        self.log.info(f'> Georeference image segment bounding boxes')
        df_ml_geo, feature_df = self.create_geojson_image(df_ml)

        return df_ml_geo, feature_df

    @staticmethod
    def get_outliers_ML(data_df, suffix='_image'):

        z_col = 'B/R_avg'

        df = data_df.copy()
        df[f'outlier{suffix}'] = data_df[z_col].apply(lambda x: -1 if x >= 1 else 1)
        df.loc[data_df['G_flux'].idxmax(), f'outlier{suffix}'] = 2
        df.loc[data_df['G_avg'].idxmax(), f'outlier{suffix}'] = 3
        df.loc[data_df['B/R_avg'].idxmax(), f'outlier{suffix}'] = 4
        df.loc[data_df['B/R'].idxmax(), f'outlier{suffix}'] = 5
        df.loc[data_df['B/GR'].idxmax(), f'outlier{suffix}'] = 6
        df.loc[data_df['B/GR_avg'].idxmax(), f'outlier{suffix}'] = 7

        return df

        # n_clusters = 2
        # features_columns = [
        #     # 'B-G',
        #     # 'G-R',
        #     'B/G',
        #     'G/R',
        #     'B/R',
        #     # 'B-R',
        #     # 'R_avg',
        #     # 'G_avg',
        #     # 'B_avg',
        #     # 'R_mag',
        #     # 'G_mag',
        #     # 'B_mag',
        #     # 'R_mag_avg',
        #     # 'G_mag_avg',
        #     # 'B_mag_avg',
        #     # 'area',
        #     # 'R_mag_surf',
        #     'G_mag_surf',
        #     # 'B_mag_surf'
        # ]
        #
        # # Applying Robust Scaling
        # robust_scaler = RobustScaler()
        # # robust_scaler = StandardScaler()
        #
        # scaled_features = robust_scaler.fit_transform(df[features_columns])
        # df_scaled = pd.DataFrame(scaled_features,
        #                          columns=features_columns)
        # # import umap
        # #
        # # # Configuration: adjust n_neighbors and min_dist
        # # reducer = umap.UMAP(n_neighbors=15, min_dist=0.5,
        # #                     n_components=2, random_state=None)
        # # data_reduced = reducer.fit_transform(df_scaled)
        # # from sklearn.cluster import AgglomerativeClustering
        # #
        # # # Configuration: adjust the number of clusters
        # # clustering = AgglomerativeClustering(n_clusters=n_clusters,
        # #                                      linkage='complete',
        # #                                      metric='l2')
        # # labels = clustering.fit_predict(data_reduced)
        # # df_scaled['cluster'] = labels
        # #
        # # import hdbscan
        # #
        # # # Configuration: adjust min_cluster_size
        # # clusterer = hdbscan.HDBSCAN(min_cluster_size=15,
        # #                             min_samples=9,
        # #                             gen_min_span_tree=True)
        # # labels_hdbscan = clusterer.fit_predict(data_reduced)
        # # # df_scaled['cluster'] = labels_hdbscan
        # #
        # # Visualizing UMAP reduction followed by HDBSCAN
        # # fig = plt.figure(figsize=(10, 10))
        # # ax = plt.axes(projection='3d')
        # # ax.scatter3D(df['B/G'], df['G/R'], df['B/R'])
        # # plt.show()
        # # Use custom normalizer
        # norm = MidpointNormalize(midpoint=1, vmin=df['B/R'].min(),
        #                          vmax=df['B/R'].max())
        #
        # # Create the colormap
        # cmap = plt.get_cmap('bwr_r')
        # plt.figure(figsize=(10, 8))
        # plt.scatter(df['B/G'], df['G/R'], c=df['B/R'],
        #             norm=norm, edgecolor='k', cmap=cmap, s=20)
        # # plt.title('Clusters formed by HDBSCAN after UMAP Reduction')
        # plt.xlabel('B/G')
        # plt.ylabel('G/R')
        # plt.colorbar()
        # plt.figure(figsize=(10, 8))
        # plt.hist(df['B/R'], bins=30)
        #
        # plt.figure(figsize=(10, 8))
        # plt.scatter(df['B/G_avg'], df['G/R_avg'], c=df['B/R_avg'], cmap='plasma_r', s=20)
        # # plt.title('Clusters formed by HDBSCAN after UMAP Reduction')
        # plt.xlabel('B/G_avg')
        # plt.ylabel('G/R_avg')
        # plt.colorbar()
        #
        # vmin, vmax = df['B-R'].min(), df['B-R'].max()
        # from matplotlib.colors import TwoSlopeNorm
        # center_point = 0  # Change this to any value you need as the center
        # norm = TwoSlopeNorm(vcenter=center_point, vmin=vmin, vmax=vmax)
        # cmap = plt.get_cmap('bwr')
        # plt.figure(figsize=(10, 8))
        # plt.scatter(df['B-G'], df['G-R'], c=df['B-R'],
        #             norm=norm, edgecolor='k', cmap=cmap, s=20)
        # # plt.title('Clusters formed by HDBSCAN after UMAP Reduction')
        # plt.xlabel('B-G')
        # plt.ylabel('G-R')
        # plt.colorbar()
        # print(df['B-R'].min(), df['B-R'].max())
        # print(df['B-R_avg'].min(), df['B-R_avg'].max())
        # # Calculate halfrange for centered normalization
        # vcenter = 0
        # vmin, vmax = df['B-R_avg'].min(), df['B-R_avg'].max()
        # halfrange = max(abs(vmin - vcenter), abs(vmax - vcenter))
        # from matplotlib.colors import TwoSlopeNorm
        # center_point = 0  # Change this to any value you need as the center
        # norm = TwoSlopeNorm(vcenter=center_point, vmin=vmin, vmax=vmax)
        # # Create the centered normalization object
        # # norm = CenteredNorm(vcenter=vcenter, halfrange=halfrange)
        #
        # # Create the colormap
        # cmap = plt.get_cmap('bwr')
        # # plt.figure(figsize=(10, 8))
        # fig, ax = plt.subplots(1, 1, sharex="all", sharey="all", figsize=(10, 6))
        # sc = plt.scatter(df['B-G_avg'], df['G-R_avg'], c=df['B-R_avg'],
        #                  norm=norm, edgecolor='k', cmap=cmap, s=20)
        # # plt.title('Clusters formed by HDBSCAN after UMAP Reduction')
        # plt.xlabel('B-G_avg')
        # plt.ylabel('G-R_avg')
        # cbar = plt.colorbar(sc, ax=ax)
        # cbar.set_label('Value')
        # plt.show()
        # # # Reapplying KMeans Clustering on Robust-Scaled Data
        # # kmeans_robust = KMeans(n_clusters=n_clusters, random_state=0)
        # # clusters_robust = kmeans_robust.fit_predict(df_scaled)
        # # df_scaled['cluster'] = clusters_robust
        #
        # # Reapplying Isolation Forest on Robust-Scaled Data for Outlier Detection
        # iso_forest_robust = IsolationForest(n_estimators=100,
        #                                     n_jobs=-1,
        #                                     contamination='auto',
        #                                     # contamination=0.1,
        #                                     max_features=1.0,
        #                                     bootstrap=False,
        #                                     random_state=0)
        # outliers_robust = iso_forest_robust.fit_predict(
        #     df_scaled[features_columns])
        # df_scaled['outlier'] = outliers_robust
        #
        # # df[f'cluster{suffix}'] = df_scaled['cluster']
        #
        # df[f'outlier{suffix}'] = np.where(df['B-R_avg'] < 0., -1, 1)
        # # df[f'outlier{suffix}'] = df_scaled['outlier']
        # print(df[df[f'outlier{suffix}'] == -1])
        # # img = self.load_image_calibrated()
        # # image_data = conversions.convert_kmn_to_mnk(img)
        # #
        # # image_filtered = apply_median_filter(image_data, 3)
        # # print(image_filtered.shape)
        # # from scipy import ndimage
        # # result = ndimage.median_filter(image_filtered, size=9, axes=(0, 1))
        # # # import matplotlib.pyplot as plt
        # # fig, ax = plt.subplots(1, 1, sharex="all", sharey="all", figsize=(10, 6))
        # # ax.imshow(result[:, :, 2] / result[:, :, 0])
        # # plt.show()
        # return df
        #
        # # Displaying the robust-scaled DataFrame with new cluster and outlier labels
        # print(df)
        # plt.figure(figsize=(10, 6))
        #
        # # Colors for the clusters
        # color_map = {0: 'magenta', 1: 'lime', 2: 'magenta'}
        #
        # # Unique cluster labels
        # cluster_labels = np.unique(df[f'cluster{suffix}'])
        # print(cluster_labels)
        # for i, cluster in enumerate(cluster_labels):
        #     print(cluster)
        #     # Data for current cluster inliers
        #     cluster_data = df[(df[f'cluster{suffix}'] == cluster) & (df[f'outlier{suffix}'] == 1)]
        #     plt.scatter(cluster_data['B/G'], cluster_data['G/R'],
        #                 color=color_map[cluster],  # Cycle through colors list
        #                 # color=colors[i % len(colors)],  # Cycle through colors list
        #                 label=f'Cluster {cluster}', alpha=0.5)
        #
        #     # Data for current cluster outliers
        #     outliers_data = df[(df[f'cluster{suffix}'] == cluster) & (df[f'outlier{suffix}'] == -1)]
        #     plt.scatter(outliers_data['B/G'], outliers_data['G/R'],
        #                 marker='x',  # X marker for outliers
        #                 s=50,  # Size of the marker
        #                 color=color_map[cluster],  # Cycle through colors list
        #                 # color=colors[i % len(colors)],  # Same color as the cluster
        #                 label=f'Outliers in Cluster {cluster}')  # Label only once for legend
        #
        # plt.xlabel('B/G')
        # plt.ylabel('G/R')
        # # plt.title('Color Difference: B-G vs. G-R')
        # plt.legend()
        #
        # plt.figure(figsize=(10, 6))
        #
        # # Simple contrast stretching
        # p2, p98 = np.percentile(image_data, (2, 99.5))
        # image_data_contrast_stretched = np.clip((image_data - p2) / (p98 - p2), 0, 1)
        # #
        # plt.imshow(image_data_contrast_stretched)
        # for i, cluster in enumerate(cluster_labels):
        #     # Data for current cluster inliers
        #     cluster_data = df[(df[f'cluster{suffix}'] == cluster) & (df[f'outlier{suffix}'] == 1)]
        #     plt.scatter(cluster_data['xcentroid'], cluster_data['ycentroid'],
        #                 color=color_map[cluster],  # Cycle through colors list
        #                 # color=colors[i % len(colors)],  # Cycle through colors list
        #                 alpha=0.5)
        #
        #     # Data for current cluster outliers
        #     outliers_data = df[(df[f'cluster{suffix}'] == cluster) & (df[f'outlier{suffix}'] == -1)]
        #     plt.scatter(outliers_data['xcentroid'], outliers_data['ycentroid'],
        #                 marker='x',  # X marker for outliers
        #                 s=50,  # Size of the marker
        #                 color=color_map[cluster],  # Cycle through colors list
        #                 # color=colors[i % len(colors)],  # Same color as the cluster
        #                 )
        #
        # # plt.xlabel('B-G')
        # # plt.ylabel('G-R')
        # # plt.title('Color Difference: B-G vs. G-R')
        # # plt.legend()
        # # import matplotlib.pyplot as plt
        # #
        # # Assuming 'kmeans_robust' is your fitted KMeans model from the previous steps
        # centroids = kmeans_robust.cluster_centers_
        # # Display the centroids for interpretation
        # print("Centroids of the clusters:")
        # for i, centroid in enumerate(centroids):
        #     print(f"Cluster {i + 1}: {dict(zip(features_columns, centroid))}")
        #
        # # Visualizing clusters
        # sns.pairplot(df, vars=features_columns, hue='cluster_image', palette='viridis')
        # plt.suptitle('Pair Plot of Features by Cluster', verticalalignment='top')
        #
        # plt.show()
        # return df

    def create_geojson_image(self, df):
        df = df.astype(object)
        self.log.info(f'  Processing {len(df)} rows/segments')
        feature_collection = {"type": "FeatureCollection", "features": []}
        feature_df = {}
        self.log.info(f'  > Create GeoJSON features')

        for index, row in df.iterrows():
            df.loc[index, 'segm_idx'] = index

            # Calculate rays for each corner of the bounding box
            ray_list = [
                (float(row['bbox_xmax_undist']), float(row['bbox_ymax_undist']), 1, -1),
                (float(row['bbox_xmin_undist']), float(row['bbox_ymax_undist']), 1, -1),
                (float(row['bbox_xmin_undist']), float(row['bbox_ymin_undist']), 1, -1),
                (float(row['bbox_xmax_undist']), float(row['bbox_ymin_undist']), 1, -1),
                (float(row['xcentroid_undist']), float(row['ycentroid_undist']), 1, -1)
            ]

            result = self.get_georef(False, ray_list)
            (coord_array, polybox,
             fixed_polygon, pos, _, properties) = result

            feature_df[index] = dict(coord_array=coord_array,
                                     polybox=polybox, properties=properties)

            geo_img_props = dict(SegmentLabel=row['label'],
                                 ImageID=row['id'],
                                 # ClusterID=row[f'cluster_image'],
                                 Outlier=row[f'outlier_image'])

            for k, v in geo_img_props.items():
                properties[k] = v

            # Create GeoJSON features for the current image
            feature_point, feature_polygon = create_geojson_feature(polybox[:-1],
                                                                    coord_array[-1][0],
                                                                    coord_array[-1][1],
                                                                    properties)

            feature_collection["features"].append(feature_point)
            feature_collection["features"].append(feature_polygon)
            # print(fixed_polygon, feature_point, feature_polygon)

        # print(feature_collection)
        self.log.info(f'    Save segments as geojson')
        geojson_file = f"segment_bbox_geojson_image_{self.name}.json"
        io.write_geojson_file(geojson_file, self.meta_dir,
                              feature_collection)

        return df, feature_df

        # # Setting up the figure
        # plt.figure(figsize=(10, 6))

    def create_geojson_area(self, df, feature_list):
        """"""
        self.log.info(f'  Processing {len(df)} rows/segments')
        feature_collection = {"type": "FeatureCollection", "features": []}
        for index, row in df.iterrows():
            img_id = row['id']
            segm_idx = row['segm_idx']

            coord_array = feature_list[img_id][segm_idx]['coord_array']
            polybox = feature_list[img_id][segm_idx]['polybox']
            properties = feature_list[img_id][segm_idx]['properties']

            geo_img_props = dict(SegmentLabel=row['label'],
                                 ImageID=row['id'],
                                 # ClusterID=row[f'cluster_area'],
                                 Outlier=row[f'outlier_area'])

            for k, v in geo_img_props.items():
                properties[k] = v

            # Create GeoJSON features for the current image
            feature_point, feature_polygon = create_geojson_feature(polybox[:-1],
                                                                    coord_array[-1][0],
                                                                    coord_array[-1][1],
                                                                    properties)

            feature_collection["features"].append(feature_point)
            feature_collection["features"].append(feature_polygon)

        self.log.info(f'  Save segments as geojson')
        geojson_file = f"segment_bbox_geojson_area_{self.folder_name}.json"
        io.write_geojson_file(geojson_file, self.meta_dir,
                              feature_collection)

    def get_georef(self, is_image=True, ray_list=None):
        """"""
        self.exif.get_all_info()
        props_raw = self.exif.all_info

        camera_info = {
            'sensor_width': self.ccd_width_mm,  # mm
            'sensor_height': self.ccd_height_mm,  # mm (Optional if not used in calculations)
            'image_width': self.width_px,  # pixels
            'image_height': self.height_px,  # pixels
            'Focal_Length': self.focal_len_mm,  # mm
            'lens_FOVw': self.scale_width,  # lens distortion in mm
            'lens_FOVh': self.scale_height  # lens distortion in mm
        }
        gimbal_orientation = {
            'roll': props_raw['ROLL1'][0],  # Gimbal roll in degrees
            'pitch': props_raw['PITCH1'][0],  # Gimbal pitch in degrees (negative if pointing downwards)
            # 'yaw': props_raw['YAW1'][0],  # Gimbal yaw in degrees
            'yaw': 0.,  # Gimbal yaw in degrees
        }
        flight_orientation = {
            'roll': props_raw['ROLL2'][0],  # Flight roll in degrees
            'pitch': props_raw['PITCH2'][0],  # Flight pitch in degrees
            'yaw': props_raw['YAW2'][0],  # Flight yaw in degrees (direction of flight)
        }

        drone_latitude = props_raw['LATOBS'][0]
        drone_longitude = props_raw['LONGOBS'][0]
        re_altitude = props_raw['ALTREL'][0]
        ab_altitude = props_raw['ALTOBS'][0]
        datetime_original = props_raw['DATE-STR'][0]

        gsd = (self.ccd_width_mm * re_altitude) / (self.focal_len_mm * self.width_px)

        properties = dict(
            File_Name=self.name,
            Focal_Length=self.focal_len_mm,
            Image_Width=self.width_px,
            Image_Height=self.height_px,
            Sensor_Model='L2D-20c',
            Sensor_Make='Hasselblad',
            RelativeAltitude=re_altitude,
            AbsoluteAltitude=ab_altitude,
            FlightYawDegree=props_raw['YAW2'][0],
            FlightPitchDegree=props_raw['PITCH2'][0],
            FlightRollDegree=props_raw['ROLL2'][0],
            DateTimeOriginal=datetime_original,
            GimbalPitchDegree=props_raw['PITCH1'][0],
            GimbalYawDegree=props_raw['YAW1'][0],
            GimbalRollDegree=props_raw['ROLL1'][0],
            DroneCoordinates=[drone_longitude, drone_latitude],
            Sensor_Width=self.ccd_width_mm,
            Sensor_Height=self.ccd_height_mm,
            CameraMake='Hasselblad',
            Drone_Make='DJI',
            Drone_Model='Mavic 3',
            MaxApertureValue=props_raw['APERTUR'][0],
            lens_FOVh=self.scale_height,
            lens_FOVw=self.scale_width,
            GSD=gsd,
            epsgCode=config.epsg_code
        )

        config.update_abso_altitude(ab_altitude)
        config.update_rel_altitude(re_altitude)
        config.update_epsg(4326)
        # config.update_elevation(True)
        K = camera.get_K()
        IK = np.linalg.inv(K)

        calculator = HighAccuracyFOVCalculator(
            drone_gps=(drone_latitude, drone_longitude),
            drone_altitude=re_altitude,
            camera_info=camera_info,
            gimbal_orientation=gimbal_orientation,
            flight_orientation=flight_orientation,
            datetime_original=datetime_original,
            w=self.width_px,
            h=self.height_px,
            K_inv=IK
        )

        if is_image:

            coord_array, polybox = calculator.get_geo_bbox()
            # print(coord_array, polybox)

            fixed_polygon = Polygon(coord_array)
        else:

            coord_array, polybox = calculator.get_geo_bbox(is_ray=True, ray_list=ray_list)
            # print(coord_array)
            # print(properties)
            fixed_polygon = Polygon(coord_array[:-1])

        return (coord_array, polybox, fixed_polygon,
                (drone_longitude, drone_latitude), datetime_original, properties)

    def georeference_segm_data(self, coord_array, fixed_polygon):

        self.load_detection_data()

        rectify_and_warp_to_geotiff(self.segment_map_undist.data, self.georef_map_file, fixed_polygon, coord_array)

    def georeference_image_data(self, coord_array, fixed_polygon):

        img = self.load_image_calibrated()

        img_undist = self.undistort_data(img)
        # img_undist[img_undist <= 0.] = 0.
        img_undist *= 1e6
        img_undist = np.array(img_undist, dtype=np.float32)

        rectify_and_warp_to_geotiff(img_undist, self.georef_image_file, fixed_polygon, coord_array)

    def georeference_image_ratio_br(self, coord_array, fixed_polygon, kernel_size=7):

        img = self.load_image_calibrated()
        img_cal_undist = np.where(img <= 0., np.nan, img)
        # b = img[2, :, :]
        # b[b <= 0.] = np.nan
        # b_med = apply_median_filter(b, kernel_size=kernel_size)
        #
        # r = img[0, :, :]
        # r[r <= 0.] = np.nan
        # r_med = apply_median_filter(r, kernel_size=kernel_size)
        # img_br = b_med / r_med
        img_br = img_cal_undist[2, :, :] / img_cal_undist[0, :, :]
        img_br[(img_br <= 0.) | ~(np.isfinite(img_br))] = np.nan

        # p2, p98 = np.nanpercentile(img_br, (1., 99.))
        # img_br_contrast_stretched = (img_br - p2) / (p98 - p2)

        # img_br_expanded = np.expand_dims(img_br_contrast_stretched, axis=0)
        img_br_expanded = np.expand_dims(img_br, axis=0)
        img_undist = self.undistort_data(img_br_expanded)
        img_undist = np.array(img_undist, dtype=np.float32)

        self.log.info(f'> Warp and store georeferenced image')
        rectify_and_warp_to_geotiff(img_undist, self.georef_image_br_file, fixed_polygon, coord_array)

    def georeference_data_old(self, data):

        """"""

        data = conversions.convert_kmn_to_mnk(data)

        _, f_number, _ = self.get_image_exposure()

        data_undist = detection.undistort_image(data, self.maker, self.model,
                                                self.maker, self.model,
                                                self.focal_len_mm, f_number)

        # print(data_undist.shape)
        # data_undist = data
        # new_segment_map = SegmentationImage(data_undist)
        # print(new_segment_map.labels)

        gps_data = self.get_image_gps()
        # print(gps_data, self.h_fov_deg, self.v_fov_deg)
        center_lat, center_lon, altitude_m = gps_data  # Drone GPS and altitude
        tilt, orientation = -90., 359.9  # Camera tilt and orientation
        altitude_m = altitude_m - 7
        corrected_lat, corrected_lon = conversions.calculate_corrected_position(center_lat,
                                                                                center_lon,
                                                                                altitude_m,
                                                                                tilt,
                                                                                orientation)
        print(f"Corrected Position: Latitude {corrected_lat}, Longitude {corrected_lon}")
        # corrected_lat, corrected_lon, shifts = conversions.calculate_corrected_location_quaternion(center_lat,
        #                                                                                            center_lon,
        #                                                                                            altitude_m,
        #                                                                                            orientation,
        #                                                                                            tilt, 0)
        # print(f"Corrected Position: Latitude {corrected_lat}, Longitude {corrected_lon}")
        # print(shifts)
        # corrected_lat, corrected_lon, shifts = conversions.calculate_corrected_location_simple(center_lat,
        #                                                                                        center_lon,
        #                                                                                        altitude_m,
        #                                                                                        orientation,
        #                                                                                        tilt, 0)
        # print(f"Corrected Position: Latitude {corrected_lat}, Longitude {corrected_lon}")
        # print(shifts)
        # print(self.ypr_to_quat(orientation, tilt, 0))
        # quit()
        gps_data = corrected_lat, corrected_lon, altitude_m
        # geotransform = conversions.calculate_ground_coverage(gps_data[0], gps_data[1], gps_data[2],
        #                                                      data_undist.shape[1], data_undist.shape[0],
        #                                                      self.h_fov_deg, self.v_fov_deg)
        # print(geotransform)
        #
        # rotation_x, rotation_y = conversions.calculate_rotation_parameters(yaw_deg,
        #                                                                    data_undist.shape[1],
        #                                                                    data_undist.shape[0])
        #
        # print(f"Rotation parameters: {rotation_x}, {rotation_y}")
        # geotransform[2] = rotation_x
        # geotransform[4] = rotation_y
        # print(geotransform)
        # quit()
        #
        # # Calculate the geographic coordinates of the image corners
        # corners = conversions.compute_corners_geodetic(gps_data, -90, self.h_fov_deg, self.v_fov_deg,
        #                                                0.)
        # print(corners)
        #
        data_type = 'image'  # or  'segmentation_map'
        # data_type = 'segmentation_map'  # or  'segmentation_map'
        # tilt = -90  # Camera tilt
        # orientation = 0  # Camera orientation
        # corners = conversions.compute_corners(gps_data, tilt, self.h_fov_deg, self.v_fov_deg,
        #                                       orientation)
        # print(corners)
        # Georeference the data
        conversions.georeference_data(data_undist, data_type,
                                      gps_data, tilt,
                                      self.h_fov_deg, self.v_fov_deg, orientation, self.georef_map_file)

        # Example usage
        georeferenced_image = str(self.georef_map_file)
        output_png_path = '/home/cadam/output_image.png'

        conversions.convert_geotiff_to_png(georeferenced_image, output_png_path)
        # conversions.plot_georeferenced_image(str(self.georef_map_file), '/home/cadam/final_map.png')
        conversions.overlay_on_osm(output_png_path)

        import rasterio

        # Open the GeoTIFF file
        with rasterio.open(georeferenced_image) as dataset:
            # Read the first band (assuming that the GeoTIFF is single-band)
            # If your dataset has multiple bands,
            # you may need to read them separately or perform some kind of compositing
            band1 = dataset.read(1)

            # Flip the band data along the y-axis
            # band1 = np.fliplr(band1)

            # Get the bounds of the image
            bounds = dataset.bounds

            # Configure the extent of the plot. This is necessary for proper geo-referenced plotting
            extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

            # Plot the band data
            plt.imshow(band1, extent=extent, cmap='gray', aspect='auto')  # Use an appropriate colormap for your data
            plt.title('GeoTIFF Plot')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')

            # Optionally, you can add color bar if needed
            plt.colorbar(label='Data values')

            # Show the plot
            # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True, sharey=True)

        ax.imshow(data_undist)
        ax.set_title('Data')
        # ax[1].imshow(data_undist, origin='lower')
        # ax[1].set_title('Data undistorted')
        plt.tight_layout()
        plt.show()
        return data_undist

    def test_srtm(self, srtm):

        from osgeo import gdal, osr
        import navpy

        self.load_detection_data()

        ref, ypr, _ = self.get_aircraft_pose()
        # print(ref, ypr)
        from pyproj import Transformer
        t = Transformer.from_crs("epsg:4326", "epsg:4326+3855", always_xy=True)
        x = t.transform(ref[1], ref[0], 0)
        # print(x)
        latitude = ref[0]
        longitude = ref[1]
        import rasterio
        from pathlib import Path

        # define coordinates
        lat = ref[0]
        lon = ref[1]
        z = 126

        # open .gtx file using rasterio
        path = Path("../../data/egm08_25.gtx")
        src = rasterio.open(path)

        # compute undulation
        undulation = next(src.sample([(lon, lat)]))[0]
        # compute height over goid, i.e. mean sea level
        orthometric_height = z - undulation
        print('undulation:', undulation, ', orthometric_height:', orthometric_height)

        corrected_lat, corrected_lon = conversions.calculate_corrected_position(lat,
                                                                                lon,
                                                                                ref[2],
                                                                                ypr[1],
                                                                                ypr[0])
        print(f"Corrected Position: Latitude {corrected_lat}, Longitude {corrected_lon}")
        print(ref)
        # return
        # Calculate the geographic coordinates of the image corners
        corners = conversions.compute_corners([ref[0], ref[1], ref[2]], ypr[1],
                                              self.h_fov_deg, self.v_fov_deg, ypr[0])
        print(corners)
        import math
        fov_x = np.deg2rad(self.h_fov_deg)
        fov_y = np.deg2rad(self.v_fov_deg)
        # Calculate ground coverage in meters
        coverage_x = 2 * ref[2] * math.tan(fov_x / 2)
        coverage_y = 2 * ref[2] * math.tan(fov_y / 2)
        print(coverage_y, coverage_x)
        print(coverage_y / 2, coverage_x / 2)
        # return
        K = camera.get_K()
        IK = np.linalg.inv(K)
        # print(K, IK)
        # print(self.segment_map)
        gdal_data_type = gdal.GDT_Int32
        ned = np.array([0, 0, -orthometric_height])

        ned = np.array([0, 0, -undulation])
        ned = np.array([0, 0, -ref[2]])

        corner_list = np.array([
            [self.width_px / 2, self.height_px / 2],
            [self.width_px / 2, 0],
            [0, self.height_px / 2],
            [0, 0],
            [self.width_px, 0], [self.width_px, self.height_px],
            [0, self.height_px]
        ])

        proj_list = project.projectVectors(IK, self.get_body2ned(),
                                           self.get_cam2body(),
                                           corner_list)
        print('proj_list:', proj_list)
        pts_ned = project.intersectVectorsWithGroundPlane(ned, 0., proj_list)
        print(self.name, "pts_3d (ned):\n", pts_ned)
        corners_lonlat = []
        for ned in pts_ned:
            print(ned)
            print(srtm.get_ground(ned))
            lla = navpy.ned2lla([ned], ref[0], ref[1], ref[2])
            print(lla)
            corners_lonlat.append([lla[0], lla[1], srtm.get_ground(ned)])

        pts_ned = srtm.interpolate_vectors(ned, proj_list)
        print(self.name, "pts_3d (ned):\n", pts_ned)

        # Create GCPs using the corner coordinates and pixel coordinates
        gcps = [gdal.GCP(corner[1], corner[0], corner[2], pixel[0], pixel[1]) for corner, pixel in
                zip(corners_lonlat, corner_list)]
        for gcp in gcps:
            print(f"Pixel: ({gcp.GCPPixel}, {gcp.GCPLine}) => Geo: ({gcp.GCPX}, {gcp.GCPY}, {gcp.GCPZ})")
        print(ref, ypr, K, IK, corner_list)
        R_inv, t = get_rot_trans_mat(ref, ypr[0], ypr[1], ypr[2])
        # Calculate world coordinates
        # Calculate world coordinates
        world_coordinates = image_to_world(corner_list, R_inv, IK, np.array(ref))
        print(world_coordinates)
        # corners_lonlat = []
        # for ned in pts_ned:
        #     print(ned)
        #     lla = navpy.ned2lla([ned], ref[0], ref[1], ref[2])
        #     print(lla)
        #     corners_lonlat.append([lla[0], lla[1], lla[2]])
        #
        # # Create GCPs using the corner coordinates and pixel coordinates
        # gcps = [gdal.GCP(corner[1], corner[0], corner[2], pixel[0], pixel[1]) for corner, pixel in
        #         zip(corners_lonlat, corner_list)]
        # for gcp in gcps:
        #     print(f"Pixel: ({gcp.GCPPixel}, {gcp.GCPLine}) => Geo: ({gcp.GCPX}, {gcp.GCPY}, {gcp.GCPZ})")
        #
        # print(ref)

        # Create a new GDAL in-memory dataset
        mem_driver = gdal.GetDriverByName('MEM')
        mem_raster = mem_driver.Create('', self.width_px, self.height_px, 1, gdal_data_type)
        mem_raster.GetRasterBand(1).WriteArray(self.segment_map_undist.data)

        # Create a spatial reference object for WGS84
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        # # Apply the GCPs to the in-memory dataset
        mem_raster.SetProjection(srs.ExportToWkt())
        mem_raster.SetGCPs(gcps, srs.ExportToWkt())

        # Georeference the in-memory dataset using the GCPs and output to a file
        gdal.Warp(str(self.georef_map_file), mem_raster, format='GTiff',
                  dstSRS='EPSG:3857', width=self.width_px, height=self.height_px)

        # Cleanup
        mem_raster = None

        # return corners_lonlat

    def create_tiff(self):
        """"""
        drone_pos, ypr, _ = self.get_aircraft_pose()
        # print(drone_pos, ypr)

        self.load_image_calibrated()
        image_RGB_cal = self.image_RGB_cal
        image_RGB_cal = np.clip(image_RGB_cal, a_min=0, a_max=None)
        image_RGB_cal = conversions.convert_kmn_to_mnk(image_RGB_cal)
        # print(self.image_RGB_cal.shape)

        # image_XYZ = self.load_XYZ_image()
        # image_sRGB = self.convert_XYZ_to_sRGB(image_XYZ)

        _, f_number, _ = self.get_image_exposure()

        data_undist = detection.undistort_image(image_RGB_cal, self.maker, self.model,
                                                self.maker, self.model,
                                                self.focal_len_mm, f_number)
        import subprocess
        import tifffile

        # Save the array as a TIFF file
        tifffile.imwrite(str(self.georef_image_file),
                         np.asarray(data_undist, dtype=np.float32))

        subprocess.run(['exiftool', '-overwrite_original', '-TagsFromFile',
                        self.image_file, self.georef_image_file])

        # # self.load_detection_data()
        # fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True, sharey=True)
        #
        # ax.imshow(image_RGB_cal[1, :, :])
        # ax.set_title('Data')
        # # ax[1].imshow(data_undist, origin='lower')
        # # ax[1].set_title('Data undistorted')
        # plt.tight_layout()
        # plt.show()


def rectify_and_warp_to_geotiff(jpeg_img_array, dst_utf8_path, fixed_polygon, coordinate_array):
    """
    Warps and rectifies a JPEG image array to a GeoTIFF format based on a fixed polygon and coordinate array.

    Parameters:
    - jpeg_img_array: The NumPy array of the JPEG image.
    - dst_utf8_path: Destination path for the output GeoTIFF image.
    - fixed_polygon: The shapely Polygon object defining the target area.
    - coordinate_array: Array of coordinates used for warping the image.
    """
    # Convert the Polygon to WKT format
    polygon_wkt = str(fixed_polygon)

    # Warp the image to the polygon using the coordinate array
    # Turn off GDAL warnings
    os.environ['CPL_LOG'] = '/dev/null'
    os.environ['GDAL_DATA'] = os.getenv('GDAL_DATA',
                                        '/var/lib/flatpak/app/org.qgis.qgis/x86_64'
                                        '/stable/0371292c5bc48ecc058e05ba7b100388bc9d3bb7622af8fb4f169ba81b4e7613/'
                                        'files/share/gdal')
    gdal.DontUseExceptions()
    gdal.SetConfigOption('CPL_DEBUG', 'OFF')

    try:
        georef_image_array = warp_image_to_polygon(jpeg_img_array, fixed_polygon, coordinate_array)
        dsArray = array2ds(georef_image_array, polygon_wkt)
    except Exception as e:
        print(f"Error during warping or dataset creation: {e}")

    # Warp the GDAL dataset to the destination path
    try:
        warp_ds(str(dst_utf8_path), dsArray)
    except Exception as e:
        print(f"Error writing GeoTIFF: {e}")


def get_rot_trans_mat(drone_position, yaw, pitch, roll):
    # Convert angles from degrees to radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)

    # Rotation matrices
    R_z = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    R_y = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])

    # Combined rotation matrix and its inverse
    R = R_z @ R_y @ R_x
    R_inv = np.linalg.inv(R)

    # Translation vector
    t = np.array(drone_position)

    return R_inv, t


# Function to transform image coordinates to world coordinates
def image_to_world(points, R_inv, K_inv, drone_position):
    # Convert to homogeneous coordinates
    points_homog = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply the inverse camera matrix to get normalized camera coordinates
    normalized_camera_coords = K_inv @ points_homog.T

    # Assuming the camera is facing downward when pitch is -90 degrees,
    # calculate the ground intersection assuming a flat Earth and using altitude:
    # Since the drone is looking directly down, the Z component in the normalized camera coordinates
    # (which is always 1 in the homogeneous representation) will intersect the ground at the altitude.
    scale_factors = drone_position[2] / (R_inv[2, :] @ normalized_camera_coords)
    world_coords = (R_inv @ (normalized_camera_coords * scale_factors)) + drone_position.reshape(-1, 1)

    return world_coords.T[:, :3]


def reverse_project_image_points(image_points, R_inv, t, K_inv, altitude):
    # Convert image points to homogeneous coordinates
    image_points_homog = np.hstack([image_points, np.ones((image_points.shape[0], 1))])

    # Apply inverse camera calibration
    world_points = K_inv @ image_points_homog.T

    # Assuming points are on the ground (z = 0 in world coordinates)
    # Scale factor to adjust for altitude and the transformation
    scale_factor = altitude / (R_inv[2, :] @ world_points)

    # Calculate world coordinates
    world_points = (R_inv @ (world_points * scale_factor)) + t.reshape(-1, 1)

    return world_points.T[:, :3]  # Return x, y, z coordinates


def calculate_photometric_vars(row, image, segm, idx_list, rgb_mask):
    """"""

    r_mag_inst = -2.5 * np.log10(row['R_flux'])
    g_mag_inst = -2.5 * np.log10(row['G_flux'])
    b_mag_inst = -2.5 * np.log10(row['B_flux'])

    r_mag_inst_surf = -2.5 * np.log10(row['R_flux'] / row['area'])
    g_mag_inst_surf = -2.5 * np.log10(row['G_flux'] / row['area'])
    b_mag_inst_surf = -2.5 * np.log10(row['B_flux'] / row['area'])

    bg_ratio = row['B_flux'] / row['G_flux']
    gr_ratio = row['G_flux'] / row['R_flux']
    rg_ratio = row['R_flux'] / row['G_flux']
    br_ratio = row['B_flux'] / row['R_flux']

    b_over_gr = row['B_flux'] / (row['G_flux'] + row['R_flux'])

    rg_mag_diff = -2.5 * np.log10(rg_ratio)
    gr_mag_diff = -2.5 * np.log10(gr_ratio)
    bg_mag_diff = -2.5 * np.log10(bg_ratio)
    br_mag_diff = -2.5 * np.log10(br_ratio)

    label_id = row['label']
    segmobj = segm.segments[segm.get_index(label_id)]
    mask = segmobj.data_ma
    sliced = (slice(None),) + segmobj.slices
    if rgb_mask is not None:
        # rgb_mask_sliced = (slice(None),) + rgb_mask
        image[:, rgb_mask] = np.nan

    segm_image = image[sliced]
    segm_image = segm_image * mask
    segm_image = np.where(mask, segm_image, np.nan)
    segm_image = np.where(segm_image != 0., segm_image, np.nan)
    segm_mean = np.nanmean(segm_image, axis=(1, 2))
    # print(segm_mean)
    # plt.figure()
    # plt.imshow(segm_image[0, :, :])
    # plt.figure()
    # plt.hist(segm_image[0, :, :].flatten(), bins=30, density=False, color='r')
    # plt.hist(segm_image[1, :, :].flatten(), bins=30, density=False, color='g')
    # plt.hist(segm_image[2, :, :].flatten(), bins=30, density=False, color='b')
    # plt.show()

    ratio_R2G = np.divide(segm_image[0, :, :], segm_image[1, :, :])
    ratio_B2G = np.divide(segm_image[2, :, :], segm_image[1, :, :])
    ratio_B2R = np.divide(segm_image[2, :, :], segm_image[0, :, :])
    ratio_G2R = np.divide(segm_image[1, :, :], segm_image[0, :, :])

    ratio_B2G_plus_R = np.divide(segm_image[2, :, :], (segm_image[0, :, :] + segm_image[1, :, :]))

    ratio_R2G_avg = np.nanmean(ratio_R2G, axis=None)
    ratio_B2G_avg = np.nanmean(ratio_B2G, axis=None)
    ratio_B2R_avg = np.nanmean(ratio_B2R, axis=None)
    ratio_G2R_avg = np.nanmean(ratio_G2R, axis=None)
    ratio_B2G_plus_R_avg = np.nanmean(ratio_B2G_plus_R, axis=None)

    mag_diff_RG = -2.5 * np.log10(ratio_R2G_avg)
    mag_diff_BG = -2.5 * np.log10(ratio_B2G_avg)
    mag_diff_BR = -2.5 * np.log10(ratio_B2R_avg)
    mag_diff_GR = -2.5 * np.log10(ratio_G2R_avg)

    r_mag_inst_avg = -2.5 * np.log10(np.nanmean(segm_image[0, :, :], axis=None))
    g_mag_inst_avg = -2.5 * np.log10(np.nanmean(segm_image[1, :, :], axis=None))
    b_mag_inst_avg = -2.5 * np.log10(np.nanmean(segm_image[2, :, :], axis=None))

    return pd.Series(data=[bg_ratio, gr_ratio, rg_ratio, br_ratio,
                           bg_mag_diff, gr_mag_diff, rg_mag_diff, br_mag_diff,
                           segm_mean[0], segm_mean[1], segm_mean[2],
                           ratio_B2G_avg, ratio_G2R_avg, ratio_R2G_avg, ratio_B2R_avg,
                           mag_diff_BG, mag_diff_GR, mag_diff_RG, mag_diff_BR,
                           r_mag_inst, g_mag_inst, b_mag_inst,
                           r_mag_inst_avg, g_mag_inst_avg, b_mag_inst_avg,
                           r_mag_inst_surf, g_mag_inst_surf, b_mag_inst_surf, b_over_gr, ratio_B2G_plus_R_avg],
                     index=idx_list)


def calculate_modified_z_scores(points):
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[:, None]

    median = np.median(points, axis=0)
    diff = np.sqrt(np.sum((points - median) ** 2, axis=-1))
    med_abs_deviation = np.median(diff)

    # Calculate the modified Z-score
    modified_z_scores = 0.6745 * diff / med_abs_deviation
    return modified_z_scores


def add_outlier_flags(df, features, suffix='_local', quantile=90):
    """
    Add outlier flags to the DataFrame for given features based on modified Z-scores.

    Args:
    df (DataFrame): The DataFrame containing the data.
    features (list): A list of column names to check for outliers.
    quantile (int): The quantile used to set the threshold for outliers.

    Returns:
    DataFrame: The DataFrame with added outlier flag columns.
    """
    for feature in features:
        # Calculate Z-scores for the feature
        z_scores = calculate_modified_z_scores(df[feature])

        # Determine the threshold from the quantile
        threshold = np.percentile(z_scores, quantile)

        # Add outlier flags to the DataFrame
        outlier_column_name = f'outlier_{feature}{suffix}'
        df[outlier_column_name] = z_scores > threshold

    return df


def get_outliers_inliers(df, suffix='_local'):
    """
    Extracts outliers and inliers from the DataFrame based on columns that end with a specified suffix.

    Args:
    df (DataFrame): The DataFrame containing the data and outlier flags.
    suffix (str): The suffix used to identify outlier flag columns.

    Returns:
    tuple: Two DataFrames, the first with all outliers and the second with all inliers.
    """
    # Identify all columns that are used for outlier flags
    outlier_columns = [col for col in df.columns if col.endswith(suffix)]

    # Build a condition for selecting outliers and inliers
    if not outlier_columns:
        raise ValueError("No outlier flag columns found in the DataFrame.")

    # Creating a single combined outlier condition
    outlier_condition = df[outlier_columns[0]]
    for col in outlier_columns[1:]:
        outlier_condition |= df[col]  # OR operation for any outlier

    # Inlier condition is simply the negation of the outlier condition
    inlier_condition = ~outlier_condition

    # Selecting data based on conditions
    outliers_data = df[outlier_condition]
    inliers_data = df[inlier_condition]

    return outliers_data, inliers_data


from scipy.ndimage import median_filter


def apply_median_filter(image, kernel_size=3):
    # Apply the median filter using the specified kernel size
    filtered_image = median_filter(image, size=kernel_size)
    return filtered_image
