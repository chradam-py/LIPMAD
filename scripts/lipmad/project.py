#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fnmatch
import logging
import os
import re
import pandas as pd
from pathlib import Path
from transformations import unit_vector

import numpy as np
from props import getNode
import props_json

from . import base_conf
from . import camera
from . import general


class ProjectMgr(object):
    """Initialize and validate given input."""

    def __init__(self, input_arg,
                 create: bool = False,
                 log: logging.Logger = base_conf.log,
                 log_level: int = 20):
        """ Constructor with default values """

        # set log level
        log.setLevel(log_level)

        # check input arguments
        if not isinstance(input_arg, str):
            raise ValueError("Input argument must be a str")

        self.analysis_dir = None
        self.input_arg = Path(input_arg).expanduser().resolve()
        self.project_dir = None
        self.images_dir = None
        self.image_file_list = []
        self.image_file_df = pd.DataFrame()
        self.image_files_grouped = {}
        self.log = log

        self.dir_node = getNode(path='/config/directories', create=True)

        if self.load(create=create):
            self.log.info("> Project loaded")
        else:
            self.log.error("> Project could not be loaded.")
            quit()

    @staticmethod
    def set_defaults() -> None:
        camera.set_defaults()  # camera defaults

    def detect_camera(self) -> tuple:
        """Detect camera parameter"""
        from .exif import Exif

        cam, make, model, lens_model, image_file, focal_length = (None for _ in range(6))

        for image_file in self.image_file_list:
            exif = Exif(image_file[0])
            exif.get_camera_info()
            cam, make, model, lens_model, focal_length = exif.camera_info
            break

        return cam, make, model, lens_model, focal_length, image_file[0]

    def validate_input(self, create_if_needed) -> bool:
        """ Validate the input arguments. """

        input_arg = self.input_arg

        if create_if_needed:
            project_dir = input_arg
            analysis_dir = input_arg / "ImageAnalysis"
            if not analysis_dir.exists():
                analysis_dir.mkdir(exist_ok=True)
        else:
            input_path = general.get_last_valid_folder(input_arg)

            # Loop through the input_path's parents until a metadata JSON file is found
            for parent in [input_path, *input_path.parents][:3]:
                analysis_dir = parent / "ImageAnalysis"
                if not analysis_dir.exists():
                    continue
                else:
                    project_dir = parent
                    self.images_dir = input_path
                    break
            else:
                self.log.error("Analysis directory not found.")
                return False

        self.project_dir = project_dir
        self.analysis_dir = analysis_dir

        return True

    def save(self):
        """ Save project configuration. """

        self.log.info("> Save project configuration")

        if not self.analysis_dir.exists():
            self.log.error(f"  Folder doesn't exist: {self.analysis_dir}")
            return

        project_file = self.analysis_dir / "lipmad-config.json"
        config_node = getNode("/config", True)
        props_json.save(project_file, config_node)

    def load(self, create=False):
        """ Load/create project configuration and create folder if not existent. """

        if not self.validate_input(create_if_needed=create):
            return False

        self.log.info("> Load project configuration")

        # load project configuration from analysis directory
        result = False
        project_file = self.analysis_dir / "lipmad-config.json"
        config_node = getNode(path="/config", create=True)
        if project_file.exists():
            self.log.info(f"  Found config: {project_file}")
            if props_json.load(project_file, config_node):
                result = True
            else:
                self.log.warning(f"  Unable to load: {project_file}")
        else:
            self.log.warning(f"  Configuration doesn't exist: {project_file}")

        if not result and create:
            self.log.info("  Continuing with an empty configuration")
            self.set_defaults()
        elif not result:
            self.log.error("  Configuration load failed, aborting...")
            quit()

        self.dir_node.setString('project_dir', self.project_dir)
        self.dir_node.setString('analysis_dir', self.analysis_dir)

        return True

    def process_input(self, input_path: str, image_fmt=None, prefix=None):

        self.log.info("> Check input")

        # convert into a Path object
        input_path = Path(input_path)
        prefix = '' if prefix is None else prefix

        image_fmt, extensions = general.check_image_format(image_fmt)

        regex_ext = general.compile_regex(prefix, extensions)
        file_to_match = re.search(regex_ext, str(input_path))
        if input_path.is_dir():
            self._handle_directory(input_path, regex_ext)
        elif input_path.is_file() and file_to_match:
            self.image_file_list.append(str(input_path))
        else:
            base_dir = general.get_last_valid_folder(input_path)
            # regex pattern to match directories
            pattern = re.compile(f'^{input_path.name}.*')

            for d in base_dir.iterdir():
                if pattern.match(d.name) and d.is_dir():
                    self._handle_directory(d, regex_ext)
                if pattern.match(d.name) and d.is_file():
                    regex_ext = general.compile_regex(input_path.name, extensions)
                    self._handle_directory(base_dir, regex_ext)
                    break

        if not self.image_file_list:
            self.log.error("  NO valid file(s) found. Please check the input.")
            quit()
        else:
            self.log.info(f"  Found valid file(s).")

        path_arr = general.split_file_names(self.image_file_list)

        columns = ['input', 'file_name', 'file_ext',
                   'file_loc', 'file_loc_parent', 'file_loc_parent_name']
        obs_df = pd.DataFrame(data=path_arr, columns=columns)

        # group data
        obs_df_grouped = obs_df.groupby('file_loc_parent')

        self.image_file_list = path_arr
        self.image_file_df = obs_df
        self.image_files_grouped = obs_df_grouped

    def save_images_info(self, images_dir_base):
        """"""
        images_dir = self.analysis_dir / images_dir_base
        meta_dir = images_dir / 'meta'
        images_node = getNode("/images", True)
        images_sub_node = images_node.getChild(images_dir_base, create=True)
        for name in images_sub_node.getChildren():
            image_node = images_sub_node.getChild(name, True)
            image_path = meta_dir / f"{name}.json"
            props_json.save(image_path, image_node)

    def load_images_info(self):
        """"""
        for group_name, group_data in self.image_files_grouped:
            first_row = group_data.iloc[0]  # Get the first row of each group
            images_dir_base = first_row['file_loc_parent_name']
            images_dir = self.analysis_dir / images_dir_base
            meta_dir = images_dir / 'meta'

            images_node = getNode("/images", True)
            images_sub_node = images_node.getChild(images_dir_base, create=True)

            for file in os.listdir(meta_dir):
                if fnmatch.fnmatch(file, '*.json') and 'geo' not in file:
                    name, _ = os.path.splitext(file)
                    image_node = images_sub_node.getChild(name, True)
                    image_path = meta_dir / f"{name}.json"
                    props_json.load(image_path, image_node)

    def _handle_directory(self, directory, regex):
        """Process files in a directory."""
        matching_files = general.handle_directory(directory, regex)
        self.image_file_list.extend(matching_files)

    @staticmethod
    def compute_ned_reference_lla(image_loc_name):
        # requires images to have their location computed/loaded
        lon_sum = 0.0
        lat_sum = 0.0
        count = 0
        images_node = getNode("/images", True)
        images_sub_node = images_node.getChild(image_loc_name, True)
        for name in images_sub_node.getChildren():
            image_node = images_sub_node.getChild(name, True)
            pose_node = image_node.getChild('aircraft_pose', True)
            if pose_node.hasChild('lon_deg') and pose_node.hasChild('lat_deg'):
                lon_sum += pose_node.getFloat('lon_deg')
                lat_sum += pose_node.getFloat('lat_deg')
                count += 1

        ned_node = getNode('/config/ned_reference', True)
        ned_sub_node = ned_node.getChild(image_loc_name, True)
        ned_sub_node.setFloat('lat_deg', lat_sum / count)
        ned_sub_node.setFloat('lon_deg', lon_sum / count)
        ned_sub_node.setFloat('alt_m', 0.0)

    @staticmethod
    def get_ned_reference_lla(image_loc_name):

        ned_node = getNode('/config/ned_reference', True)
        ned_sub_node = ned_node.getChild(image_loc_name, True)

        ref = [ned_sub_node.getFloat('lat_deg'),
               ned_sub_node.getFloat('lon_deg'),
               ned_sub_node.getFloat('alt_m')]

        return ref


def projectVectors(IK, body2ned, cam2body, uv_list):
    proj_list = []
    for uv in uv_list:
        # print("uv:", uv)
        uvh = np.array([uv[0], uv[1], 1.0])
        proj = body2ned.dot(cam2body).dot(IK).dot(uvh)
        # proj = cam2body.dot(IK).dot(uvh)
        proj_norm = unit_vector(proj)
        # print("cam vec=", proj_norm)

        proj_list.append(proj_norm)

    return proj_list


def intersectVectorsWithGroundPlane(pose_ned, ground_m, v_list):
    pt_list = []
    for v in v_list:
        # solve projection
        p = pose_ned
        if v[2] > 0.0:
            d_proj = -(pose_ned[2] + ground_m)
            factor = d_proj / v[2]
            n_proj = v[0] * factor
            e_proj = v[1] * factor
            p = [pose_ned[0] + n_proj, pose_ned[1] + e_proj, pose_ned[2] + d_proj]
        pt_list.append(p)
    return pt_list
