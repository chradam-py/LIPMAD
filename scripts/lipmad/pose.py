#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import navpy

import numpy as np

from props import getNode
import props_json
from transformations import euler_from_quaternion, quaternion_multiply
from tqdm import tqdm

from . import base_conf
from . import camera
from .image import Image

log = base_conf.log


# for each image, compute the estimated camera pose in NED space from
# the aircraft body pose and the relative camera orientation
def compute_camera_poses(image_list, image_loc_name):
    """"""

    log.info("> Setting camera poses (offset from aircraft pose.)")

    ref_node = getNode("/config/ned_reference", True)
    ref_sub_node = ref_node.getChild(image_loc_name, True)
    ref_lat = ref_sub_node.getFloat("lat_deg")
    ref_lon = ref_sub_node.getFloat("lon_deg")
    ref_alt = ref_sub_node.getFloat("alt_m")
    body2cam = camera.get_body2cam()

    # load the image object
    image = Image()
    for file in tqdm(image_list):
        file_loc = file[0]
        name = file[1]
        name_loc = file[5]

        # initiate the image object
        image.init_image(file_loc, name, name_loc)

        ac_pose_node = image.node.getChild("aircraft_pose", True)
        aircraft_lat = ac_pose_node.getFloat("lat_deg")
        aircraft_lon = ac_pose_node.getFloat("lon_deg")
        aircraft_alt = ac_pose_node.getFloat("alt_m")
        ned2body = []
        for i in range(4):
            ned2body.append(ac_pose_node.getFloatEnum("quat", i))

        ned2cam = quaternion_multiply(ned2body, body2cam)
        (yaw_rad, pitch_rad, roll_rad) = euler_from_quaternion(ned2cam, "szyx")
        ned = navpy.lla2ned(aircraft_lat, aircraft_lon, aircraft_alt,
                            ref_lat, ref_lon, ref_alt)

        image.set_camera_pose(ned, np.rad2deg(yaw_rad), np.rad2deg(pitch_rad), np.rad2deg(roll_rad))


def set_poses(proj, image_list, images_dir_base, max_angle=25.0):
    """"""

    images_dir = proj.analysis_dir / images_dir_base
    meta_dir = images_dir / 'meta'
    images_node = getNode("/images", True)
    pix4d_file = images_dir / 'pix4d.csv'

    log.info("> Setting aircraft poses")

    # load the image object
    image = Image()
    images = []
    for file in tqdm(image_list):
        file_loc = file[0]
        name = file[1]
        name_loc = file[5]

        # initiate the image object
        image.init_image(file_loc, name, name_loc)

        # load exif data
        exif = image.exif

        image_pose = get_image_pose(exif)
        lat_deg, lon_deg, alt_m, roll_deg, pitch_deg, yaw_deg = image_pose

        camera_make = camera.camera_node.getString("make")
        # if camera_make in ["DJI", "Hasselblad"]:
        #     if pitch_deg > -45:
        #         # log.error(f"  Gimbal not looking down: {name}, roll: {roll_deg}, pitch: {pitch_deg}")
        #         continue
        # elif abs(roll_deg) > max_angle or abs(pitch_deg) > max_angle:
        #     # log.error(f"  Extreme attitude:  {name}, roll: {roll_deg}, pitch: {pitch_deg}")
        #     continue

        #
        row = {
            "File Name": name,
            "Lat (decimal degrees)": f"{lat_deg:.10f}",
            "Lon (decimal degrees)": f"{lon_deg:.10f}",
            "Alt (meters MSL)": f"{alt_m:.2f}",
            "Roll (decimal degrees)": f"{roll_deg:.2f}",
            "Pitch (decimal degrees)": f"{pitch_deg:.2f}",
            "Yaw (decimal degrees)": f"{yaw_deg:.2f}"
        }
        images.append(row)

        #
        image.set_aircraft_pose(lat_deg, lon_deg, alt_m,
                                yaw_deg, pitch_deg, roll_deg)

        #
        exif.get_exposure_info()
        image_exposure = exif.exposure_info
        image.set_image_exposure(*image_exposure)

        image_loc_node = images_node.getChild(name_loc, True)
        image_node = image_loc_node.getChild(name, True)
        image_path = meta_dir / f"{name}.json"
        props_json.save(image_path, image_node)

        # log.info(f"  Pose: {name}, yaw={yaw_deg:.1f} pitch={pitch_deg:.1f} roll={roll_deg:.1f}")

    #
    log.info(f"> Save poses to pix4d file: {pix4d_file}, "
             f"images: {len(image_list)}")

    # Assuming 'images' is a list of tuples/lists as per your original code
    fieldnames = ["File Name", "Lat (decimal degrees)", "Lon (decimal degrees)",
                  "Alt (meters MSL)", "Roll (decimal degrees)",
                  "Pitch (decimal degrees)", "Yaw (decimal degrees)"]

    # Write to CSV
    with open(pix4d_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(images)

    del image, images


def get_image_pose(exif):
    """"""

    exif.get_pose_info()

    pose = exif.pose_info

    (lon_deg, lat_deg,
     alt_m_abs, alt_m_rel,
     yaw_deg, pitch_deg, roll_deg,
     yaw_deg_drone, pitch_deg_drone, roll_deg_drone,
     x_speed, y_speed, z_speed) = pose

    altitude = alt_m_abs
    # altitude = alt_m_rel
    roll = roll_deg_drone or 0
    pitch = -90 if (camera.camera_node.getString("make") == "DJI" and
                    camera.camera_node.getString("model") in ["FC7303"]) else pitch_deg or 0
    yaw = yaw_deg_drone or 0
    while yaw < 0:
        yaw += 360
    return [lat_deg, lon_deg, altitude, roll, pitch, yaw]


