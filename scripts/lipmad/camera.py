#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from props import getNode
from transformations import quaternion_from_euler

camera_node = getNode('/config/camera', True)


def set_defaults():
    """Set default camera values"""

    # meta data
    camera_node.setString('make', 'unknown')
    camera_node.setString('model', 'unknown')
    camera_node.setString('lens_model', 'unknown')

    # camera lens parameters
    camera_node.setFloat('ccd_width_mm', 17.3)
    camera_node.setFloat('ccd_height_mm', 13.0)
    camera_node.setFloat('focal_len_mm', 12.9)
    camera_node.setFloat('scale_width', 1.)
    camera_node.setFloat('scale_height', 1.)

    h_fov, v_fov, diag_fov = calc_fov(12.9, 17.3, 13.0, 1, 1)
    camera_node.setFloat('h_fov_deg', round(np.rad2deg(h_fov), 2))
    camera_node.setFloat('v_fov_deg', round(np.rad2deg(v_fov), 2))
    camera_node.setFloat('diag_fov_deg', round(np.rad2deg(diag_fov), 2))

    # camera calibration parameters
    camera_node.setLen('K', 9, init_val=0.0)
    camera_node.setLen('dist_coeffs', 5, init_val=0.0)

    # image parameter
    camera_node.setFloat('pixel_size', 0)
    camera_node.setInt('height_px_full', 0)
    camera_node.setInt('width_px_full', 0)
    camera_node.setInt('height_px', 0)
    camera_node.setInt('width_px', 0)

    camera_node.setLen('bayer_pattern', 4)
    camera_node.setLen('bias', 4)
    camera_node.setInt('bit_depth', 16)
    camera_node.setInt('n_colours', 3)
    camera_node.setString('colour_description', 'RGBG')
    camera_node.setString('raw_extension', 'DNG')


def set_meta(make, model, lens_model):
    camera_node.setString('make', make)
    camera_node.setString('model', model)
    camera_node.setString('lens_model', lens_model)


def get_lens_params():
    return (camera_node.getFloat('ccd_width_mm'),
            camera_node.getFloat('ccd_height_mm'),
            camera_node.getFloat('focal_len_mm'),
            camera_node.getFloat('h_fov_deg'),
            camera_node.getFloat('v_fov_deg'),
            camera_node.getFloat('diag_fov_deg'),
            camera_node.getString('make'),
            camera_node.getString('model'),
            camera_node.getString('scale_width'),
            camera_node.getString('scale_height')
            )


def set_lens_params(ccd_width_mm, ccd_height_mm, focal_len_mm, scale_width, scale_height):
    camera_node.setFloat('ccd_width_mm', ccd_width_mm)
    camera_node.setFloat('ccd_height_mm', ccd_height_mm)
    camera_node.setFloat('focal_len_mm', focal_len_mm)
    camera_node.setFloat('scale_width', scale_width)
    camera_node.setFloat('scale_height', scale_height)

    h_fov, v_fov, diag_fov = calc_fov(focal_len_mm, ccd_width_mm, ccd_height_mm, scale_width, scale_height)

    camera_node.setFloat('h_fov_deg', round(np.rad2deg(h_fov), 3))
    camera_node.setFloat('v_fov_deg', round(np.rad2deg(v_fov), 3))
    camera_node.setFloat('diag_fov_deg', round(np.rad2deg(diag_fov), 3))


def get_image_params():
    bias = [camera_node.getFloatEnum('bias', i) for i in range(4)]
    tmp_pattern = [camera_node.getIntEnum('bayer_pattern', i) for i in range(4)]
    bayer_pattern = np.copy(np.array(tmp_pattern, dtype=int)).reshape(2, 2)

    return (camera_node.getFloat('pixel_size'),
            camera_node.getInt('width_px_full'),
            camera_node.getInt('height_px_full'),
            camera_node.getInt('width_px'),
            camera_node.getInt('height_px'),
            bias,
            camera_node.getInt('n_colours'),
            camera_node.getString('colour_description'),
            bayer_pattern, camera_node.getInt('bit_depth'))


def set_image_params(pixel_size, width_px, height_px,
                     bias, n_colors, colors, bayer_pattern, bit_depth):
    camera_node.setFloat('pixel_size', round(pixel_size, 4))

    camera_node.setInt('width_px_full', width_px)
    camera_node.setInt('height_px_full', height_px)

    camera_node.setInt('width_px', width_px // 2)
    camera_node.setInt('height_px', height_px // 2)

    camera_node.setString('colour_description', colors)
    camera_node.setInt('n_colours', n_colors)

    [camera_node.setFloatEnum('bias', i, bias[i]) for i in range(4)]
    camera_node.setInt('bit_depth', bit_depth)

    tmp = np.array(bayer_pattern, dtype=int).ravel().tolist()
    [camera_node.setIntEnum('bayer_pattern', i, tmp[i]) for i in range(4)]


def calc_fov(focal_length, sensor_width, sensor_height, scale_width, scale_height):
    """"""

    diag_mm = np.sqrt(sensor_width ** 2 + sensor_height ** 2)

    # FoV
    h_fov = 2. * np.arctan2(sensor_width, (2. * focal_length))
    v_fov = 2. * np.arctan2(sensor_height, (2. * focal_length))

    h_fov_eff = h_fov * scale_width
    v_fov_eff = v_fov * scale_height
    print(h_fov, h_fov_eff, np.rad2deg(h_fov), np.rad2deg(h_fov_eff))
    print(v_fov, v_fov_eff, np.rad2deg(v_fov), np.rad2deg(v_fov_eff))

    diag_fov = 2. * np.arctan2(diag_mm, (2. * focal_length))
    diag_fov_eff = 2. * np.arctan(np.sqrt(np.arctan2(h_fov_eff, 2)**2 + np.arctan2(v_fov_eff, 2)**2))
    print(diag_fov, diag_fov_eff, np.rad2deg(diag_fov), np.rad2deg(diag_fov_eff))
    return h_fov_eff, v_fov_eff, diag_fov_eff


def set_mount_params(yaw_deg, pitch_deg, roll_deg):
    mount_node = camera_node.getChild('mount', True)
    mount_node.setFloat('yaw_deg', yaw_deg)
    mount_node.setFloat('pitch_deg', pitch_deg)
    mount_node.setFloat('roll_deg', roll_deg)


def get_mount_params():
    mount_node = camera_node.getChild('mount', True)
    return [mount_node.getFloat('yaw_deg'),
            mount_node.getFloat('pitch_deg'),
            mount_node.getFloat('roll_deg')]


def get_body2cam():
    yaw_deg, pitch_deg, roll_deg = get_mount_params()
    body2cam = quaternion_from_euler(np.deg2rad(yaw_deg),
                                     np.deg2rad(pitch_deg),
                                     np.deg2rad(roll_deg), axes="rzyx")
    return body2cam


def get_K(optimized=False):
    """
    Form the camera calibration matrix K using five parameters of a
    Finite Projective Camera model.  (Note skew parameter is 0)

    See Eqn (6.10) in:
    R.I. Hartley & A. Zisserman, Multiview Geometry in Computer Vision,
    Cambridge University Press, 2004.
    """
    tmp = []
    if optimized and camera_node.hasChild('K_opt'):
        for i in range(9):
            tmp.append(camera_node.getFloatEnum('K_opt', i))
    else:
        for i in range(9):
            tmp.append(camera_node.getFloatEnum('K', i))
    K = np.copy(np.array(tmp)).reshape(3, 3)
    return K


def set_K(fx, fy, cu, cv, optimized=False):
    K = np.identity(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cu
    K[1, 2] = cv

    tmp = K.ravel().tolist()
    if optimized:
        camera_node.setLen('K_opt', 9)
        for i in range(9):
            camera_node.setFloatEnum('K_opt', i, tmp[i])
    else:
        camera_node.setLen('K', 9)
        for i in range(9):
            camera_node.setFloatEnum('K', i, tmp[i])


def calc_K(ccd_width_mm, ccd_height_mm, focal_len_mm, width_px, height_px):

    fx = focal_len_mm * width_px / ccd_width_mm
    fy = focal_len_mm * height_px / ccd_height_mm
    cu = width_px * 0.5
    cv = height_px * 0.5
    print('ccd: %.3f x %.3f' % (ccd_width_mm, ccd_height_mm))
    print('fx fy = %.2f %.2f' % (fx, fy))
    print('cu cv = %.2f %.2f' % (cu, cv))
    return fx, fy, cu, cv
