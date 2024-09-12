#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
import navpy
from .Utils import config as config
from . import base_conf

tile_smrt = None
ned_interp = None
tile_size = 3601
fill_value = -32768

log = base_conf.log


class SRTM:
    def __init__(self):
        self.lat, self.lon = -24, -71
        self.srtm_z = None
        self.lla_interp = None

    def parse(self):
        tilename = "S24W071"  # Tile name
        cache_file = f'../../data/S24W071.SRTMGL1/{tilename}.hgt'
        cache_file = config.dtm_path
        if not os.path.exists(cache_file):
            print(f"File not found: {cache_file}")
            return False
        print(f"SRTM: parsing .hgt file: {cache_file}")
        with open(cache_file, "rb") as f:
            self.srtm_z = np.fromfile(f, np.dtype('>i2'), tile_size * tile_size).reshape((tile_size, tile_size))
            # contents = f.read()
        # self.srtm_z = np.frombuffer(contents, dtype='>i2').reshape((tile_size, tile_size))
        return True

    def make_lla_interpolator(self):
        print("SRTM: constructing LLA interpolator")

        # Copy data directly, assuming self.srtm_z is correctly shaped
        srtm_pts = np.copy(self.srtm_z)
        # import matplotlib.pyplot as plt
        # plt.imshow(srtm_pts)
        # plt.show()
        # Apply conditions on the array
        # Values set to 0.0 where conditions are met: value equals 65535, is less than 0, or greater than 10000
        condition = (srtm_pts == 65535) | (srtm_pts < 0) | (srtm_pts > 10000)
        srtm_pts[condition] = 0.0

        # Flip the array along the vertical axis because SRTM data is top-down and NumPy's default is bottom-up
        # srtm_pts = np.flipud(srtm_pts)

        x = np.linspace(self.lon, self.lon + 1, tile_size)
        y = np.linspace(self.lat + 1, self.lat, tile_size)

        self.lla_interp = RegularGridInterpolator(points=(y, x), values=srtm_pts,
                                                  bounds_error=False,
                                                  fill_value=fill_value)
        print(self.lla_interp([-23.597769736, -70.385284747]))

    def lla_interpolate(self, point_list):
        return self.lla_interp(point_list)


def initialize(lla_ref, width_m, height_m, step_m):
    print("Initializing the SRTM interpolator with a single tile")
    srtm = SRTM()  # Adapt lat, lon, and path as needed
    if srtm.parse():
        srtm.make_lla_interpolator()
    else:
        print("Failed to parse the SRTM tile.")

    make_interpolator(lla_ref, width_m, height_m, step_m, srtm)


def make_interpolator(lla_ref, width_m, height_m, step_m, srtm):
    print("SRTM: constructing NED area interpolator for a single tile")
    # Define the grid dimensions
    rows = int(height_m / step_m) + 1
    cols = int(width_m / step_m) + 1

    # Calculate the NED grid points
    n_list = np.linspace(-height_m * 0.5, height_m * 0.5, rows)
    e_list = np.linspace(-width_m * 0.5, width_m * 0.5, cols)
    ned_pts = np.array([[n, e, 0] for e in e_list for n in n_list])

    # Convert NED points to latitude and longitude using navpy
    lat_lon_alt = navpy.ned2lla(ned_pts, lla_ref[0], lla_ref[1], lla_ref[2])
    # print(lat_lon_alt)
    ll_pts = np.array(lat_lon_alt)[:2, :].T  # Extract just the latitude and longitude
    # print(ll_pts)
    ned_ds = np.full((rows, cols), -32768, dtype=float)
    # zs = srtm.lla_interp(ll_pts)
    #
    # for idx, z in enumerate(zs):
    #     r, c = divmod(idx, rows)
    #     if z > -10000:
    #         ned_ds[r, c] = z

    # Interpolate the elevation data at each lat, lon point
    for idx, (lat, lon) in enumerate(ll_pts):
        elevation = srtm.lla_interpolate([lat, lon])
        r, c = divmod(idx, cols)  # Note: ensure the correct ordering here
        if elevation > -10000:  # Assuming -10000 is a placeholder for invalid data
            ned_ds[r, c] = elevation
    # Visualization of the data
    # import matplotlib.pyplot as plt
    # plt.imshow(ned_ds, origin='lower')
    # plt.colorbar(label='Elevation (m)')
    # plt.show()

    # Creating an interpolator for NED grid
    global ned_interp
    ned_interp = RegularGridInterpolator((n_list, e_list), ned_ds.T,  # Note the transpose of ned_ds
                                         bounds_error=False, fill_value=-32768)


def make_interpolator_old(lla_ref, width_m, height_m, step_m, srtm):
    print("SRTM: constructing NED area interpolator for a single tile")
    # Define the grid dimensions
    rows = int(height_m / step_m) + 1
    cols = int(width_m / step_m) + 1

    # Calculate the NED grid points
    n_list = np.linspace(-height_m * 0.5, height_m * 0.5, rows)
    e_list = np.linspace(-width_m * 0.5, width_m * 0.5, cols)
    ned_pts = []
    for e in e_list:
        for n in n_list:
            ned_pts.append([n, e, 0])
    print(lla_ref)
    navpy_pts = navpy.ned2lla(ned_pts, lla_ref[0], lla_ref[1], lla_ref[2])
    ll_pts = []
    for i in range(len(navpy_pts[0])):
        lat = navpy_pts[0][i]
        lon = navpy_pts[1][i]
        ll_pts.append([lon, lat])

    ned_ds = np.full((rows, cols), -32768)
    zs = srtm.lla_interpolate(ll_pts)

    for r in range(0, rows):
        for c in range(0, cols):
            idx = (rows * c) + r
            if zs[idx] > -10000:

                ned_ds[r, c] = zs[idx]
    # print("ned_ds:", ned_ds)
    import matplotlib.pyplot as plt
    plt.imshow(ned_ds)
    plt.show()
    global ned_interp
    ned_interp = RegularGridInterpolator((n_list, e_list), ned_ds,
                                         bounds_error=False, fill_value=-32768)


def get_ground(v):
    tmp = ned_interp([v[0], v[1]])
    if not np.isnan(tmp[0]) and tmp[0] > -32768:
        ground = tmp[0]
    else:
        ground = 0.0

    return ground


def interpolate_vector(ned, v):
    eps = 0.01
    count = 0
    p = v[:]

    # print("start:", p)
    # print("vec:", v)
    # print("ned:", ned)
    tmp = ned_interp([p[0], p[1]])
    if not np.isnan(tmp[0]) and tmp[0] > -32768:
        ground = tmp[0]
    else:
        ground = 0.0
    error = abs(p[2] + ground)
    print("  p=%s ground=%s error=%s" % (p, ground, error))
    while error > eps and count < 25:
        d_proj = -(ned[2] + ground)
        factor = d_proj / v[2]
        n_proj = v[0] * factor
        e_proj = v[1] * factor
        # print("proj = %s %s" % (n_proj, e_proj))
        p = [ned[0] + n_proj, ned[1] + e_proj, ned[2] + d_proj]
        # print("new p:", p)
        tmp = ned_interp([p[0], p[1]])
        if not np.isnan(tmp[0]) and tmp[0] > -32768:
            ground = tmp[0]
        error = abs(p[2] + ground)

        p[2] = ground
        print("  p=%s ground=%.2f error = %.3f" % (p, ground, error))
        count += 1
    print("  p=%s ground=%.2f error = %.3f" % (p, ground, error))
    return p


def interpolate_vectors(ned, v_list):
    pt_list = []
    for v in v_list:
        p = interpolate_vector(ned, v.flatten())
        pt_list.append(p)
    return pt_list
