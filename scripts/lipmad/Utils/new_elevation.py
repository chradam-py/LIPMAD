#  Copyright (c) 2024.
#  __author__ = "Dean Hand"
#  __license__ = "AGPL"
#  __version__ = "1.0"

from rasterio import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from scipy.ndimage import map_coordinates
from urllib.request import urlopen
from urllib.error import HTTPError
import json
# from logger_config import *
from . import config as config
from .. import base_conf
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator

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

        cache_file = config.dtm_path
        if not os.path.exists(cache_file):
            log.error(f"File not found: {cache_file}")
            return False
        log.info(f"  SRTM: parsing .hgt file: {cache_file}")
        with open(cache_file, "rb") as f:
            self.srtm_z = np.fromfile(f, np.dtype('>i2'), tile_size * tile_size).reshape((tile_size, tile_size))

        return True

    def make_lla_interpolator(self):
        log.info("  SRTM: constructing LLA interpolator")

        # Copy data directly, assuming self.srtm_z is correctly shaped
        srtm_pts = np.copy(self.srtm_z)

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
        # print(self.lla_interp([-23.597769736, -70.385284747]))

    def lla_interpolate(self, point_list):
        return self.lla_interp(point_list)


class ElevationAdjuster:
    def __init__(self, elevation_data, crs, affine_transform):
        self.elevation_data = elevation_data
        self.crs = crs  # Store the CRS
        self.affine_transform = affine_transform

    def terrain_adjustment(self, col, row):
        try:
            row_f, col_f = float(row), float(col)
            interpolated_elevation = map_coordinates(self.elevation_data, [[row_f], [col_f]], order=1, mode='nearest')[
                0]
            return interpolated_elevation
        except Exception as e:
            log.error(
                f"Error calculating interpolated elevation: {e} for {config.im_file_name}. "
                f"Switching to Default Altitudes.")
            return config.abso_altitude


def load_elevation_data_and_crs():
    if config.dtm_path is not None:
        dsm = ''
        crs = ''
        src = ''
        affine_transform = ''
        with rasterio.open(config.dtm_path) as src:
            dsm = src.read(1)
            crs = src.crs  # Get the CRS directly
            affine_transform = src.transform  # Get the affine transform
        return dsm, crs, src, affine_transform


def translate_geo_to_utm(drone_longitude, drone_latitude):
    elevation_data, crs, _, affine_transform = load_elevation_data_and_crs()
    adjuster = ElevationAdjuster(elevation_data, crs, affine_transform)

    # Initialize transformer to convert from geographic coordinates to the CRS of the raster
    transformer = Transformer.from_crs("EPSG:4326", adjuster.crs, always_xy=True)

    # Transform drone coordinates
    utm_x, utm_y = transformer.transform(drone_longitude, drone_latitude)
    adjuster = ElevationAdjuster(elevation_data, crs, affine_transform)
    return utm_x, utm_y, adjuster


def get_altitude_at_point(x, y):
    srtm = config.srtm
    elevation = srtm.lla_interp([x, y])[0]

    if not np.isnan(elevation) and elevation > -32768:
        new_altitude = config.abso_altitude - elevation
        return new_altitude
    else:
        print(f"Point ({x}, {y}) is outside the elevation data bounds for file {config.im_file_name}. "
              f"Switching to default elevation.")
        return None


def get_altitude_from_open(lat, long):
    yy = 0
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{long}"
        response = urlopen(url)
        data = response.read().decode('utf-8')
        elevation = json.loads(data)['results'][0]['elevation']
        new_altitude = config.abso_altitude - elevation

        return new_altitude
    except HTTPError as err:
        print(
            f"Unable to Connect to OpenElevation for file {config.im_file_name}. Switching to Default Altitudes. Error: {err}")
        yy += 1
        if yy > 20:
            print("Too many failures in this dataset. Switching to default elevation.")
            config.update_elevation(False)
        return None
