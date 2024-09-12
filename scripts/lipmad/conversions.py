#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import colour
import colour_hdri
from typing import Literal
from pyproj import CRS, Transformer, Geod
from osgeo import gdal, osr


def convert_kmn_to_mnk(data):
    """Transpose the data to change the shape from (k, m, n) to (m, n, k)."""
    return data.transpose(1, 2, 0)


def convert_mnk_to_kmn(data):
    """Transpose the data to change the shape from (m, n, k) to (k, m, n)."""
    return data.transpose(2, 0, 1)


def convert_CAM_to_XYZ(data, AsShotNeutral, M_color_matrix_1, M_color_matrix_2):
    """ Convert from the Camera Space to CIE XYZ """

    M_camera_calibration_1 = np.identity(3)
    M_camera_calibration_2 = np.identity(3)
    analog_balance = np.ones(3)

    # Convert given Camera Neutral coordinates to xy white balance
    # chromaticity coordinates.
    xy = colour_hdri.camera_neutral_to_xy(
        camera_neutral=AsShotNeutral,
        CCT_calibration_illuminant_1=2856,
        CCT_calibration_illuminant_2=6504,
        M_color_matrix_1=M_color_matrix_1,
        M_color_matrix_2=M_color_matrix_2,
        M_camera_calibration_1=M_camera_calibration_1,
        M_camera_calibration_2=M_camera_calibration_2,
        analog_balance=analog_balance)

    # Inverse of the CIE XYZ to Camera Space matrix
    M_CAMERA_RGB_to_XYZ_D50 = np.linalg.inv(
        colour_hdri.matrix_XYZ_to_camera_space(
            xy,
            CCT_calibration_illuminant_1=2856,
            CCT_calibration_illuminant_2=6504,
            M_color_matrix_1=M_color_matrix_1,
            M_color_matrix_2=M_color_matrix_2,
            M_camera_calibration_1=M_camera_calibration_1,
            M_camera_calibration_2=M_camera_calibration_2,
            analog_balance=analog_balance))

    # Apply colour matrix to image
    XYZ_IMAGE = colour.algebra.vector_dot(M_CAMERA_RGB_to_XYZ_D50,
                                          data)
    XYZ_IMAGE = np.array(XYZ_IMAGE, dtype=np.float32)

    # XYZ_IMAGE = np.clip(XYZ_IMAGE, 0., 1.)

    return XYZ_IMAGE


def convert_XYZ_to_sRGB(xyz_image):
    """ Convert from CIE XYZ tristimulus values to an RGB colourspace array."""

    from colour.models import RGB_COLOURSPACE_CIE_RGB
    from colour_hdri.models.datasets.dng import CCS_ILLUMINANT_ADOBEDNG

    XYZ_IMAGE = np.clip(xyz_image, 0., 1.)

    sRGB_IMAGE = colour.XYZ_to_RGB(
        XYZ=XYZ_IMAGE,
        colourspace=RGB_COLOURSPACE_CIE_RGB,
        # colourspace='sRGB',
        illuminant_RGB=CCS_ILLUMINANT_ADOBEDNG,
        chromatic_adaptation_transform='XYZ Scaling',
        # chromatic_adaptation_transform=None,
        apply_cctf_encoding=False)

    sRGB_IMAGE = np.clip(sRGB_IMAGE, 0., 1.)

    # sRGB_IMAGE = convert_mnk_to_kmn(sRGB_IMAGE)

    return sRGB_IMAGE


def convert_XYZ_to_xy(XYZ_IMAGE):
    """Return the CIE xy chromaticity coordinates from given CIE XYZ tristimulus values."""

    XYZ_IMAGE = np.clip(XYZ_IMAGE, 0., 1.)
    xy_IMAGE = colour.XYZ_to_xy(XYZ_IMAGE)

    return xy_IMAGE


def convert_xy_to_CCT(xy_image, method: Literal["Hernandez 1999"]):
    """Return the correlated colour temperature CCT from given CIE xy chromaticity coordinates."""

    x = np.array([colour.CCT_to_xy(cct, "Kang 2002") for cct in range(1667, 25000)])
    print(x, x.min(axis=0), x.max(axis=0))
    # Given minimum values for each channel
    min_values = x.min(axis=0)
    print(colour.xy_to_CCT([0.2, 0.2], 'Kang 2002'))

    # Reshape min_values for broadcasting
    min_values_reshaped = min_values.reshape((1, 1, 2))

    # Repeat min_values to match the shape of the array
    min_values_broadcast = min_values_reshaped.repeat(xy_image.shape[0], axis=0).repeat(xy_image.shape[1], axis=1)

    # Create a boolean mask where the array is less than the broadcast minimum values
    mask = xy_image < min_values_broadcast

    # Replace values in the array where the mask is True
    xy_image[mask] = min_values_broadcast[mask]

    # Given minimum values for each channel
    max_values = x.max(axis=0)

    # Reshape max_values for broadcasting
    max_values_reshaped = max_values.reshape((1, 1, 2))

    # Repeat max_values to match the shape of the array
    max_values_broadcast = max_values_reshaped.repeat(xy_image.shape[0], axis=0).repeat(xy_image.shape[1], axis=1)

    # Create a boolean mask where the array is less than the broadcast minimum values
    mask = xy_image > max_values_broadcast

    # Replace values in the array where the mask is True
    xy_image[mask] = max_values_broadcast[mask]
    # y = np.array([colour.CCT_to_xy(cct) for cct in range(4000, 25001)])
    # print(y, y.min(axis=0), y.max(axis=0))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1])
    # plt.scatter(y[:, 0], y[:, 1])

    CCT_IMAGE = colour.xy_to_CCT(xy_image, method=method)

    # Replace invalid values with NAN
    # CCT_IMAGE_masked = np.where(
    #     np.logical_and(CCT_IMAGE > 1667,
    #                    CCT_IMAGE < 2.5e4), CCT_IMAGE, 0.)
    plt.figure()
    plt.imshow(CCT_IMAGE)
    plt.show()

    quit()
    return CCT_IMAGE_masked


def pixel_size_in_steradians(sensor_width_mm, sensor_height_mm, resolution_x, resolution_y, focal_length_mm):
    import math

    # Convert focal length and sensor dimensions from mm to m
    focal_length_m = focal_length_mm / 1000
    sensor_width_m = sensor_width_mm / 1000
    sensor_height_m = sensor_height_mm / 1000

    # Calculate the field of view (FoV) in radians
    fov_x_rad = 2 * math.atan(sensor_width_m / (2 * focal_length_m))
    fov_y_rad = 2 * math.atan(sensor_height_m / (2 * focal_length_m))

    # Calculate the angular size of a pixel
    delta_theta_x = fov_x_rad / resolution_x
    delta_theta_y = fov_y_rad / resolution_y

    # Calculate the solid angle in steradians covered by a pixel
    omega_pixel = delta_theta_x * delta_theta_y

    return omega_pixel


def km_to_deg_latitude(km):
    """
    Convert kilometers to degrees (latitude).
    """
    return km / 110.574


def km_to_deg_longitude(km, latitude):
    """
    Convert kilometers to degrees (longitude), taking into account the latitude.
    """
    radius_of_earth_km = 6371.01  # Average radius of the Earth in kilometers
    # Calculate the radius of a circle at the given latitude
    radius_at_latitude_km = radius_of_earth_km * math.cos(math.radians(latitude))
    return km / (math.pi / 180. * radius_at_latitude_km)


def calculate_ground_coverage(center_lat, center_lon, altitude_m, image_width_px, image_height_px,
                              camera_fov_horizontal_deg, camera_fov_vertical_deg, orientation_deg):
    """
    Approximate the ground coverage of an image and calculate pixel dimensions in degrees,
    with more accurate km to degree conversion.
    """
    ground_coverage_width_km = 2. * (altitude_m / 1000.) * math.tan(math.radians(camera_fov_horizontal_deg / 2.))
    ground_coverage_height_km = 2. * (altitude_m / 1000.) * math.tan(math.radians(camera_fov_vertical_deg / 2.))

    ground_coverage_width_deg = km_to_deg_longitude(ground_coverage_width_km, center_lat)
    ground_coverage_height_deg = km_to_deg_latitude(ground_coverage_height_km)

    longitude_origin = center_lon - (ground_coverage_width_deg / 2.)
    latitude_origin = center_lat + (ground_coverage_height_deg / 2.)

    pixel_width = ground_coverage_width_deg / image_width_px
    pixel_height = ground_coverage_height_deg / image_height_px

    # Compute the rotation parameters based on the orientation
    rotation_x, rotation_y = calculate_rotation_parameters(orientation_deg, pixel_width, pixel_height)

    return [longitude_origin, pixel_width, 0, latitude_origin, 0, -pixel_height]


def georeference_data(input_data, data_type, drone_pos, tilt, h_fov, v_fov, orientation, output_file_path):
    """
    Georeferences an input image or segmentation map based on drone positioning and camera parameters.
    """
    rows, cols = input_data.shape[:2]
    num_bands = input_data.shape[2] if input_data.ndim == 3 else 1  # Check if input_data is RGB

    # Calculate the geographic coordinates of the image corners
    corners = compute_corners(drone_pos, tilt, h_fov, v_fov, orientation)
    print(corners)
    # corners = compute_corners_geodetic(drone_pos, tilt, h_fov, v_fov, orientation)
    # print(corners)
    # Define pixel coordinates for the image corners
    # image_width, image_height = input_data.shape[1], input_data.shape[0]
    pixel_coords = [(0, 0), (cols, 0), (cols, rows), (0, rows)]

    # Create GCPs using the corner coordinates and pixel coordinates
    gcps = [gdal.GCP(corner[1], corner[0], drone_pos[2], pixel[0], pixel[1]) for corner, pixel in
            zip(corners, pixel_coords)]
    for gcp in gcps:
        print(f"Pixel: ({gcp.GCPPixel}, {gcp.GCPLine}) => Geo: ({gcp.GCPX}, {gcp.GCPY}, {gcp.GCPZ})")
    print(input_data.dtype)

    # Define data type based on input
    if data_type == 'image':
        gdal_data_type = gdal.GDT_Byte
        if input_data.dtype == np.float32:
            gdal_data_type = gdal.GDT_Float32
        elif input_data.dtype == np.float64:
            gdal_data_type = gdal.GDT_Float32
    elif data_type == 'segmentation_map':
        gdal_data_type = gdal.GDT_Int32
    else:
        raise ValueError("Unsupported data_type. Choose 'image' or 'segmentation_map'.")

    # Create a new GDAL in-memory dataset
    mem_driver = gdal.GetDriverByName('MEM')
    mem_raster = mem_driver.Create('', cols, rows, num_bands, gdal_data_type)

    # Set geotransform and projection
    # mem_raster.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    mem_raster.SetProjection(srs.ExportToWkt())
    mem_raster.SetGCPs(gcps, srs.ExportToWkt())

    # Write data to the in-memory dataset
    if num_bands == 1:  # Single band (grayscale or segmentation map)
        mem_raster.GetRasterBand(1).WriteArray(input_data)
    else:  # RGB image
        for i in range(num_bands):
            mem_raster.GetRasterBand(i + 1).WriteArray(input_data[:, :, i])

    # Export the in-memory dataset to a file
    gdal.Warp(str(output_file_path), mem_raster, format='GTiff', dstSRS='EPSG:3857')

    # Cleanup
    mem_raster = None


def georeference_data2(input_data, data_type, drone_pos, tilt, h_fov, v_fov, orientation, output_file_path):
    """
    Georeferences an input image or segmentation map based on drone positioning and camera parameters.

    :param input_data: Numpy array of the image or segmentation map.
    :param data_type: Type of the input data ('image' or 'segmentation_map').
    :param drone_pos: Tuple of drone position (latitude, longitude, altitude).
    :param tilt: Camera tilt in degrees.
    :param h_fov: Horizontal field of view in degrees.
    :param v_fov: Vertical field of view in degrees.
    :param orientation: Camera orientation in degrees relative to north.
    :param output_file_path: Path to save the georeferenced output.
    """
    # Calculate the geographic coordinates of the image corners
    corners = compute_corners(drone_pos, tilt, h_fov, v_fov, orientation)
    print(corners)

    # Define pixel coordinates for the image corners
    image_width, image_height = input_data.shape[1], input_data.shape[0]
    pixel_coords = [(0, 0), (image_width, 0), (image_width, image_height), (0, image_height)]

    # Create GCPs using the corner coordinates and pixel coordinates
    gcps = [gdal.GCP(corner[1], corner[0], drone_pos[2], pixel[0], pixel[1]) for corner, pixel in
            zip(corners, pixel_coords)]
    for gcp in gcps:
        print(f"Pixel: ({gcp.GCPPixel}, {gcp.GCPLine}) => Geo: ({gcp.GCPX}, {gcp.GCPY}, {gcp.GCPZ})")

    # Create an in-memory raster
    mem_driver = gdal.GetDriverByName('MEM')
    if data_type == 'image':
        gdal_data_type = gdal.GDT_Byte
    elif data_type == 'segmentation_map':
        gdal_data_type = gdal.GDT_UInt16
    else:
        raise ValueError("Unsupported data_type. Choose 'image' or 'segmentation_map'.")

    mem_raster = mem_driver.Create('', image_width, image_height, 1, gdal_data_type)
    mem_raster.GetRasterBand(1).WriteArray(input_data)

    # Create a spatial reference object for WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # # Apply the GCPs to the in-memory dataset
    mem_raster.SetProjection(srs.ExportToWkt())
    mem_raster.SetGCPs(gcps, srs.ExportToWkt())

    # Georeference the in-memory dataset using the GCPs and output to a file
    gdal.Warp(str(output_file_path), mem_raster, format='GTiff', dstSRS='EPSG:3857')

    # Cleanup
    mem_raster = None


import folium
from osgeo import gdal


def get_image_bounds(georeferenced_image):
    """
    Extract the bounding box from the georeferenced image.
    :param georeferenced_image: Path to the georeferenced image.
    :return: Bounds as [[min_lat, min_lon], [max_lat, max_lon]]
    """
    dataset = gdal.Open(georeferenced_image)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    gt = dataset.GetGeoTransform()

    # Calculate bounds
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]

    return [[miny, minx], [maxy, maxx]]


def overlay_on_osm(georeferenced_image):
    """
    Overlay a georeferenced image on an OSM map using Folium.
    :param georeferenced_image: Path to the georeferenced image.
    """
    bounds = get_image_bounds(georeferenced_image)
    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]

    # Initialize the map at the center of the image
    m = folium.Map(location=center, zoom_start=14)

    # Add the georeferenced image as an overlay
    folium.raster_layers.ImageOverlay(
        image=georeferenced_image,
        bounds=bounds,
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        name='Georeferenced Image',
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save to HTML
    m.save('/home/cadam/final_map.html')


def compute_corners(drone_pos, tilt, h_fov, v_fov, orientation):
    """
    Computes the corners of the projected rectangle on the ground.

    Parameters:
    - drone_pos: (latitude, longitude, altitude) of the drone.
    - tilt: Tilt of the camera (0 is horizontal, -90 is straight down).
    - h_fov: Horizontal field of view of the camera in degrees.
    - v_fov: Vertical field of view of the camera in degrees.
    - orientation: Orientation of the drone (yaw) in degrees.

    Returns:
    - corners: List of (latitude, longitude) for the four corners of the projected rectangle.
    """

    # Compute ground distance from drone to the center of the image based on tilt
    d_center = drone_pos[2] * np.tan(np.radians(90. - abs(tilt)))

    # Compute the displacements for the center of the image
    delta_lat_center = d_center * np.cos(np.radians(orientation))
    delta_lon_center = d_center * np.sin(np.radians(orientation))

    center_lat = drone_pos[0] + delta_lat_center / (111.32 * 1000)  # convert from meters to degrees
    center_lon = drone_pos[1] + delta_lon_center / (111.32 * 1000 * np.cos(np.radians(drone_pos[0])))

    # Compute half-diagonals based on FOVs
    half_width = drone_pos[2] * np.tan(np.radians(h_fov / 2))
    half_height = drone_pos[2] * np.tan(np.radians(v_fov / 2))

    # Compute corner displacements
    corners = []
    for dx, dy in [(-half_width, half_height),
                   (half_width, half_height),
                   (half_width, -half_height),
                   (-half_width, -half_height)]:

        # Take into account the drone's orientation
        delta_lat = dx * np.sin(np.radians(orientation)) + dy * np.cos(np.radians(orientation))
        delta_lon = dx * np.cos(np.radians(orientation)) - dy * np.sin(np.radians(orientation))

        lat = center_lat + delta_lat / (111.32 * 1000)
        lon = center_lon + delta_lon / (111.32 * 1000 * np.cos(np.radians(center_lat)))
        corners.append((lat, lon))

    return corners


def compute_corners_geodetic_old(drone_pos, tilt, h_fov, v_fov, orientation):
    lat, lon, alt = drone_pos  # Drone position: latitude, longitude, altitude

    # Initialize geodetic calculations with WGS84 ellipsoid
    geod = Geod(ellps="WGS84")

    # Compute distance to the center point on the ground
    d = alt / np.cos(np.radians(90 - abs(tilt)))  # Adjust for tilt

    # Calculate the extents of the footprint based on the field of view and altitude
    half_diag = np.sqrt((d * np.tan(np.radians(h_fov / 2))) ** 2 + (d * np.tan(np.radians(v_fov / 2))) ** 2)

    # Define bearings from the orientation to reach each corner
    # Assuming orientation is aligned with the top edge of the image
    bearings = [(orientation - 45) % 360, (orientation + 45) % 360,
                (orientation + 135) % 360, (orientation + 225) % 360]

    print("Inputs:")
    print(f"Latitude: {lat}, Longitude: {lon}, Altitude: {alt}")
    print(f"Yaw (Orientation): {orientation}, Pitch (Tilt): {tilt}, Roll: Not used")
    print(f"Horizontal FoV: {h_fov}, Vertical FoV: {v_fov}")
    print(f"Computed distance to center: {d}")
    print(f"Half-diagonal of footprint: {half_diag}")

    print("Computed Bearings (degrees):")
    print(bearings)

    print("Corner Coordinates:")
    corners = []
    for bearing in bearings:
        end_lat, end_lon, _ = geod.fwd(lat, lon, bearing, half_diag)
        print(f"Bearing: {bearing}, Corner (Lat, Lon): ({end_lat}, {end_lon})")
        corners.append((end_lat, end_lon))

    return corners


def compute_corners_geodetic(drone_pos, tilt, h_fov, v_fov, orientation):
    """
    Compute the geodetic corners of the image footprint on the ground, ensuring that
    a 0-degree bearing aligns the y-axis with the north-south direction.
    """
    lat, lon, alt = drone_pos
    geod = Geod(ellps="WGS84")
    d = alt / np.cos(np.radians(90 - abs(tilt)))
    # Calculate half-diagonal distance for corner calculation
    half_diag = np.sqrt((d * np.tan(np.radians(h_fov / 2))) ** 2 + (d * np.tan(np.radians(v_fov / 2))) ** 2)

    # # Adjust bearings to account for north-south alignment at 0-degree orientation
    # bearings = [(orientation - 45) % 360, (orientation + 45) % 360,
    #             (orientation + 135) % 360, (orientation + 225) % 360]
    # Adjust bearings to ensure y-axis alignment with north at 0-degree orientation
    bearings = [(orientation + 135) % 360,  # Forward direction (north)
                (orientation + 45) % 360,  # Right direction (east)
                (orientation - 45) % 360,  # Backward direction (south)
                (orientation - 135) % 360]  # Left direction (west)

    print("Inputs:")
    print(f"Latitude: {lat}, Longitude: {lon}, Altitude: {alt}")
    print(f"Yaw (Orientation): {orientation}, Pitch (Tilt): {tilt}, Roll: Not used")
    print(f"Horizontal FoV: {h_fov}, Vertical FoV: {v_fov}")
    print(f"Computed distance to center: {d}")
    print(f"Half-diagonal of footprint: {half_diag}")

    print("Computed Bearings (degrees):")
    print(bearings)

    print("Corner Coordinates:")
    corners = []
    for bearing in bearings:
        end_lat, end_lon, _ = geod.fwd(lat, lon, bearing, half_diag)
        print(f"Bearing: {bearing}, Corner (Lat, Lon): ({end_lat}, {end_lon})")
        corners.append((end_lat, end_lon))

    return corners


def calculate_rotation_parameters(yaw_deg, pixel_width, pixel_height):
    """
    Calculate the rotation parameters for the geotransform array based on the yaw angle.
    Yaw is the rotation in degrees from north (clockwise).
    """
    # Convert the yaw angle from degrees to radians
    yaw_rad = math.radians(-yaw_deg)  # Negative because rotation is clockwise

    # Calculate the rotation parameters
    rotation_x = math.sin(yaw_rad)  # Rotation about the y-axis
    rotation_y = -math.sin(yaw_rad)  # Rotation about the x-axis

    # Adjust the rotation factors by the size of a pixel
    rotation_x *= pixel_width
    rotation_y *= pixel_height

    return rotation_x, rotation_y


def calculate_corrected_position(center_lat, center_lon, altitude_m, tilt_deg, yaw_deg):
    """
    Calculate the corrected GPS position on the ground, accounting for camera tilt and orientation.
    """
    geod = Geod(ellps="WGS84")

    # Adjust for camera tilt (pitch)
    # Calculate the distance from the drone to the point directly below the center of the camera view
    d = altitude_m * math.tan(math.radians(90. - abs(tilt_deg)))

    # Calculate the new ground position using the forward geodetic calculation
    corrected_lat, corrected_lon, _ = geod.fwd(center_lat, center_lon, yaw_deg, d)

    return corrected_lat, corrected_lon


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import rasterio


def plot_georeferenced_image(image_path, output_path):
    """
    Plot a georeferenced image onto a map projection using rasterio and Matplotlib.

    :param image_path: Path to the georeferenced image.
    :param output_path: Path where the output plot will be saved.
    """
    with rasterio.open(image_path) as src:
        # Get the raster data as an array
        data_array = src.read(1)

        # Mask the array for no data values
        no_data_value = src.nodatavals[0]
        if no_data_value is not None:
            data_array = np.ma.masked_where(data_array == no_data_value, data_array)

        # Get the metadata of the raster
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

        # Get the extent of the raster
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        # Setup the figure and axis with a specified map projection
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot the image using the extent and transformation from rasterio
        img = ax.imshow(data_array, origin='upper', extent=extent, transform=ccrs.PlateCarree(), alpha=0.7,
                        cmap='viridis')

        # Optionally add coastlines or other features
        ax.coastlines()

        # Save the generated plot to the specified output path
        plt.savefig(output_path)
        plt.close(fig)


import subprocess


def convert_geotiff_to_png(geotiff_path, output_png_path):
    command = ['gdal_translate', '-of', 'PNG', geotiff_path, output_png_path]
    subprocess.run(command)


import pyproj
from scipy.spatial.transform import Rotation as R


# Function to convert GPS to UTM
def convert_gps_to_utm(lat, lon):
    projection = pyproj.Proj(proj='utm', zone=33, ellps='WGS84')  # Adjust the zone
    return projection(lon, lat)


# Function to convert UTM to GPS
def convert_utm_to_gps(easting, northing, zone):
    projection = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84', inverse=True)
    return projection(easting, northing)


# Function to calculate corrected location using quaternion
def calculate_corrected_location_quaternion(lat, lon, altitude, yaw, pitch, roll):
    # Convert GPS to UTM
    utm_x, utm_y = convert_gps_to_utm(lat, lon)
    zone = math.floor((lon + 180) / 6) + 1

    # Convert Yaw, Pitch, Roll to Quaternion
    quaternion = R.from_euler('zyx', [yaw, 90 - abs(pitch), roll], degrees=True).as_quat(canonical=False)
    print(quaternion)
    # Downward-pointing unit vector
    down_vector = np.array([0, 0, -1])

    # Apply quaternion rotation
    rotated_vector = R.from_quat(quaternion).apply(down_vector)

    # Project onto ground plane
    x_shift = altitude * rotated_vector[0] / rotated_vector[2]
    y_shift = altitude * rotated_vector[1] / rotated_vector[2]

    # Apply shifts to UTM coordinates
    corrected_utm_x = utm_x + x_shift
    corrected_utm_y = utm_y + y_shift

    # Convert back to GPS
    corrected_lat, corrected_lon = convert_utm_to_gps(corrected_utm_x, corrected_utm_y, zone)

    return corrected_lat, corrected_lon, (x_shift, y_shift)


# Function to calculate corrected location using simple rotation matrices
def calculate_corrected_location_simple(lat, lon, altitude, yaw, pitch, roll):
    # Convert GPS to UTM
    utm_x, utm_y = convert_gps_to_utm(lat, lon)
    zone = math.floor((lon + 180) / 6) + 1

    # Convert angles from degrees to radians
    yaw_radians = math.radians(yaw)
    pitch_radians = math.radians(90 - abs(pitch))  # Adjust pitch
    roll_radians = math.radians(roll)

    # Define rotation matrices
    R_z = np.array([
        [math.cos(yaw_radians), -math.sin(yaw_radians), 0],
        [math.sin(yaw_radians), math.cos(yaw_radians), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [math.cos(pitch_radians), 0, -math.sin(pitch_radians)],
        [0, 1, 0],
        [math.sin(pitch_radians), 0, math.cos(pitch_radians)]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll_radians), -math.sin(roll_radians)],
        [0, math.sin(roll_radians), math.cos(roll_radians)]
    ])

    # Apply the rotations to the initial vector
    initial_vector = np.array([0, 0, -1])
    camera_direction = R_x @ R_y @ R_z @ initial_vector

    # Calculate the ground shifts
    x_shift = altitude * camera_direction[0] / -camera_direction[2] if camera_direction[2] != 0 else 0
    y_shift = altitude * camera_direction[1] / -camera_direction[2] if camera_direction[2] != 0 else 0

    # Apply shifts to UTM coordinates
    corrected_utm_x = utm_x + x_shift
    corrected_utm_y = utm_y + y_shift

    # Convert back to GPS
    corrected_lat, corrected_lon = convert_utm_to_gps(corrected_utm_x, corrected_utm_y, zone)

    return corrected_lat, corrected_lon, (x_shift, y_shift)
