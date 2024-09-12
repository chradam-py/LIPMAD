#  Copyright (c) 2024.
#  __author__ = "Dean Hand"
#  __license__ = "AGPL"
#  __version__ = "1.0"


import numpy as np
import quaternion
from .Utils.geospatial_conversions import *
from mpmath import mp, radians, sqrt
from vector3d.vector import Vector
from .Utils.new_elevation import get_altitude_at_point, get_altitude_from_open
from .Utils import config as config
from .Utils.declination import find_declination
# from .Utils.logger_config import *

latitude = 0
longitude = 0
datetime_original = None
lens_FOVh = 0.0
lens_FOVw = 0.0
mp.dps = 50  # set a higher precision


class HighAccuracyFOVCalculator:
    def __init__(self, drone_gps, drone_altitude, camera_info, gimbal_orientation, flight_orientation,
                 datetime_original, w, h, K_inv):
        global latitude, longitude, lens_FOVh, lens_FOVw
        self.drone_gps = drone_gps
        latitude = drone_gps[0]
        longitude = drone_gps[1]
        self.datetime_original = datetime_original
        self.drone_altitude = drone_altitude
        self.camera_info = camera_info
        lens_FOVh = camera_info['lens_FOVh']
        lens_FOVw = camera_info['lens_FOVw']
        self.gimbal_orientation = gimbal_orientation
        self.flight_orientation = flight_orientation
        self.w = w
        self.h = h
        self.K_inv = K_inv

    def calculate_fov_dimensions(self):
        FOVw = 2 * mp.atan(mp.mpf(self.camera_info['sensor_width']) / (2 * self.camera_info['Focal_Length']))
        FOVh = 2 * mp.atan(mp.mpf(self.camera_info['sensor_height']) / (2 * self.camera_info['Focal_Length']))

        adj_FOVw, adj_FOVh = HighAccuracyFOVCalculator._sensor_lens_correction(FOVw, FOVh)

        return adj_FOVw, adj_FOVh

    @staticmethod
    def calculate_rads_from_angles(gimbal_yaw_deg, gimbal_pitch_deg, gimbal_roll_deg, declination):
        """
        Adjusts the gimbal's angles for magnetic declination and normalizes the roll orientation.
        - Yaw is adjusted for magnetic declination.
        - Roll is normalized if within a specific range to handle edge cases around 90 degrees.
        - Pitch is converted directly to radians.

        Parameters:
        - gimbal_yaw_deg (float): The gimbal's yaw angle in degrees.
        - gimbal_pitch_deg (float): The gimbal's pitch angle in degrees.
        - gimbal_roll_deg (float): The gimbal's roll angle in degrees.
        - declination (float): Magnetic declination in degrees.

        Returns:
        - tuple: Adjusted yaw, pitch, and roll angles in radians.
        """
        # Normalize yaw for magnetic declination
        # if config.decl:
        #     yaw_rad = (mp.pi / 2) - radians(gimbal_yaw_deg + declination)
        # else:
        #     yaw_rad = (mp.pi / 2) - radians(gimbal_yaw_deg)
        yaw_rad = (mp.pi / 2) - radians(gimbal_yaw_deg)

        # Normalize yaw within the range [0, 2π]
        yaw_rad = yaw_rad % (2 * mp.pi)

        # Convert pitch to radians and calculate as deviation from vertical
        pitch_rad = (mp.pi / 2) - radians(gimbal_pitch_deg)

        roll_rad = radians(gimbal_roll_deg)

        return yaw_rad, pitch_rad, roll_rad

    def getBoundingPolygon(self, FOVh, FOVv):
        """
        Calculates the bounding polygon of a camera's footprint given its field of view, position, and orientation.

        Parameters:
            FOVh (float): The horizontal field of view in radians.
            FOVv (float): The vertical field of view in radians.

        Returns:
            list: A list of Vector objects representing the corners of the bounding polygon on the ground.
        """

        # Define camera rays based on field of view
        rays = [
            Vector(-mp.tan(FOVv / 2), mp.tan(FOVh / 2), 1).normalize(),  # Flip horizontally
            Vector(-mp.tan(FOVv / 2), -mp.tan(FOVh / 2), 1).normalize(),  # Flip horizontally
            Vector(mp.tan(FOVv / 2), -mp.tan(FOVh / 2), 1).normalize(),  # Flip horizontally
            Vector(mp.tan(FOVv / 2), mp.tan(FOVh / 2), 1).normalize()  # Flip horizontally
        ]
        # Rotate rays according to camera orientation
        rotated_vectors = self.rotateRays(rays)

        return rotated_vectors

    def rotateRays(self, rays):
        # Calculate adjusted angles for gimbal and flight orientations
        declination = find_declination(self.drone_altitude, self.camera_info['Focal_Length'], *self.drone_gps,
                                       self.datetime_original)
        adj_yaw, adj_pitch, adj_roll = self.calculate_rads_from_angles(self.gimbal_orientation['yaw'],
                                                                       self.gimbal_orientation['pitch'],
                                                                       self.gimbal_orientation['roll'],
                                                                       declination)

        q = quaternion.from_euler_angles(adj_yaw, adj_pitch, adj_roll)
        # Normalize the quaternion
        q = q.normalized()

        # Apply rotation to each ray
        rotated_rays = [Vector(*(q * np.quaternion(0, ray.x, ray.y, ray.z) * q.inverse()).vec) for ray in rays]
        return rotated_rays

    @staticmethod
    def get_ray_direction2(px, py, K_inv, sx=1, sy=1):
        """
        Calculate the 3D ray direction for any pixel in the image using the camera's intrinsic matrix.

        Parameters:
            px (int): X-coordinate of the pixel.
            py (int): Y-coordinate of the pixel.
            K_inv (np.array): Inverse of the camera's intrinsic matrix.

        Returns:
            np.array: Normalized 3D direction vector of the ray.
        """
        # Convert pixel coordinates to homogeneous coordinates
        pixel_vector = np.array([px, py, 1])

        # Apply the inverse camera matrix to get direction in camera space
        ray_direction = np.dot(K_inv, pixel_vector)
        # Normalize the ray direction
        ray_direction /= np.linalg.norm(ray_direction)
        # print(ray_direction)
        direction_vector = Vector(sx*ray_direction[1], sy*ray_direction[0], ray_direction[2]).normalize()  # Normalize the vector

        return direction_vector

    @staticmethod
    def get_ray_direction(px, py, w, h, FOVh, FOVv, sx=1, sy=1):
        """
        Calculate the 3D ray direction for any pixel in the image based on the camera's field of view.

        Parameters:
            px (int): X-coordinate of the pixel.
            py (int): Y-coordinate of the pixel.
            w (int): Width of the image in pixels.
            h (int): Height of the image in pixels.
            FOVh (float): Horizontal field of view in radians.
            FOVv (float): Vertical field of view in radians.

        Returns:
            Vector: Normalized 3D direction vector of the ray.
        """
        theta = ((py - h / 2) / (h / 2)) * (FOVv / 2)
        phi = ((px - w / 2) / (w / 2)) * (FOVh / 2)
        x = mp.tan(theta) * sx
        y = mp.tan(phi) * sy
        z = 1
        direction_vector = Vector(x, y, z).normalize()  # Normalize the vector

        return direction_vector

    def get_geo_bbox(self, is_ray=False, ray_list=None):
        try:
            # get FoV
            FOVw, FOVh = self.calculate_fov_dimensions()
            # print('FOVw, FOVh:', FOVw, FOVh)
            if not is_ray:

                rotated_vectors = self.getBoundingPolygon(FOVw, FOVh)

            else:
                rotated_vectors = self.getBoundingPolygon(FOVw, FOVh)
                # print([[r.x, r.y, r.z]for r in rotated_vectors])
                # print('here')
                rays = [self.get_ray_direction2(int(px), int(py),
                                                self.K_inv, sx, sy) for px, py, sx, sy in ray_list]
                # print(rays)
                # Rotate rays according to camera orientation
                rotated_vectors = self.rotateRays(rays)
                # print([[r.x, r.y, r.z] for r in rotated_vectors])

                # rays = [self.get_ray_direction(int(px), int(py),
                #                                self.w, self.h,
                #                                FOVw, FOVh, sx, sy) for px, py, sx, sy in ray_list]
                # # Rotate rays according to camera orientation
                # rotated_vectors = self.rotateRays(rays)
                # print([[r.x, r.y, r.z]for r in rotated_vectors])

                # quit()

            utmx, utmy, zone_number, zone_letter = gps_to_utm(latitude, longitude)
            # print(utmx, utmy, zone_number, zone_letter)

            new_altitude = None
            # Determine new altitude based on different data sources
            if config.dtm_path:
                new_altitude = get_altitude_at_point(latitude, longitude)
            if config.global_elevation:
                new_altitude = get_altitude_from_open(latitude, longitude)
            if config.rel_altitude == 0.0:
                new_altitude = config.abso_altitude
            if new_altitude and abs(new_altitude - config.rel_altitude) > 20:
                new_altitude = config.rel_altitude
            if new_altitude is None:  # and not config.dtm_path or not config.global_elevation is False or config.rtk:
                new_altitude = config.rel_altitude
                if config.global_elevation is True or config.dtm_path:
                    print(
                        f"Failed to get elevation for {config.im_file_name}, using drone altitude.")

            corrected_altitude = self._atmospheric_refraction_correction(new_altitude)

            elevation_bbox = HighAccuracyFOVCalculator.getRayGroundIntersections(rotated_vectors,
                                                                                 Vector(0, 0,
                                                                                        float(corrected_altitude)))

            translated_bbox = find_geodetic_intersections(elevation_bbox, longitude, latitude)
            # drone_distance_to_polygon_center(translated_bbox, (utmx, utmy), corrected_altitude)

            new_translated_bbox = translated_bbox
            if config.dtm_path:
                # altitudes = [get_altitude_at_point(*box[:2]) for box in new_translated_bbox]
                trans_utmbox = [utm_to_latlon(box[0], box[1], zone_number, zone_letter) for box in new_translated_bbox]
                altitudes = [get_altitude_at_point(*box[:2]) for box in trans_utmbox]
                if None in altitudes:
                    print(
                        f"Failed to get elevation for image {config.im_file_name}. See log for details.")
                    return translate_to_wgs84(new_translated_bbox, longitude, latitude)
                if len(new_translated_bbox) == 4:
                    # Calculate the ratios of distances to check the 5 times condition
                    distances = [sqrt((new_translated_bbox[(i + 1) % len(new_translated_bbox)][0] - box[0]) ** 2 +
                                      (new_translated_bbox[(i + 1) % len(new_translated_bbox)][1] - box[1]) ** 2)
                                 for i, box in enumerate(new_translated_bbox)]
                    for dist in distances:
                        if any(other_dist * 6 < dist for other_dist in distances if other_dist != dist):
                            print(
                                f"One side of the polygon for {config.im_file_name} "
                                f"is at least 5 times longer than another.")
                            return translate_to_wgs84(new_translated_bbox, longitude, latitude)

            if config.global_elevation is True:
                trans_utmbox = [utm_to_latlon(box[0], box[1], zone_number, zone_letter) for box in new_translated_bbox]
                altitudes = [get_altitude_from_open(*box[:2]) for box in trans_utmbox]
                if None in altitudes:
                    print(f"Failed to get elevation at point for {config.im_file_name}.")
                    return translate_to_wgs84(new_translated_bbox, longitude, latitude)
                if len(new_translated_bbox) == 4:
                    # Calculate the ratios of distances to check the 5 times condition
                    distances = [sqrt((new_translated_bbox[(i + 1) % len(new_translated_bbox)][0] - box[0]) ** 2 +
                                      (new_translated_bbox[(i + 1) % len(new_translated_bbox)][1] - box[1]) ** 2)
                                 for i, box in enumerate(new_translated_bbox)]
                    for dist in distances:
                        if any(other_dist * 5 < dist for other_dist in distances if other_dist != dist):
                            print(
                                f"One side of the polygon for {config.im_file_name} "
                                f"is at least 5 times longer than another.")
                            return translate_to_wgs84(new_translated_bbox, longitude, latitude)

            # If no special conditions are met, process normally
            return translate_to_wgs84(new_translated_bbox, longitude, latitude)

        except Exception as e:
            print(f"Error in get_fov_bbox: {e}")
            return None, None

    @staticmethod
    def _sensor_lens_correction(fov_width, fov_height):
        # Placeholder for sensor and lens correction
        corrected_fov_width = fov_width * mp.mpf(lens_FOVw)  # Example correction factor
        corrected_fov_height = fov_height * mp.mpf(lens_FOVh)
        return corrected_fov_width, corrected_fov_height

    @staticmethod
    def getRayGroundIntersections(rays, origin):
        """
        Calculates the intersection points of the given rays with the ground plane.

        Parameters:
            rays (list): A list of Vector objects representing the rays.
            origin (Vector): The origin point of the rays.

        Returns:
            list: A list of Vector objects representing the intersection points on the ground.
        """

        intersections = [HighAccuracyFOVCalculator.findRayGroundIntersection(ray, origin) for ray in rays if
                         HighAccuracyFOVCalculator.findRayGroundIntersection(ray, origin) is not None]

        return intersections

    @staticmethod
    def findRayGroundIntersection(ray, origin):
        """
        Finds the intersection point of a single ray with the ground plane.

        Parameters:
            ray (Vector): The ray vector.
            origin (Vector): The origin point of the ray.

        Returns:
            Vector: The intersection point with the ground, or None if the ray is parallel to the ground.
        """

        if ray.z == 0:  # Ray is parallel to ground
            return None

        # Calculate intersection parameter t
        t = -origin.z / ray.z
        return Vector(origin.x + ray.x * t, origin.y + ray.y * t, 0)

    @staticmethod
    def _atmospheric_refraction_correction(altitude):
        return altitude + (altitude * 0.0001)


def calculate_centroid(polygon_coords):
    """Calculate the centroid of a polygon given its vertices in UTM coordinates."""
    x_sum = 0
    y_sum = 0
    for (x, y) in polygon_coords:
        x_sum += x
        y_sum += y
    centroid = (x_sum / len(polygon_coords), y_sum / len(polygon_coords))
    return centroid


def distance_3d(point1, point2):
    """Calculate the 3D distance between two points in UTM coordinates."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)


def drone_distance_to_polygon_center(polygon_coords, drone_coords, drone_altitude):
    """
    Calculate the distance from a drone to the center of a polygon in UTM coordinates.
    
    Parameters:
    - polygon_coords: List of tuples, each representing the (x, y) UTM coordinates of a polygon's vertex.
    - drone_coords: Tuple representing the (x, y) UTM coordinates of the drone's location.
    - drone_altitude: Float representing the drone's altitude in meters above the ground.
    
    Returns:
    - Float: The distance from the drone to the centroid of the polygon.
    """
    # Calculate the centroid of the polygon
    centroid = calculate_centroid(polygon_coords)
    centroid_3d = (centroid[0], centroid[1], 0)
    drone_position_3d = (drone_coords[0], drone_coords[1], drone_altitude)
    # config.update_center_distance(distance_3d(centroid_3d, drone_position_3d))
    # Calculate and return the 3D distance
    return
