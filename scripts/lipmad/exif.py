#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import piexif
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------

""" Parameter used in the script """

_log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


class Exif(object):
    """Handle exif data of an image file"""

    def __init__(self, image_file: str, log=_log):
        """ Constructor with default values """

        self.log = log
        self.image_file = image_file
        self._image_dict = {}

        self.exif_dict = piexif.load(image_file)
        self.xmp = self.get_xmp()

        self.img_idf = self.exif_dict['0th']
        self.exif_idf = self.exif_dict['Exif']
        self.gps_idf = self.exif_dict['GPS']

        self._camera_info = (None for _ in range(4))
        self._colour_info = (None for _ in range(5))
        self._pose_info = (None for _ in range(14))
        self._exposure_info = (None for _ in range(7))

    @property
    def all_info(self):
        return self._image_dict

    @property
    def camera_info(self):
        return self._camera_info

    @property
    def colour_info(self):
        return self._colour_info

    @property
    def pose_info(self):
        return self._pose_info

    @property
    def exposure_info(self):
        return self._exposure_info

    @staticmethod
    def dms_to_decimal(degrees: tuple,
                       minutes: tuple,
                       seconds: tuple,
                       sign: str = ' ') -> float:
        """Convert degrees, minutes, seconds into decimal degrees."""
        return (-1 if sign[0] in 'SWsw' else 1) * (
                float(degrees[0] / degrees[1])
                + float(minutes[0] / minutes[1]) / 60.
                + float(seconds[0] / seconds[1]) / 3600.)

    def get_all_info(self):
        """ Collect all info at once"""
        self.get_camera_info()
        self.get_exposure_info()
        self.get_pose_info()

    def get_camera_info(self):
        """ Extract camera info """

        lens_model = None
        camera, make, model = ("" for _ in range(3))
        if piexif.ImageIFD.Make in self.img_idf:
            make = self.img_idf[piexif.ImageIFD.Make].decode('utf-8').rstrip('\x00')
            camera = make
        if piexif.ImageIFD.Model in self.img_idf:
            model = self.img_idf[piexif.ImageIFD.Model].decode('utf-8').rstrip('\x00')
            camera += '_' + model
        if piexif.ExifIFD.LensModel in self.exif_idf:
            lens_model = self.exif_idf[piexif.ExifIFD.LensModel].decode('utf-8').rstrip('\x00')
            camera += '_' + lens_model

        focal_len_mm = 12.29
        if piexif.ExifIFD.FocalLength in self.exif_idf:
            focal_len_mm_ratio = self.exif_idf[piexif.ExifIFD.FocalLength]
            focal_len_mm = focal_len_mm_ratio[0] / focal_len_mm_ratio[1]

        self._image_dict['INSTRUME'] = (camera, None)
        self._image_dict['CAMERA'] = (camera, 'camera model')
        self._image_dict['MAKE'] = (make, 'camera manufacturer')
        self._image_dict['DETNAM'] = (model, 'name of the detector')
        self._image_dict['FOCAL'] = (focal_len_mm, 'focal length in mm')

        self._camera_info = (camera, make, model, lens_model, focal_len_mm)

    def get_pose_info(self):
        """"""
        xmp = self.xmp

        if 'drone-dji:GpsLatitude' in xmp:
            lat_deg = float(xmp['drone-dji:GpsLatitude'])
        else:
            elat = self.gps_idf[piexif.GPSIFD.GPSLatitude]
            lat_deg = self.dms_to_decimal(elat[0], elat[1], elat[2],
                                          self.gps_idf[piexif.GPSIFD.GPSLatitudeRef].decode('utf-8'))
        self._image_dict['LATOBS'] = (lat_deg, 'GPS latitude in deg (N>0)')

        if 'drone-dji:GpsLongitude' in xmp:
            lon_deg = float(xmp['drone-dji:GpsLongitude'])
        else:
            elon = self.gps_idf[piexif.GPSIFD.GPSLongitude]
            lon_deg = self.dms_to_decimal(elon[0], elon[1], elon[2],
                                          self.gps_idf[piexif.GPSIFD.GPSLongitudeRef].decode('utf-8'))
        self._image_dict['LONGOBS'] = (lon_deg, 'GPS longitude in deg (E>0)')

        if 'drone-dji:AbsoluteAltitude' in xmp:
            alt_m_abs = float(xmp['drone-dji:AbsoluteAltitude'])
            if alt_m_abs < 0:
                self.log.warning("Image meta data is reporting negative absolute altitude!")
        else:
            ealt = self.gps_idf[piexif.GPSIFD.GPSAltitude]
            alt_m_abs = ealt[0] / ealt[1]
        self._image_dict['ALTOBS'] = (alt_m_abs, 'absolute GPS altitude in m')

        if 'drone-dji:RelativeAltitude' in xmp:
            alt_m_rel = float(xmp['drone-dji:RelativeAltitude'])
            if alt_m_rel < 0:
                self.log.warning("Image meta data is reporting negative relative altitude!")
        else:
            ealt = self.gps_idf[piexif.GPSIFD.GPSAltitude]
            alt_m_rel = ealt[0] / ealt[1]
        self._image_dict['ALTREL'] = (alt_m_rel, 'relative GPS altitude in m')

        alt_type = None
        if 'drone-dji:AltitudeType' in xmp:
            alt_type = xmp['drone-dji:AltitudeType']
        self._image_dict['ALTTYP'] = (alt_type, None)

        alt_ref = None
        if piexif.GPSIFD.GPSAltitudeRef in self.gps_idf:
            alt_ref = self.gps_idf[piexif.GPSIFD.GPSAltitudeRef]
        self._image_dict['ALTREF'] = (alt_ref, '0=Above SL, 1=Below SL')

        map_date = None
        if piexif.GPSIFD.GPSMapDatum in self.gps_idf:
            map_date = self.gps_idf[piexif.GPSIFD.GPSMapDatum].decode('utf-8').rstrip('\x00')
        self._image_dict['MAPDATE'] = (map_date, 'GPS map datum')

        # drone yaw, roll and pitch
        if 'drone-dji:FlightYawDegree' in xmp:
            yaw_deg_drone = float(xmp['drone-dji:FlightYawDegree'])
            # while yaw_deg_drone < 0:
            #     yaw_deg_drone += 360
        elif 'Flight:Yaw' in xmp:
            yaw_deg_drone = float(xmp['Flight:Yaw'])
            # while yaw_deg_drone < 0:
            #     yaw_deg_drone += 360
        else:
            yaw_deg_drone = None
        self._image_dict['YAW2'] = (yaw_deg_drone, 'drone yaw in degree')

        if 'drone-dji:FlightPitchDegree' in xmp:
            pitch_deg_drone = float(xmp['drone-dji:FlightPitchDegree'])
        elif 'Flight:Pitch' in xmp:
            pitch_deg_drone = float(xmp['Flight:Pitch'])
        else:
            pitch_deg_drone = None
        self._image_dict['PITCH2'] = (pitch_deg_drone, 'drone pitch in degree')

        if 'drone-dji:FlightRollDegree' in xmp:
            roll_deg_drone = float(xmp['drone-dji:FlightRollDegree'])
        elif 'Flight:Roll' in xmp:
            roll_deg_drone = float(xmp['Flight:Roll'])
        else:
            roll_deg_drone = None
        self._image_dict['ROLL2'] = (roll_deg_drone, 'drone roll in degree')

        # camera/gimbal yaw, roll, pitch
        if 'drone-dji:GimbalYawDegree' in xmp:
            yaw_deg = float(xmp['drone-dji:GimbalYawDegree'])
            # while yaw_deg < 0:
            #     yaw_deg += 360
        elif 'Camera:Yaw' in xmp:
            yaw_deg = float(xmp['Camera:Yaw'])
            # while yaw_deg < 0:
            #     yaw_deg += 360
        else:
            yaw_deg = None
        self._image_dict['YAW1'] = (yaw_deg, 'camera yaw in degree')

        if 'drone-dji:GimbalPitchDegree' in xmp:
            pitch_deg = float(xmp['drone-dji:GimbalPitchDegree'])
        elif 'Camera:Pitch' in xmp:
            pitch_deg = float(xmp['Camera:Pitch'])
        else:
            pitch_deg = None
        self._image_dict['PITCH1'] = (pitch_deg, 'camera pitch in degree')

        if 'drone-dji:GimbalRollDegree' in xmp:
            roll_deg = float(xmp['drone-dji:GimbalRollDegree'])
        elif 'Camera:Roll' in xmp:
            roll_deg = float(xmp['Camera:Roll'])
        else:
            roll_deg = None
        self._image_dict['ROLL1'] = (roll_deg, 'camera roll in degree')

        if 'drone-dji:FlightXSpeed' in xmp:
            x_speed = float(xmp['drone-dji:FlightXSpeed'])
        elif 'Flight:XSpeed' in xmp:
            x_speed = float(xmp['Flight:XSpeed'])
        else:
            x_speed = None
        self._image_dict['SPEEDX'] = (x_speed, 'flight speed in X')

        if 'drone-dji:FlightYSpeed' in xmp:
            y_speed = float(xmp['drone-dji:FlightYSpeed'])
        elif 'Flight:YSpeed' in xmp:
            y_speed = float(xmp['Flight:YSpeed'])
        else:
            y_speed = None
        self._image_dict['SPEEDY'] = (y_speed, 'flight speed in Y')

        if 'drone-dji:FlightZSpeed' in xmp:
            z_speed = float(xmp['drone-dji:FlightZSpeed'])
        elif 'Flight:ZSpeed' in xmp:
            z_speed = float(xmp['Flight:ZSpeed'])
        else:
            z_speed = None
        self._image_dict['SPEEDZ'] = (z_speed, 'flight speed in Z')

        # print(lon_deg, lat_deg, alt_m_abs, alt_m_rel,
        #       yaw_deg, pitch_deg, roll_deg, yaw_deg_drone, pitch_deg_drone, roll_deg_drone,
        #       x_speed, y_speed, z_speed)
        self._pose_info = (lon_deg, lat_deg,
                           alt_m_abs, alt_m_rel,
                           yaw_deg, pitch_deg, roll_deg,
                           yaw_deg_drone, pitch_deg_drone, roll_deg_drone,
                           x_speed, y_speed, z_speed)

    def get_colour_info(self):
        """ Get colour info """
        wb_as_shot, calibration_illuminant1, calibration_illuminant2, M_color_matrix_1, M_color_matrix_2 = \
            (None for _ in range(5))

        if piexif.ImageIFD.CalibrationIlluminant1 in self.img_idf:
            calibration_illuminant1 = self.img_idf[piexif.ImageIFD.CalibrationIlluminant1]
        if piexif.ImageIFD.CalibrationIlluminant2 in self.img_idf:
            calibration_illuminant2 = self.img_idf[piexif.ImageIFD.CalibrationIlluminant2]

        if piexif.ImageIFD.ColorMatrix1 in self.img_idf:
            color_matrix_1_ratios = self.img_idf[piexif.ImageIFD.ColorMatrix1]
            numbers = [cm[0] / cm[1] for cm in color_matrix_1_ratios]
            M_color_matrix_1 = np.array([numbers[i:i + 3] for i in range(0, len(numbers), 3)], dtype=np.double)

        if piexif.ImageIFD.ColorMatrix2 in self.img_idf:
            color_matrix_2_ratios = self.img_idf[piexif.ImageIFD.ColorMatrix2]

            numbers = [cm[0] / cm[1] for cm in color_matrix_2_ratios]
            M_color_matrix_2 = np.array([numbers[i:i + 3] for i in range(0, len(numbers), 3)], dtype=np.double)

        if piexif.ImageIFD.AsShotNeutral in self.img_idf:
            wb_as_shot_ratios = self.img_idf[piexif.ImageIFD.AsShotNeutral]
            wb_as_shot = np.array([wb[0] / wb[1] for wb in wb_as_shot_ratios])

        self._colour_info = (wb_as_shot,
                             calibration_illuminant1, calibration_illuminant2,
                             M_color_matrix_1, M_color_matrix_2)

    def get_exposure_info(self):
        """ Get exposure info """

        iso, aperture, f_number, exptime, shutter_speed, unixtime, obsdate, obsdate_iso = \
            (None for _ in range(8))

        if piexif.ExifIFD.ISOSpeedRatings in self.exif_idf:
            iso = float(self.exif_idf[piexif.ExifIFD.ISOSpeedRatings])
        self._image_dict['ISO'] = ("%d" % iso, 'ISO speed')

        if piexif.ExifIFD.ApertureValue in self.exif_idf:
            aperture_ratio = self.exif_idf[piexif.ExifIFD.ApertureValue]
            aperture = aperture_ratio[0] / aperture_ratio[1]
        self._image_dict['APERTUR'] = (aperture, 'lens aperture value')

        if piexif.ExifIFD.FNumber in self.exif_idf:
            f_number_ratio = self.exif_idf[piexif.ExifIFD.FNumber]
            f_number = f_number_ratio[0] / f_number_ratio[1]

        self._image_dict['FNUMBER'] = ("%.1f" % f_number,
                                       'f-stop; f/FNUMBER=sqrt(2)^APERTUR')

        if piexif.ExifIFD.ShutterSpeedValue in self.exif_idf:
            shutter_ratio = self.exif_idf[piexif.ExifIFD.ShutterSpeedValue]
            shutter_speed = shutter_ratio[0] / shutter_ratio[1]
            # print(1/2**shutter_speed)

        self._image_dict['SHUTTER'] = (shutter_speed, 'shutter speed. EXPTIME=1/(2^SHUTTER)')

        if piexif.ExifIFD.ExposureTime in self.exif_idf:
            exposure_ratio = self.exif_idf[piexif.ExifIFD.ExposureTime]
            exptime = exposure_ratio[0] / exposure_ratio[1]
        self._image_dict['EXPTIME'] = (exptime, 'exposure time in s')

        if piexif.ExifIFD.DateTimeOriginal in self.exif_idf:
            dateStr = self.exif_idf[piexif.ExifIFD.DateTimeOriginal].decode('utf-8')
            # print(dateStr)
            if piexif.ExifIFD.SubSecTimeOriginal in self.exif_idf:
                sub_sec = str(self.exif_idf[piexif.ExifIFD.SubSecTimeOriginal])
            else:
                sub_sec = '0'
            value = dateStr + '.' + str(sub_sec)
            frmt = "%Y:%m:%d %H:%M:%S.%f"
            obsdate = pd.to_datetime(value, format=frmt, utc=False)
            obsdate_iso = obsdate.isoformat(sep='T', timespec='milliseconds')
            unixtime = float(obsdate.strftime('%s'))
            frmt = "%Y:%m:%d %H:%M:%S"
            obsdate = pd.to_datetime(dateStr, format=frmt, utc=False)
            obsdate = obsdate.isoformat(sep=' ', timespec='seconds')
            self._image_dict['DATE-STR'] = (dateStr, 'date of the observation original')

        self._image_dict['DATE-OBS'] = (obsdate_iso, 'date of the observation')
        self._image_dict['CTIME'] = (unixtime, 'exposure start (seconds since 1.1.1970)')
        # print(unixtime, obsdate, obsdate_iso)
        # print(iso, aperture, exptime, shutter_speed)
        self._exposure_info = (obsdate, obsdate_iso, unixtime,
                               iso, aperture, f_number, exptime, shutter_speed)

    def get_xmp(self) -> dict:
        """ Extended xmp tags (hack) """

        fd = open(self.image_file, "rb")
        d = str(fd.read())
        xmp_start = d.find('<x:xmpmeta')
        xmp_end = d.find('</x:xmpmeta')
        xmp_str = d[xmp_start:xmp_end + 12]

        lines = xmp_str.split("\\n")

        xmp = {}
        for line in lines:
            line = line.rstrip().lstrip()
            if line[0] == "<":
                continue
            token, val = line.split("=")
            val = val.strip('"')
            xmp[token] = val

        return xmp

# -----------------------------------------------------------------------------
