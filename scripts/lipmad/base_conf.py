#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Modules """
from __future__ import annotations

import logging
from colorlog import ColoredFormatter

# -----------------------------------------------------------------------------


# Logging and console output
LOG_LEVEL = logging.INFO
LOGFORMAT = '[%(log_color)s%(levelname)8s%(reset)s] %(log_color)s%(message)s%(reset)s'
FORMATTER = ColoredFormatter(LOGFORMAT)


# Logging and console output
logging.root.handlers = []
log = logging.getLogger()
log.setLevel(LOG_LEVEL)
stream = logging.StreamHandler()
stream.setFormatter(FORMATTER)
log.addHandler(stream)
log_level = log.level


# Colored console output
class BCOLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[32m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# List of supported image formats
SUPPORTED_IMAGE_FORMATS = ['DNG', 'JPG',
                           # 'FITS'
                           ]

# Possible image format file extensions
DEF_FILE_EXT = {'JPG': 'jpg|JPG',
                # 'FITS': 'fits|FITS|fit|FIT|Fits|fts|FTS',
                'DNG': 'dng|DNG'}

BIT_DEPTH_CONVERSION = {255: 8, 511: 9, 1023: 10, 2047: 11, 4095: 12, 8191: 13,
                        16383: 14, 32767: 15, 65535: 16}

# Supported colors
SUPPORTED_COLORS = ['r', 'g', 'b', 'g2']

DEF_COLORS = {'r': ('R', '_red'),
              'g': ('G', '_green'),
              'g2': ('G2', '_green2'),
              'b': ('B', '_blue'),
              }

HDR_KEYS_ORDERED = ['CTIME', 'DATE-OBS', 'OBJECT', 'INSTRUME',
                    'CAMERA', 'MAKE', 'DETNAM', 'DETSER',
                    'EXPTIME', 'ISO', 'APERTUR', 'FILTER',
                    'FNUMBER', 'SHUTTER', 'FOCAL',
                    # 'LATOBS', 'LONGOBS',
                    # 'ALTOBS', 'ALTREL', 'ALTTYP', 'ALTREF', 'MAPDATE',
                    # 'YAW1', 'PITCH1', 'ROLL1', 'YAW2', 'PITCH2', 'ROLL2',
                    # 'SPEEDX', 'SPEEDY', 'SPEEDZ',
                    'FILENAME', 'INPUTFMT', 'DATE']
