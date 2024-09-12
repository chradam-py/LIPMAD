#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Modules """
from __future__ import annotations

import logging
import os
import socket
from datetime import datetime
from colorlog import ColoredFormatter

LOG_LEVEL = logging.INFO
LOGFORMAT = '[%(log_color)s%(levelname)8s%(reset)s] %(log_color)s%(message)s%(reset)s'


class Logger:
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    def __init__(self):
        self.logfile = None
        self.logger = logging.getLogger()
        self.formatter = ColoredFormatter(LOGFORMAT)
        self.setup_logger()

    def setup_logger(self):

        self.logger.setLevel(LOG_LEVEL)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(stream_handler)

    def init_logfile(self, log_path):
        self.logfile = os.path.join(log_path, "messages-" + socket.gethostname())

        if not os.path.exists(os.path.dirname(self.logfile)):
            os.makedirs(os.path.dirname(self.logfile))

    def log(self, message, level="info", fancy=False):
        log_level = self.LEVELS.get(level.lower(), logging.INFO)
        # Create log record
        log_record = self.logger.makeRecord(self.logger.name, log_level,
                                            None, None, message,
                                            None, None)
        formatted_msg = self.formatter.format(log_record)

        # Fancy formatting
        if fancy:
            formatted_msg = "\n############################################################################\n" \
                            "### " + formatted_msg + \
                            "\n############################################################################\n"

        # Write to file and log using standard logger
        with open(self.logfile, "a") as f:
            f.write(formatted_msg + "\n")
        self.logger.handle(log_record)
