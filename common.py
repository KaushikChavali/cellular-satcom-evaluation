#!/usr/bin/env python3

# cellular-satcom-evaluation : Evaluation sciprts for the Multipath
# Cellular and SATCOM Emulation Testbed.
#
# Copyright (C) 2023 Kaushik Chavali
# 
# This file is part of the cellular-satcom-evaluation.
#
# cellular-satcom-evaluation is free software: you can redistribute it 
# and/or modify it under the terms of the GNU General Public License 
# as published by the Free Software Foundation, either version 3 of 
# the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import sys
from enum import Enum

RAW_DATA_DIR = 'parsed'
GRAPH_DIR = 'graphs'
DATA_DIR = 'data'
CLIENT_IP_LTE = '172.20.5.16'
CLIENT_IP_SAT = '192.168.26.34'
SERVER_IP = '10.30.4.18'
SERVER_IP_MP = "172.20.5.116"
GSTREAMER_PORT = "4242"
IPERF_PORT = "5201"

TYPE_FILE = '.type'
AUTO_DETECT_FILE = '.auto-detect'

logger = logging.getLogger(__name__)


def setup_logger():
    global logger

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s %(processName)-12s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)


class Mode(Enum):
    PARSE = 1
    ANALYZE = 2
    ALL = 255

    def do_parse(self):
        return self == Mode.PARSE or self == Mode.ALL

    def do_analyze(self):
        return self == Mode.ANALYZE or self == Mode.ALL


class MeasureType(Enum):
    NETEM = 1
    OPENSAND = 2

    @classmethod
    def from_name(cls, name):
        for t in cls:
            if t.name == name:
                return t
        return None
