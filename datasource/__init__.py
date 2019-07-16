#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:35:57 2019

@author: drizl
"""

from . import gracedb
from . import datasource
from glue import gpstime
from astropy.time import Time
import time

def get_nowtime():
    t = time.time()
    return gpstime.GpsSecondsFromPyUTC(t)

def GPS2ISO(gps):
    scTime = Time(int(gps), format = 'gps')
    return scTime.iso


__all__ = ['gracedb', 'datasource', 'get_nowtime']