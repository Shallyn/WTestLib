#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:24:30 2019

@author: drizl
"""

import numpy as np
import time
import astropy.time as at

VERSION = 'old'

EPOCH_J_2000_0_JD = 2451545.0
XLAL_EPOCH_GPS_TAI_UTC = 19
XLAL_EPOCH_UNIX_GPS = 315964800

class leaptime(object):
    def __init__(self, setting):
        jd, gpssec, taiutc = setting
        self.jd = jd
        self.gpssec = gpssec
        self.taiutc = taiutc

leaps = [
    leaptime([2444239.5,    -43200, 19]), # 1980-Jan-01
    leaptime([2444786.5,  46828800, 20]), # 1981-Jul-01 
    leaptime([2445151.5,  78364801, 21]), # 1982-Jul-01 
    leaptime([2445516.5, 109900802, 22]), # 1983-Jul-01 
    leaptime([2446247.5, 173059203, 23]), # 1985-Jul-01 
    leaptime([2447161.5, 252028804, 24]), # 1988-Jan-01 
    leaptime([2447892.5, 315187205, 25]), # 1990-Jan-01 
    leaptime([2448257.5, 346723206, 26]), # 1991-Jan-01 
    leaptime([2448804.5, 393984007, 27]), # 1992-Jul-01 
    leaptime([2449169.5, 425520008, 28]), # 1993-Jul-01 
    leaptime([2449534.5, 457056009, 29]), # 1994-Jul-01 
    leaptime([2450083.5, 504489610, 30]), # 1996-Jan-01 
    leaptime([2450630.5, 551750411, 31]), # 1997-Jul-01 
    leaptime([2451179.5, 599184012, 32]), # 1999-Jan-01 
    leaptime([2453736.5, 820108813, 33]), # 2006-Jan-01 
    leaptime([2454832.5, 914803214, 34]), # 2009-Jan-01 
    leaptime([2456109.5, 1025136015, 35]), # 2012-Jul-01 
    leaptime([2457204.5, 1119744016, 36]), # 2015-Jul-01 
    leaptime([2457754.5, 1167264017, 37]), # 2017-Jan-01 
]
numleaps = len(leaps)

def LeapSeconds(gpssec):
    if gpssec < leaps[0].gpssec:
        msg = 'Dont know leap seconds before GPS time {}'.format(gpssec)
        raise ValueError(msg)
        
    for leap in range(numleaps):
        if gpssec < leaps[leap].gpssec:
            break;
    return leaps[leap-1].taiutc;

def delta_tai_utc(gpssec):
    for leap in range(numleaps):
        if gpssec == leaps[leap].gpssec:
            return leaps[leap].taiutc - leaps[leap-1].taiutc
    return 0

def GPStoUTC_old(gpssec):
    leapsec = LeapSeconds( gpssec )
    unixsec  = gpssec - leapsec + XLAL_EPOCH_GPS_TAI_UTC
    unixsec += XLAL_EPOCH_UNIX_GPS
    utc = time.gmtime(unixsec)
    delta = delta_tai_utc( gpssec )
    if  delta  > 0:
        utc.tm_sec += 1
    return utc, 0

def GPStoUTC_new(gpssec):
    T = at.Time(gpssec, format = 'gps')
    unixsec = T.unix
    utc = time.gmtime(unixsec)
    return utc, T

def ConvertCivilTimeToJD_old(civil, T):
    sec_per_day = 60 * 60 * 24
    if civil.tm_year <= 0:
        raise ValueError('Wrong civil time')
    year = civil.tm_year
    month = civil.tm_mon
    day = civil.tm_mday
    sec   = civil.tm_sec + 60*(civil.tm_min + 60*(civil.tm_hour))
    jd = 367*year - 7*(year + (month + 9)/12)/4 + 275*month/9 + day + 1721014
    jd += sec/sec_per_day - 0.5
    return jd

def ConvertCivilTimeToJD_new(civil, T):
    return T.jd

if VERSION == 'new':
    GPStoUTC = GPStoUTC_new
    ConvertCivilTimeToJD = ConvertCivilTimeToJD_new
else:
    GPStoUTC = GPStoUTC_old
    ConvertCivilTimeToJD = ConvertCivilTimeToJD_old


def GreenwichMeanSiderealTime(gps):
    if isinstance(gps, list):
        gps_sec, gps_nano = gps
    else:
        gps_sec = int(gps)
        gps_nano = (gps - gps_sec) * 1e9
    utc, _ = GPStoUTC(gps_sec)
    julian_day = ConvertCivilTimeToJD(utc, _)
    t_hi = (julian_day - EPOCH_J_2000_0_JD) / 36525.0
    t_lo = gps_nano / (36525 * 86400) * 1e-9
    t = t_hi + t_lo
    sidereal_time = (-6.2e-6 * t + 0.093104) * t * t
    sidereal_time += 67310.54841
    sidereal_time += 8640184.81266 * t_lo + 3155760000 * t_lo
    sidereal_time += 8640184.812866 * t_hi + 3155760000 * t_hi
    return sidereal_time * np.pi / 43200.0

def GreenwichMeanSiderealTime_new(gps):
    if isinstance(gps, list):
        gps_sec, gps_nano = gps
    else:
        gps_sec = int(gps)
        gps_nano = (gps - gps_sec) * 1e9
    utc, _ = GPStoUTC(gps_sec)
    julian_day = ConvertCivilTimeToJD(utc, _)
    t_hi = (julian_day - EPOCH_J_2000_0_JD) 
    t_lo = gps_nano / (86400) * 1e-9
    t = t_hi + t_lo
    sidereal_time = (0.671262 + 1.0027379094 * t)
    return sidereal_time


