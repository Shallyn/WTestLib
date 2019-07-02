#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:46:47 2019

@author: drizl
"""

import numpy as np
from scipy.interpolate import interp1d
from .Utils import interp1d_complex, LOG, WARNING
import sys

#-----Constants-----#
k_B_SI = 1.3806505e-23 # J/K
h_SI = 6.62606896e-34 # J s
ly_SI = 9.4605284e15 # 1 ly to m
AU_SI = 149597870700 # 1 AU to m
pc_SI = 3.08567758e16 # 1 Parsec to m
c_SI = 299792458 # Speed Of Light [m s^-1]
M_Sun_SI = 1.98892e30 # Mass of Sun [kg]
R_Sun_SI = 6.96342e8 # Radius of Sun [m]
alp_GP_SI = 192.85948 * np.pi / 180 # Direction Of Galactic Polar
det_GP_SI = 27.12825 * np.pi / 180 # Direction Of Galactic Polar
l_CP_SI = 122.932
G_SI = 6.672e-11 # Gravitational Constant [m^3 s^-2 kg^-1]
h_0_SI = 0.678 # Hubble Constant

#---------Comm--------#
def dim_t(M):
    return c_SI**3 / ( M * M_Sun_SI * G_SI)

def dim_h(r, M):
    return r * pc_SI * 1e6 * c_SI**2 / ( M * M_Sun_SI * G_SI )

def loaddata(filename, idx_cut = 0):
    data = np.loadtxt(filename)
    time = data[:,0]
    hp = data[:,1]
    hc = data[:,2]
    if idx_cut == 0:
        return [time, hp, hc]
    lth = len(time)
    if idx_cut > 1:
        idx_cut = min(int(lth/2),int(idx_cut))
    else:
        if 0 < idx_cut < 1:
            idx_cut = int( min(idx_cut, 0.5) * lth)
    t = time[idx_cut:]
    h_real = hp[idx_cut:]
    h_imag = hc[idx_cut:]
    return t, h_real, h_imag


#---------CLASS--------#
class h22base(object):
    def __init__(self, time, hreal, himag, srate, verbose = False):
        time = np.asarray(time)
        self._time = time - time[0]
        self._h22 = np.asarray(hreal) + 1.j * np.asarray(himag)
        if verbose:
            sys.stderr.write(f'{LOG}:Resampling...\n')
        self.resample(srate)
        if verbose:
            sys.stderr.write(f'{LOG}:Resampling...Done\n')
        self._verbose = verbose
            
    @property
    def time(self):
        return self._time
    
    @property
    def h22(self):
        return self._h22
    
    @property
    def real(self):
        return self._h22.real
    
    @property
    def imag(self):
        return self._h22.imag
    
    @property
    def amp(self):
        return np.abs(self._h22)
    
    @property
    def phase(self):
        return np.abs(np.unwrap(np.angle(self._h22)))
    
    @property
    def frequency(self):
        return np.gradient(self.phase) / np.gradient(self._time)
    
    @property
    def argpeak(self):
        return np.argmax(self.amp)
    
    @property
    def tpeak(self):
        return self.time[self.argpeak]
    
    @property
    def duration(self):
        return self.time[-1] - self.time[0]
    
    @property
    def h22f(self):
        return np.fft.fft(self._h22)
    
    @property
    def fftfreq(self):
        return np.fft.fftfreq(len(self._h22), 1./self._srate)

    @property
    def itp_h22(self):
        return interp1d_complex(self._time, self._h22)
    
    @property
    def srate(self):
        return self._srate

    def resample(self, srate):
        self._srate = srate
        new_time = np.arange(self._time[0], self._time[-1], 1./self._srate)
        self._h22 = self.itp_h22(new_time)
        self._time = new_time
        
    def pad(self, pad_width, mode, **kwargs):
        self._h22 = np.pad(self._h22, pad_width, mode, **kwargs)
        self._time = np.arange(self._time[0], self._h22.size / self._srate, 1./self._srate)

    def apply(self, tc, phic):
        self._time -= tc
        self._h22 *= np.exp(-1.j*phic)
        
    def saveh22(self, fname, **kwargs):
        data = np.stack([self.time, self.real, self.imag], axis = 1)
        np.savetxt(fname, data, **kwargs)

    def copy(self):
        return h22base(self._time.copy(), self._h22.copy().real, self._h22.copy().imag, self._srate, self._verbose)
        
    def __str__(self):
        return '{}'.format(self._h22)
    
    def __len__(self):
        return len(self._h22)
    
    def __repr__(self):
        return self.__str__()
    
    def __format__(self):
        return self.__str__()
        
    def __iter__(self):
        for x in self._h22:
            yield x
    
    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, np.integer):
            return self._h22[key]
        return self._getslice(key)

    def __setitem__(self, key, value):
        self.s[key] = value
    
    def _getslice(self, index):
        if index.start is not None and index.start < 0:
            raise ValueError(('Negative start index ({}) is not supported').format(index.start))        
        return h22base(self._time[index], self.real[index], self.imag[index], self._srate, self._verbose)



