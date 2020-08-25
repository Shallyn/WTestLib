#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:46:47 2019

@author: drizl
"""

import numpy as np
from scipy.interpolate import interp1d
from .Utils import interp1d_complex, LOG, WARNING
import matplotlib.pyplot as plt
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

def get_Mtotal(fdimless, fmin):
    return fdimless * pow(c_SI, 3) / (G_SI * M_Sun_SI * fmin)

def get_fmin(fdimless, Mtotal):
    return fdimless * pow(c_SI, 3) / (G_SI * M_Sun_SI * Mtotal)

def get_fini_dimless(fmin, Mtotal):
    return fmin * G_SI * Mtotal * M_Sun_SI / pow(c_SI, 3)

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

    


class ModeBase(object):
    def __init__(self, time, hreal, himag):
        time = np.asarray(time)
        self._time = time - time[0]
        self._mode = np.asarray(hreal) + 1.j * np.asarray(himag)

    @property
    def length(self):
        return len(self._time)
            
    @property
    def time(self):
        return self._time
    
    @property
    def dot(self):
        return np.gradient(self._mode) / np.gradient(self.time)
    
    @property
    def value(self):
        return self._mode
    
    @property
    def real(self):
        return self._mode.real
    
    @property
    def imag(self):
        return self._mode.imag
    
    @property
    def amp(self):
        return np.abs(self._mode)
    
    @property
    def phase(self):
        return np.abs(np.unwrap(np.angle(self._mode)))
    
    @property
    def phaseFrom0(self):
        phase = self.phase
        return phase - phase[0]

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
    def interpolate(self):
        return interp1d_complex(self._time, self._mode)
    
    @property
    def conjvalue(self):
        return self._mode.conjugate()
    
    def conjugate(self):
        return ModeBase(self.time, self.real, -self.imag)

    def copy(self):
        return ModeBase(self._time.copy(), self._mode.copy().real, self._mode.copy().imag)
        
    def __str__(self):
        return '{}'.format(self._mode)
    
    def __len__(self):
        return len(self._mode)
    
    def __repr__(self):
        return self.__str__()
            
    def __iter__(self):
        for x in self._mode:
            yield x
    
    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, np.integer):
            return self._mode[key]
        return self._getslice(key)

    def __setitem__(self, key, value):
        self.s[key] = value
    
    def _getslice(self, index):
        if index.start is not None and index.start < 0:
            raise ValueError(('Negative start index ({}) is not supported').format(index.start))        
        return h22base(self._time[index], self.real[index], self.imag[index], self._srate, self._verbose)
    
    def __del__(self):
        del self._time
        del self._mode

    def dump(self, fname, **kwargs):
        data = np.stack([self.time, self.real, self.imag], axis = 1)
        np.savetxt(fname, data, **kwargs)
        return

class h22base(ModeBase):
    def __init__(self, time, hreal, himag, srate, verbose = False):
        super(h22base, self).__init__(time, hreal, himag)
        if verbose:
            sys.stderr.write(f'{LOG}:Resampling...\n')
        self.resample(srate)
        if verbose:
            sys.stderr.write(f'{LOG}:Resampling...Done\n')
        self._verbose = verbose

    @property
    def h22f(self):
        return np.fft.fft(self._mode)
    
    @property
    def fftfreq(self):
        return np.fft.fftfreq(len(self._mode), 1./self._srate)

    def apply(self, tc, phic):
        self._time -= tc
        self._mode *= np.exp(1.j*phic)
    
    @property
    def srate(self):
        return self._srate

    def resample(self, srate):
        self._srate = srate
        new_time = np.arange(self._time[0], self._time[-1], 1./self._srate)
        self._mode = self.interpolate(new_time)
        self._time = new_time

    def pad(self, pad_width, mode, **kwargs):
        self._mode = np.pad(self._mode, pad_width, mode, **kwargs)
        self._time = np.arange(self._time[0], self._mode.size / self._srate, 1./self._srate)
        
    def saveh22(self, fname, **kwargs):
        data = np.stack([self.time, self.real, self.imag], axis = 1)
        np.savetxt(fname, data, **kwargs)

    def plot(self, fname = 'save.png'):
        plt.figure(figsize = (14,5))
        plt.title('waveform')
        plt.plot(self.time, self.real)
        plt.savefig(fname, dpi = 200)
        plt.close()

def h22_alignment(wfA, wfB):
    fs_A = wfA.srate
    fs_B = wfB.srate
    fs = fs_A
    if fs_A != fs_B:
        if fs_A > fs_B:
            wfA.resample(fs_B)
            fs = fs_B
        else:
            wfB.resample(fs_A)
            fs = fs_A
    peak_A = wfA.argpeak
    peak_B = wfB.argpeak           

    if peak_A > peak_B:
        idx_A = peak_A - peak_B
        idx_B = 0
        tmove = -idx_A / fs
    else:
        idx_A = 0
        idx_B = peak_B - peak_A
        tmove = idx_B / fs

    wfA = wfA[idx_A:]
    wfB = wfB[idx_B:]
    len_A = len(wfA)
    len_B = len(wfB)
    peak_A = wfA.argpeak
    peak_B = wfB.argpeak           
    tail_A = len_A - peak_A
    tail_B = len_B - peak_B

    # Check data shape 
    if tail_A > tail_B:
        lpad = tail_A - tail_B
        wfB.pad((0,lpad), 'constant')
    else:
        lpad = tail_B - tail_A
        wfA.pad((0,lpad), 'constant')
        
    return wfA, wfB, tmove

