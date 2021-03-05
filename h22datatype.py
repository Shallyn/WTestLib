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
MRSUN_SI = 1.47662504e3 
MTSUN_SI = 4.92549095e-6 
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
        self._t0 = time[0]
        self._time = time - time[0]
        self._mode = np.asarray(hreal) + 1.j * np.asarray(himag)

    def apply_phic(self, phic):
        self._mode * np.exp(1.j*phic)

    @property
    def t0(self):
        return self._t0

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
        return np.unwrap(np.angle(self._mode))
    
    @property
    def phaseFrom0(self):
        phase = self.phase
        return np.abs(phase - phase[0])

    @property
    def frequency(self):
        return np.abs(np.gradient(self.phase) / np.gradient(self._time))
    
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
        self._mode[key] = value
    
    def _getslice(self, index):
        if index.start is not None and index.start < 0:
            raise ValueError(('Negative start index ({}) is not supported').format(index.start))        
        return ModeBase(self._time[index], self.real[index], self.imag[index])
    
    def __del__(self):
        del self._time
        del self._mode

    def dump(self, fname, **kwargs):
        data = np.stack([self.time, self.real, self.imag], axis = 1)
        np.savetxt(fname, data, **kwargs)
        return

    def pad(self, pad_width, mode, deltaT, **kwargs):
        self._mode = np.pad(self._mode, pad_width, mode, **kwargs)
        self._time = np.arange(self._time[0], self._mode.size * deltaT, deltaT)


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

    def copy(self):
        return h22base(self._time.copy(), self._mode.copy().real, self._mode.copy().imag, self._srate, self._verbose)

    def _getslice(self, index):
        if index.start is not None and index.start < 0:
            raise ValueError(('Negative start index ({}) is not supported').format(index.start))        
        return h22base(self._time[index], self.real[index], self.imag[index], self._srate, self._verbose)

    def plot(self, fname = 'save.png'):
        plt.figure(figsize = (14,5))
        plt.title('waveform')
        plt.plot(self.time, self.real)
        plt.savefig(fname, dpi = 200)
        plt.close()

def Mode_alignment(modeA, modeB, deltaT = None):
    tA = modeA.time
    tB = modeB.time
    dtA = tA[1] - tA[0]
    dtB = tB[1] - tB[0]
    if dtA != dtB:
        if deltaT is not None:
            dt_final = deltaT
        elif dtA < dtB:
            dt_final = dtB
        else:
            dt_final = dtA
        tA_new = np.arange(tA[0], tA[-1], dt_final)
        tB_new = np.arange(tB[0], tB[-1], dt_final)
        valA = modeA.interpolate(tA_new)
        valB = modeB.interpolate(tB_new)
    else:
        tA_new = tA
        tB_new = tB
        valA = modeA.value
        valB = modeB.value
        dt_final = dtA
    wfA = ModeBase(tA_new, valA.real, valA.imag)
    wfB = ModeBase(tB_new, valB.real, valB.imag)
    if len(valA) == len(valB):
        return wfA, wfB
    ipeak_A = wfA.argpeak
    ipeak_B = wfB.argpeak
    if ipeak_A > ipeak_B:
        idx_A = ipeak_A - ipeak_B
        idx_B = 0
    else:
        idx_A = 0
        idx_B = ipeak_B - ipeak_A
    # tmove = (ipeak_A - ipeak_B) * dt_final
    wfA = wfA[idx_A:]
    wfB = wfB[idx_B:]
    lenA = len(wfA)
    lenB = len(wfB)
    ipeak_A = ipeak_A - idx_A
    ipeak_B = ipeak_B - idx_B
    tail_A = lenA - ipeak_A
    tail_B = lenB - ipeak_B
    if tail_A > tail_B:
        lpad = tail_A - tail_B
        wfB.pad((0,lpad), 'constant', dt_final)
    else:
        lpad = tail_B - tail_A
        wfA.pad((0,lpad), 'constant', dt_final)
    return wfA, wfB        

def calculate_ModeFF(modeA, modeB, psd, Mtotal = 20, deltaT = None, retall = False):
    modeA, modeB = Mode_alignment(modeA, modeB, deltaT = deltaT)
    Atilde = np.fft.fft(modeA.value)
    Btilde = np.fft.fft(modeB.value)
    dtM = (modeA.time[1] - modeA.time[0])
    NFFT = len(modeA)
    if not hasattr(Mtotal, '__len__'):
        dt = dtM / dim_t(Mtotal)
        df = 1./NFFT/dt
        freqs = np.abs(np.fft.fftfreq(NFFT, dt))
        power_vec = psd(freqs)
        O11 = np.sum(Atilde * Atilde.conjugate() / power_vec).real * df
        O22 = np.sum(Btilde * Btilde.conjugate() / power_vec).real * df
        Ox = Atilde * Btilde.conjugate() / power_vec
        Oxt = np.fft.ifft(Ox) * NFFT * df
        Oxt_abs = np.abs(Oxt) / np.sqrt(O11 * O22)
        idx = np.argmax(Oxt_abs)
        # should apply to ModeA by ModeA * np.exp(1.j*delta_phase)
        delta_phase = -np.angle(Oxt[idx])
        lth = len(Oxt_abs)
        if idx > lth / 2:
            tc = (idx - lth) * dtM
        else:
            tc = idx * dtM
        if retall:
            return Oxt_abs[idx], delta_phase, tc, modeA, modeB
        return Oxt_abs[idx], delta_phase, tc
    FF_list = []
    d_phase_list = []
    tc_list = []
    for mtotal in Mtotal:
        dt = dtM / dim_t(mtotal)
        df = 1./NFFT/dt
        freqs = np.abs(np.fft.fftfreq(NFFT, dt))
        power_vec = psd(freqs)
        O11 = np.sum(Atilde * Atilde.conjugate() / power_vec).real * df
        O22 = np.sum(Btilde * Btilde.conjugate() / power_vec).real * df
        Ox = Atilde * Btilde.conjugate() / power_vec
        Oxt = np.fft.ifft(Ox) * NFFT * df
        Oxt_abs = np.abs(Oxt) / np.sqrt(O11 * O22)
        idx = np.argmax(Oxt_abs)
        # should apply to ModeA by ModeA * np.exp(1.j*delta_phase)
        delta_phase = -np.angle(Oxt[idx])
        FF_list.append(Oxt_abs[idx])
        d_phase_list.append(delta_phase)
        lth = len(Oxt_abs)
        if idx > lth / 2:
            tc = (idx - lth) * dtM
        else:
            tc = idx * dtM
        tc_list.append(tc)
    if retall:
        return np.asarray(FF_list), np.asarray(d_phase_list), np.asarray(tc_list), modeA, modeB
    return np.asarray(FF_list), np.asarray(d_phase_list), np.asarray(tc_list)




def h22_alignment(wfA, wfB, peak_A = None, peak_B = None):
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
    if peak_A is None:
        peak_A = wfA.argpeak
    if peak_B is None:
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

