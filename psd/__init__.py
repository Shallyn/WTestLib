#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:25:38 2019

@author: drizl
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from ..Utils import switch
from pathlib import Path

__all__ = ['DetectorPSD']

LOC = Path(__file__).parent

class DetectorPSD(object):
    def __init__(self, name = None, flow = 0, fhigh = None):
        if isinstance(name, DetectorPSD):
            name = name.name
        self._name = name
        self._choose_psd(flow, fhigh)
                
    def __call__(self, *args, **kwargs):
        return self._psd(*args, **kwargs)
    
    def _choose_psd(self, flow = 0, fhigh = None):
        file = None
        for case in switch(self._name):
            if case('ET'):
                file = LOC / 'LIGO-P1600143-v18-ET_D.txt'
                self._psd = loadPSD_from_file(file, flow, fhigh = fhigh)
                break
            if case('ET_fit'):
                file = None
                self._psd = PSD_ET_fit
                break
            if case('CE_Pes'):
                file = LOC / 'LIGO-P1600143-v18-CE_Pessimistic.txt'
                self._psd = loadPSD_from_file(file, flow, fhigh = fhigh)
                break
            if case('CE_Wide'):
                file = LOC / 'LIGO-P1600143-v18-CE_Wideband.txt'
                self._psd = loadPSD_from_file(file, flow, fhigh = fhigh)
                break
            if case('CE'):
                file = LOC / 'LIGO-P1600143-v18-CE.txt'
                self._psd = loadPSD_from_file(file, flow, fhigh = fhigh)
                break
            if case('advLIGO'):
                file = LOC / 'LIGO-P1200087-v18-aLIGO_DESIGN.txt'
                self._psd = loadPSD_from_file(file, flow, fhigh = fhigh)
                break
            if case('advLIGO_fit'):
                file = None
                self._psd = PSD_advLIGO_fit
                break
            if case('advLIGO_zerodethp'):
                file = LOC /"ZERO_DET_high_P.txt"
                self._psd = loadPSD_from_file(file, flow, fhigh = fhigh)
                break
            if case('L1'):
                file = LOC / 'LIGOLivingston_O3PSD-1241571618-21600.txt'
                self._psd = loadPSD_from_file(file, flow, fhigh = fhigh, exp = False)
                break
            if case('H1'):
                file = LOC / 'LIGOHanford_O3PSD-1241571618-21600.txt'
                self._psd = loadPSD_from_file(file, flow, fhigh = fhigh, exp = False)
                break
            if case('V1'):
                file = LOC / 'Virgo_O3PSD-1241571618-21600.txt'
                self._psd = loadPSD_from_file(file, flow, fhigh = fhigh, exp = False)
                break
            if case('cut'):
                self._psd = get_lowCutPSD(flow = flow, fhigh = fhigh)
                break
            if case('LISA'):
                self._psd = get_PSD_Space_fit(name = 'LISA', flow = flow, fhigh = fhigh)
                break
            if case('Taiji'):
                self._psd = get_PSD_Space_fit(name = 'Taiji', flow = flow, fhigh = fhigh)
                break
            if case('Tianqin'):
                self._psd = get_PSD_Space_fit(name = 'Tianqin', flow = flow, fhigh = fhigh)
                break
            if case('DECIGO_fit'):
                self._psd = PSD_DECIGO_fit
                break
            if case('DECIGO_fit2'):
                self._psd = PSD_DECIGO_fit2
                break
            if case('DECIGO_fit3'):
                self._psd = PSD_DECIGO_fit3
                break
            self._psd = lambda x : 1
        self._file = file
            
    @property
    def name(self):
        return self._name
        
    def get_psd_data(self, exp = True):
        if self._file is None:
            return None
        data = np.loadtxt(self._file)
        freq = data[:,0]
        h = data[:,1]
        valpsd = np.zeros(len(h))
        idxsift = np.where(h > 0)
        if exp:
            valpsd[idxsift] = np.exp(2 * np.log(h[idxsift]))
        else:
            valpsd[idxsift] = h[idxsift]
        return freq, valpsd

    def _sim_noise_seg(self, seg, psd_data, srate):
        segLen = len(seg)
        df = srate / segLen
        sigma = np.sqrt(psd_data / df) / 2
        stilde = np.random.randn(len(sigma)) + 1.j*np.random.randn(len(sigma))
        return np.fft.irfft(stilde)

    def _sim_noise(self, stride, psd_data, seg, srate):
        segLen = len(seg)
        if stride == 0:
            return self._sim_noise_seg(seg, psd_data, srate)
        elif stride == segLen:
            seg = self._sim_noise_seg(seg, psd_data, srate)
            stride = 0
        overlap = seg[stride:]
        lenolp = len(overlap)
        seg = self._sim_noise_seg(seg, psd_data, srate)
        for i in range(len(overlap)):
            x = np.cos(np.pi * i / (2*lenolp))
            y = np.sin(np.pi * i / (2*lenolp))
            seg[i] = x * overlap[i] + y * seg[i]
        return seg

    def generate_noise(self, duration, srate):
        ncount = duration * srate
        segdur = 4
        length = segdur * srate
        df = srate / length
        freqs = np.fft.rfftfreq(length, df)
        psd_data = self.__call__(freqs)
        stride = int(length / 2)
        seg = np.zeros(length)
        ret = []
        while(1):
            for j in range(0, stride):
                ncount -= 1
                if ncount == 0:
                    return np.asarray(ret)
                ret.append(seg[j])
            seg = self._sim_noise(stride, psd_data, seg, srate)
            if ret is None:
                ret = seg.copy()
            else:
                ret = np.append(ret, seg)


    

def loadPSD_from_file(file, flow = 0, fhigh = None, exp = True):
    if fhigh is None:
        fhigh = np.inf
    data = np.loadtxt(file)
    freq = data[:,0]
    h = data[:,1]
    valpsd = np.zeros(len(h))
    idxsift = np.where(h > 0)
    if exp:
        valpsd[idxsift] = np.exp(2 * np.log(h[idxsift]))
    else:
        valpsd[idxsift] = h[idxsift]
    func = InterpolatedUnivariateSpline(freq, valpsd)
    if flow < freq[0]:
        flow = freq[0]
    if fhigh > freq[-1]:
        fhigh = freq[-1]
    def funcPSD(freq):
        ret = func(freq)
        if hasattr(freq, '__len__'):
            ret[np.where(freq < flow)] = np.inf
            ret[np.where(freq > fhigh)] = np.inf
        elif freq < flow:
            ret = np.inf
        elif freq > fhigh:
            ret = np.inf
        return ret
    return funcPSD
        
def get_lowCutPSD(flow = 0, fhigh = None):
    if fhigh is None:
        fhigh = np.inf
    def funcPSD(freq):
        if hasattr(freq, '__len__'):
            ret = np.ones(len(freq))
            ret[np.where(freq < flow)] = np.inf
            ret[np.where(freq > fhigh)] = np.inf
        elif freq < flow:
            ret = np.inf
        elif freq > fhigh:
            ret = np.inf
        return ret
    return funcPSD
    

"""
"""
def PSD_DECIGO_fit(f):
    fac_f_4 = np.power(f, -4)
    fac_f2 = np.power(f/7.36, 2)
    return 6.53e-49 * ( 1+fac_f2 ) + \
        4.45e-51*fac_f_4 / (1 + 1/fac_f2) + \
        4.49e-52*fac_f_4

def PSD_DECIGO_fit2(f):
    fac_f_4 = np.power(f, -4)
    fac_f2 = np.power(f, 2)
    return 1.25e-47 + 4.21e-50 * fac_f_4 + 3.92e-49 * fac_f2

def PSD_DECIGO_fit3(f):
    fac_f_4 = np.power(f, -4)
    fac_f2 = np.power(f, 2)
    return 1.88e-48 + 6.31e-51 * fac_f_4 + 5.88e-50 * fac_f2


"""
	arXiv:0908.0353
"""
def PSD_ET_fit(f):
    x = f / 200
    x2 = np.power(x,2)
    x3 = np.power(x,3)
    x4 = np.power(x,4)
    x5 = np.power(x,5)
    x6 = np.power(x,6)
    p1 = np.power(x, -4.05)
    a1p2 = 185.62 * np.power(x, -0.69)
    ret =  1.449e-52 * (p1 + a1p2 + 232.56 * \
                        (1 + 31.18*x - 46.72*x2 + 52.24*x3 - 42.16*x4 + 10.17 * x5 + 11.53 * x6) / \
                        (1 + 13.58*x - 36.46*x2 + 18.56*x3 + 27.43*x4) )
    if hasattr(f, '__len__'):
        f = np.asarray(f)
        if f.any() < 1:
            idx = np.where(f < 1)[0][0]
            ret[idx] = np.inf
        return ret
    elif f < 1:
        return np.inf
    else:
        return ret
    
    


"""
    PhysRevD.71.084008
"""
def PSD_advLIGO_fit(f):
    fac = f / 215
    fac2 = np.power(fac,2)
    fac4 = np.power(fac,4)
    ret = 1e-49 * ( np.power(fac, -4.14) - 5/fac2 + \
                    111 * ( 1 - fac2 + fac4/2 ) / (1 + fac2/2) )
    if hasattr(ret, '__len__'):
        f = np.asarray(f)
        if f.any() < 10:
            ret[np.where(f < 10)] = np.inf
        return ret
    elif f < 10:
        return np.inf
    else:
        return ret

"""
    CQG. 36.105011
"""
def get_PSD_Space_fit(name = 'LISA', flow = 0, fhigh = None):
    C_SI = 299792458
    name_low = name.lower()
    if fhigh is None:
        fhigh = np.inf
    for case in switch(name_low):
        if case('lisa'):
            P_OMS = np.power(1.5e-11, 2)
            fP_acc = lambda f : np.power(3.e-15, 2) * (1 + np.power(4.e-4/f,2))
            L = 2.5e9
            break
        if case('taiji'):
            P_OMS = np.power(8e-11, 2)
            fP_acc = lambda f : np.power(3.e-15, 2) * (1 + np.power(4.e-4/f,2))
            L = 3.e9
            break
        if case('tianqin'):
            P_OMS = np.power(1.e-12, 2)
            fP_acc = lambda f : np.power(1.e-15, 2) * (1 + np.power(1.e-4/f,2))
            L = np.sqrt(3) * 1e8
            break
        raise Exception(f'Unrecognized PSD name {name}')
    def func_PSD(f):
        fstar = C_SI/(2*np.pi*L)
        ret = (10. / (3.*L*L)) * \
            (P_OMS + 2*(1+ np.power(np.cos(f/fstar),2))*\
            (fP_acc(f)/np.power(2*np.pi*f,4)) ) * \
            (1 + 6.*np.power(f/fstar,2)/10.)
        if hasattr(f, '__len__'):
            f = np.asarray(f)
            ret[np.where(f < flow)] = np.inf
            ret[np.where(f > fhigh)] = np.inf
        elif f < flow:
            ret = np.inf
        elif f > fhigh:
            ret = np.inf
        return ret
    
    return func_PSD
    