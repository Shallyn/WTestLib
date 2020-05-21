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
    def __init__(self, name = None, flow = 0):
        if isinstance(name, DetectorPSD):
            name = name.name
        self._name = name
        self._choose_psd(flow)
                
    def __call__(self, *args, **kwargs):
        return self._psd(*args, **kwargs)
    
    def _choose_psd(self, flow = 0):
        file = None
        for case in switch(self._name):
            if case('ET'):
                file = LOC / 'LIGO-P1600143-v18-ET_D.txt'
                self._psd = loadPSD_from_file(file, flow)
                break
            if case('ET_fit'):
                file = None
                self._psd = PSD_ET_fit
                break
            if case('CE_Pes'):
                file = LOC / 'LIGO-P1600143-v18-CE_Pessimistic.txt'
                self._psd = loadPSD_from_file(file, flow)
                break
            if case('CE_Wide'):
                file = LOC / 'LIGO-P1600143-v18-CE_Wideband.txt'
                self._psd = loadPSD_from_file(file, flow)
                break
            if case('CE'):
                file = LOC / 'LIGO-P1600143-v18-CE.txt'
                self._psd = loadPSD_from_file(file, flow)
                break
            if case('advLIGO'):
                file = LOC / 'LIGO-P1200087-v18-aLIGO_DESIGN.txt'
                self._psd = loadPSD_from_file(file, flow)
                break
            if case('advLIGO_fit'):
                file = None
                self._psd = PSD_advLIGO_fit
                break
            if case('advLIGO_zerodethp'):
                file = LOC /"ZERO_DET_high_P.txt"
                self._psd = loadPSD_from_file(file, flow)
                break
            if case('L1'):
                file = LOC / 'LIGOLivingston_O3PSD-1241571618-21600.txt'
                self._psd = loadPSD_from_file(file, flow, exp = False)
                break
            if case('H1'):
                file = LOC / 'LIGOHanford_O3PSD-1241571618-21600.txt'
                self._psd = loadPSD_from_file(file, flow, exp = False)
                break
            if case('V1'):
                file = LOC / 'Virgo_O3PSD-1241571618-21600.txt'
                self._psd = loadPSD_from_file(file, flow, exp = False)
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

    

def loadPSD_from_file(file, flow = 0, exp = True):
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
    def funcPSD(freq):
        ret = func(freq)
        if hasattr(freq, '__len__'):
            ret[np.where(freq < flow)] = np.inf
        elif freq < flow:
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

