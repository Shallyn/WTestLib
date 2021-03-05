#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:19:40 2019

@author: drizl
"""

import numpy as np
from ..Utils import interp1d_complex
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib.mlab import psd as mypsd
from .detectors import Detector
from .qTransform import q_scanf
from .snr_qTansform import snr_q_scanf
from .signal import resample

# H1-118 L1-150 V1-53
def get_sigma2(ifo):
    if ifo is 'L1':
        return (150 * 2)**2
    if ifo is 'H1':
        return (118 * 2)**2
    if ifo is 'V1':
        return (53 * 2)**2

#-----------------------My Time Series Class--------------------#
class TimeSeriesBase(object):
    def __init__(self, data, epoch, fs, info = ''):
        self._value = np.asarray(data)
        self._epoch = epoch
        self._fs = fs
        self._role = info

    @property
    def epoch(self):
        return self._epoch
    
    @property
    def deltat(self):
        return 1./self._fs
    
    @property
    def fs(self):
        return self._fs
        
    @property
    def Nt(self):
        return len(self)

    @property
    def value(self):
        return self._value

    @property
    def time(self):
        return np.arange(0, self.Nt * self.deltat, self.deltat) + self._epoch

    @property
    def sfun(self):
        if self.value.dtype == complex:
            return interp1d_complex(self.time, self.value)
        else:
            return InterpolatedUnivariateSpline(self.time, self.value)
    
    @property
    def duration(self):
        return len(self) * self.deltat
    
    @property
    def argpeak(self):
        return np.argmax(np.abs(self.value))
    
    @property
    def time_peak(self):
        return self.time[self.argpeak]
    
    @property
    def real(self):
        if hasattr(self.value, 'real'):
            return self.value.real
        else:
            return self.value
    
    @property
    def imag(self):
        if hasattr(self.value, 'imag'):
            return self.value.imag
        else:
            return np.zeros(len(self))
        
    def resample(self, fs):
        if fs != self.fs:
            time, value = resample(self.time, self.value, fs)
            return TimeSeriesBase(value, self.epoch, self.fs, self._role)
        else:
            return self
    
    def _getslice(self, index):
        if index.start < 0:
            raise ValueError(('Negative start index ({}) is not supported').format(index.start))
        new_epoch = self.epoch + index.start * self.deltat
        
        if index.step is not None:
            new_deltat = self.deltat * index.step
        else:
            new_deltat = self.deltat
        new_fs = 1./new_deltat
        return TimeSeriesBase(self.value[index], epoch = new_epoch, fs = new_fs)
        
    def __str__(self):
        return '{}'.format(self.value)
    
    def __repr__(self):
        return self.__str__()
    
    def __format__(self):
        return self.__str__()
    
    def __iter__(self):
        for x in self.value:
            yield x
            
    def __len__(self):
        return len(self.value)
    
    def __getitem__(self, key):
        return self._getslice(key)
    
    def __setitem__(self, key, value):
        self.s[key] = value
    
    def __max__(self):
        return max(self.s)
    
    def __min__(self):
        return min(self.s)
    
    def __abs__(self):
        return abs(self.s)





        
        
