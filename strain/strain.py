#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:05:30 2019

@author: drizl
"""

import numpy as np
from .datatypebase import TimeSeriesBase
from .detectors import Detector
from scipy.interpolate import interp1d
from . import signal as sgl
from .snr_qTansform import snr_q_scanf
from scipy.signal import resample

# H1-118 L1-150 V1-53
def get_sigma2(ifo):
    if ifo == 'L1':
        return (150 * 2)**2
    if ifo == 'H1':
        return (118 * 2)**2
    if ifo == 'V1':
        return (53 * 2)**2


#-----------------------My Strain Series Class--------------------#
# Must be real series
class gwStrain(TimeSeriesBase):
    def __init__(self, strain, epoch, ifo, fs, info = ''):
        super(gwStrain, self).__init__(data = strain, epoch = epoch, fs = fs, info = info)
        self._ifo = ifo
        Det = Detector(self.ifo)
        self.ifo_latitude = Det.latitude
        self.ifo_longtitude = Det.longtitude
        self.ifo_location = Det.location
        self.ifo_response = Det.response
        self.ifo_antenna_pattern = Det.antenna_pattern
        self.ifo_delay = Det.time_delay_from_earth_center
        self._psdfun_setted = False
        
    @property
    def ifo(self):
        return self._ifo
    
    @property
    def stilde(self):
        return np.fft.rfft(self.value)
    
    @property
    def sigma2(self):
        return get_sigma2(self.ifo)
    
    @property
    def freq(self):
        return np.fft.rfftfreq(n = self.Nt, d = self._deltat)
    
    @property
    def Nf(self):
        return int(self.Nt / 2) + 1
    
    @property
    def deltaf(self):
        return 1./self.deltat/self.Nt
    
    def resample(self, fs):
        if fs != self.fs:
            nsamp = int(self.value.size * self.deltat * fs)
            new = resample(self.value, nsamp)
            return gwStrain(new, self.epoch, self.ifo, fs, self._role)
        else:
            return self
    
    def __getitem__(self, key):
        return self._getslice(key)

    def _getslice(self, index):
        if index.start < 0:
            raise ValueError(('Negative start index ({}) is not supported').format(index.start))
        new_epoch = self.epoch + index.start * self.deltat
        
        if index.step is not None:
            new_deltat = self.deltat * index.step
        else:
            new_deltat = self.deltat
        new_fs = 1./new_deltat
        return gwStrain(strain = self.value[index], epoch = new_epoch, ifo = self.ifo, fs = new_fs)

        
    def psdfun(self, NFFT = None, NOVL = None, window = False):
        return sgl.get_psdfun(data = self.value, fs = self.fs, NFFT = NFFT, NOVL = NOVL, window = window)
    
    def set_psd(self, psd):
        self._psdfun_setted = psd
        
    @property
    def psdfun_setted(self):
        return self._psdfun_setted
    
    def veto(self, start, end):
        idx_start = int((start - self.epoch) * self.fs)
        idx_end = int((end - self.epoch) * self.fs)
        self._value[idx_start:idx_end] = 0
        
    
    def plot(self, 
             xrange = None, 
             yrange = None, 
             epoch = False, 
             fsave = None,
             title = None,
             ylabel = None,
             figsize = None):
        if epoch:
            epoch = self.epoch
        else:
            epoch = None
        if title is None:
            title = self._role
        if self.value.dtype == complex:
            data = np.abs(self.value)
        else:
            data = self.value
        sgl.plotit(self.time, data,
                   xrange = xrange, yrange = yrange,
                   title = title,
                   xlabel = 'time', ylabel = ylabel,
                   epoch = epoch, save = fsave, 
                   figsize = figsize)

    def matched_filter(self, 
                       tmpl, 
                       cut = None, 
                       window = True, 
                       psd = None,
                       ret_complex = False,
                       shift = 0):
        h = np.asarray(tmpl)
        if len(h) > self.Nt:
            s, h = sgl.cutinsert(self.value, h)
        elif len(h) < self.Nt:
            s,h = sgl.padinsert(self.value, h)
        else:
            s = self.value
        if psd in ('self',):
            psd = self.psdfun()
        if psd in ('set',):
            psd = self._psdfun_setted
        SNR, sigma2 = sgl.matched_filter(s, h, 
                                         fs = self.fs, psdfun = psd, 
                                         cut = cut, window = window,
                                         ret_complex = ret_complex)
        SNRstrain = gwStrain(SNR, epoch = self.epoch + shift, ifo = self.ifo, fs = self.fs)
        return SNRstrain
    
    def matched_filter_cplx(self,
                            tmpl,
                            cut = None,
                            window = True,
                            psd = None):
        h = np.asarray(tmpl)
        if len(h) > self.Nt:
            s, h = sgl.cutinsert(self.value, h)
        elif len(h) < self.Nt:
            s,h = sgl.padinsert(self.value, h)
        else:
            s = self.value
        if psd is 'self':
            psd = self.psdfun()
        snr = sgl.matched_filter_Real(s, h, fs = self.fs, psdfun = psd, cut = cut, window = window)
        SNRstrain = gwStrain(np.abs(snr), epoch = self.epoch, ifo = self.ifo, fs = self.fs)
        return SNRstrain

    
    def matched_filter_convolve(self, tmpl, cut = None, psd = None, mode = 'full', shift = 0):
        if psd in ('self', None,):
            psd = self.psdfun()
        s_whiten, s_sig2 = sgl.whiten(self.value, psd, self.fs)
        if tmpl.dtype == complex:
            hr_whiten, hr_sig2 = sgl.whiten(tmpl.real, psd, self.fs)
            hr_whiten /= np.sqrt(np.abs(hr_sig2))
            hi_whiten, hi_sig2 = sgl.whiten(tmpl.imag, psd, self.fs)
            hi_whiten /= np.sqrt(np.abs(hi_sig2))
            import matplotlib.pyplot as plt
            plt.plot(hr_whiten)
            plt.show()
            op = 2*np.convolve(s_whiten, hr_whiten, mode = mode) + \
                2.j*np.convolve(s_whiten, hi_whiten, mode = mode)
        else:
            h_whiten, h_sig2 = sgl.whiten(tmpl, psd, self.fs)
            op = 2*np.convolve(s_whiten, h_whiten, mode = mode)
        SNRstrain = gwStrain(np.abs(op), epoch = self.epoch + shift, ifo = self.ifo, fs = self.fs)
        return SNRstrain

    #FIXME
    def snr_q_scan(self, tmpl, cut = None, window = True, psd = None, **kwargs):
        if psd in ('set',) and self._psdfun_setted is not False:
            psd = self._psdfun_setted
        ret = snr_q_scanf(data = self.value, tmpl = tmpl, window = window, psd = psd,
                          sampling = self.fs, epoch = self.epoch, retfunc = True ,**kwargs)
        self.q_snrfunc = ret[0]
        self.q_snrtime = ret[1]
        self.q_snrfreq = ret[2]
        
    # def q_scan(self, psd = None, ret_complex = True,**kwargs):
    #     if psd is None:
    #         psd = self.psdfun
    #     elif psd in ('NoPsd',):
    #         psd = None
    #     ret = q_scanf(self.stilde, self.time, retfunc = True, psd = psd, ret_complex = ret_complex,**kwargs)
    #     self.q_energyfunc = ret[0]
    #     self.q_time = ret[1]
    #     self.q_freq = ret[2]
    #     self.q_dtenergy = self.q_energyfunc(self.q_time, self.q_freq)
    

