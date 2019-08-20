#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 09:56:42 2019

@author: drizl
"""

import numpy as np
from ..generator import CMD_lalsim_inspiral
from ..Utils import cmd_stdout_cev, CEV
from .detectors import Detector
from .strain import gwStrain
from . import signal as sgl
from scipy.interpolate import InterpolatedUnivariateSpline
from ..h22datatype import pc_SI

#------------CMD to generate ifo template----------#
def detector_strain(CMD_tmpl, 
                        ra, de, ifo, t_inj = 1186624818):
    CMD = f'{CMD_tmpl} | -r -a {ra} -d {de} -D {ifo} -t {t_inj} -p 0'
    return CMD

class template(object):
    def __init__(self, m1, m2, s1z, s2z, 
                 fini = 20, 
                 approx = 'SEOBNRv4', 
                 srate = 4096, 
                 D = 100,
                 duration = None):
        self._role = 'template'
        self._m1 = m1
        self._m2 = m2
        self._s1z = s1z
        self._s2z = s2z
        self._fini = fini
        self._srate = srate
        self._approx = approx
        self._D = D
        state, data = cmd_stdout_cev(self.CMD, name_out = 'template')
        if data is None or len(data) == 0:
            state = CEV.GEN_FAIL
        self._STATE = state
        if state is not CEV.SUCCESS:
            # regeneration
            fail = True
            for i in range(1,5):
                fs_retry = int(self._srate * i)
                state, data = cmd_stdout_cev(self.fCMD(fs_retry), name_out = 'template')
                if data is not None and len(data) != 0:
                    fail = False
                    self._STATE = state
                    break
            if fail:
                self._ht = None
            else:
                # do resample
                ht = np.asarray(data[:,1]) + 1.j * np.asarray(data[:,2])
                time_new, ht_new = sgl.resample(data[:,0], ht, self._srate)
                self._ht = ht_new
        else:
            self._ht = np.asarray(data[:,1]) + 1.j*np.asarray(data[:,2])
            if duration is not None:
                self._check_duration(duration)
                
    def _check_duration(self, duration):
        if self.dtpeak > abs(duration):
            cut_idx = self.argpeak - int(abs(duration) * self.fs)
            self._ht = self._ht[cut_idx:]
    
    def fCMD(self, fs):
        return CMD_lalsim_inspiral(exe = 'lalsim-inspiral',
                                   m1 = self._m1,
                                   m2 = self._m2,
                                   s1z = self._s1z,
                                   s2z = self._s2z,
                                   D = self._D,
                                   srate = fs,
                                   f_ini = self._fini,
                                   approx = self._approx)
    @property
    def distance(self):
        return self._D
    
    @property
    def distance_SI(self):
        return self._D * pc_SI * 1e6
    
    @property
    def CMD(self):
        return self.fCMD(self._srate)
    
    @property
    def fs(self):
        return self._srate
    
    @property
    def template(self):
        if self._ht is not None:
            return self._ht
        else:
            return None
    
    @property
    def real(self):
        if self._ht is not None:
            return self._ht.real
        else:
            return None
    
    @property
    def imag(self):
        if self._ht is not None:
            return self._ht.imag
        else:
            return None
    
    @property
    def STATE(self):
        return self._STATE
            
    @property
    def m1(self):
        return self._m1
    
    @property
    def m2(self):
        return self._m2
    
    @property
    def s1z(self):
        return self._s1z
    
    @property
    def s2z(self):
        return self._s2z
    
    @property
    def approx(self):
        return self._approx
    
    def __len__(self):
        if self._ht is not None:
            return len(self._ht)
        else:
            return 0
    
    @property
    def htilde_real(self):
        return np.fft.rfft(self.real)
    
    @property
    def htilde_imag(self):
        return np.fft.rfft(self.imag)
    
    @property
    def rfftfreq(self):
        return np.fft.rfftfreq(len(self), d = 1./self.fs)
        
    def get_track(self, tpeak, extra_index = 5):
        return sgl.get_track(self.time, self.template, tpeak, extra_index = extra_index)
    
    @property
    def deltat(self):
        return 1./self._srate
    
    @property
    def duration(self):
        return len(self) / self._srate
    
    @property
    def argpeak(self):
        return np.argmax(np.abs(self.template))
    
    @property
    def dtpeak(self):
        return self.time[self.argpeak] - self.time[0]
    
    @property
    def time(self):
        return np.arange(0, len(self) * self.deltat, self.deltat)
    
    @property
    def track(self):
        return self.get_track(0, 0)
    
    def get_time_shift(self, freq):
        track_x, track_y = self.track
        track_x -= track_x[0]
        idx = np.where(np.abs(track_y - freq) == np.min(np.abs(track_y - freq)))[0][0]
        return track_x[idx]
        
    def get_horizon(self, psd, ret_SI = True):
        h = self.template
        htilde = np.fft.rfft(h.real) / self.fs
        hfreq = np.fft.rfftfreq(h.size, d = 1./self.fs)
        df = hfreq[1] - hfreq[0]
        power_vec = psd(hfreq)
        ohf = 1*(htilde * htilde.conjugate() / power_vec).sum() * df
        sig2 = np.abs(ohf)
        rhor = self.distance_SI * np.sqrt(sig2)
        if ret_SI:
            return rhor
        else:
            # In Mpc
            return rhor / pc_SI / 1e6


        
    def plot(self, 
             xrange = None, 
             yrange = None, 
             epoch = False, 
             fsave = None,
             title = None,
             figsize = None):
        if epoch:
            epoch = self.epoch
        else:
            epoch = None
        if title is None:
            title = self._role
        data = self.real
        sgl.plotit(self.time, data,
                   xrange = xrange, yrange = yrange,
                   title = title,
                   xlabel = 'time', ylabel = None,
                   epoch = epoch, save = fsave, 
                   figsize = figsize)
    
    
    def construct_detector_strain(self, ifo, ra, de, t_inj,
                                  psi = 0, phic = 0,
                                  D = None, noise = None):
        if self.STATE is not CEV.SUCCESS:
            return None
        det = Detector(ifo)
        ap = det.antenna_pattern(ra, de, psi, gps = t_inj)
        dt = det.time_delay_from_earth_center(ra, de, t_inj)
        t_inj += dt
        wf = (self.distance/D) * self.template * np.exp(1.j*phic)
        strain = wf.real * ap[0] + wf.imag * ap[1]
        t_start = t_inj - np.abs(wf).argmax() / self.fs
        t_end = t_start + len(strain) / self.fs
        if not isinstance(noise, gwStrain):
            return gwStrain(strain, epoch = t_start, fs = self.fs, ifo = ifo)
        else:
            if t_end < noise.epoch or t_start > noise.epoch + noise.duration:
                return noise
            if self.fs != noise.fs:
                th, strain = sgl.resample(self.time, strain, noise.fs)
            else:
                th = self.time
            fs = noise.fs
            if t_start < noise.epoch:
                strain = strain[int((noise.epoch-t_start)*fs):]
                th = th[int((noise.epoch-t_start)*fs):]
                t_start = noise.epoch
            if t_end > noise.epoch + noise.duration:
                strain = strain[:int((t_end - noise.epoch - noise.duration)*fs)]
                th = th[:int((t_end - noise.epoch - noise.duration)*fs)]
                t_end = noise.epoch + noise.duration
            th = th - th[0] + t_start
            itp_strain = InterpolatedUnivariateSpline(th, strain)
            idx_start = int((t_start - noise.epoch) * fs)
            idx_end = int((t_end - noise.epoch) * fs)
            val = noise.value
            vtime = noise.time
            val[idx_start:idx_end] += itp_strain(vtime[idx_start:idx_end])
            return gwStrain(val, epoch = noise.epoch, fs = fs, ifo = ifo)
            

#-------------------------------------------------------#
#-------------------------------------------------------#
        
