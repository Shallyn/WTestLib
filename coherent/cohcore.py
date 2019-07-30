#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:14:28 2019

@author: drizl
"""

import numpy as np
from ..strain.strain import gwStrain
from ..h22datatype import c_SI
from ..strain.template import template
from ..strain.detectors import gmst_accurate, Detector
from ..strain import signal as sgl

#--------------------Combined Strain-------------------#
class gwStrainCOH(object):
    def __init__(self, sLIST):
        self._sLIST = sLIST
        # Check input
        self._check_sLIST()
        
    def _check_sLIST(self):
        for i, strain in enumerate(self):
            # Check datatype
            if not isinstance(strain , gwStrain):
                raise TypeError('The type of input is not gwStrain.')
            
            # Check datalength
            if i==0:
                _len = len(strain)
            elif len(strain) != _len:
                raise ValueError('The length of input strains are not equal to each other.')
            self._Nt = _len
            
            # Check sample rate
            if i==0:
                _srate = strain.fs
            elif strain.fs != _srate:
                raise ValueError('The sample rate of input strains are not equal to each other.')
            self._srate = _srate
            
            # Check epoch
            if i==0:
                _epoch = strain.epoch
            elif strain.epoch != _epoch:
                raise ValueError('The epoch of input strains are not equal to each other.')
            self._epoch = _epoch
            
    def __len__(self):
        return self._len
    
    @property
    def fs(self):
        return self._srate
    
    @property
    def epoch(self):
        return self._epoch
    
    def __iter__(self):
        for strain in self._sLIST:
            yield strain

    def get_bayesian_localizer(self, tmpl, gps = None,
                               distance_factor = 2):
        if not isinstance(tmpl, template):
            raise TypeError('The type of tmpl should be strain.template.template.')
            
        if tmpl.fs != self.fs:
            raise ValueError('Sample rate of template and strain data is not equal to each other.')
        
        
        if gps is None:
            gps = self.epoch
        gmst = gmst_accurate(gps)
            
class bayesian_localizer(object):
    def __init__(self, 
                 gps_trigger_geocent,
                 distance_factor = 2):
        self._gps = gps_trigger_geocent
        self._gmst = gmst_accurate(self._gps)
        self._bLIST = []
        self._r_max = 0
        self._T_max = 0
        self._prior_r = lambda r : np.power(r, distance_factor)
    
    def __len__(self):
        return len(self._bLIST)
    
    def __iter__(self):
        for ele in self._bLIST:
            yield ele
    
    @property
    def r_max(self):
        return self._r_max
    
    @property
    def T_max(self):
        return self._T_max
        
    def append_sngl(self, new):
        if isinstance(new, bayesian_sngl):
            self._bLIST.append(new)
            if new.T_max > self.T_max:
                self._T_max = new.T_max
            if new.r_max > self.r_max:
                self._r_max = new.r_max
        else:
            raise Exception('Incorrect input type(should be bayesian_sngl).')
       
    def __call__(self, ra, dec):
        if len(self) == 0:
            raise Exception('You have not added sngl bayesian yet!')
        
        lnprob = 0
        samp_phic = np.linspace(0, np.pi, 20)
        samp_iota = np.linspace(-1, 1, 20)
        samp_psi = np.linspace(0, np.pi, 20)
        
        func_dict = {}
        for bayes in self:
            delay = bayes.ftimedelay(ra, dec, self._gmst)
            Gpc = bayes.fGpc(ra, dec, self._gmst)
            gps_trigger = self._gps + delay
            
            # Calculate template
            hrtilde = np.fft.rfft(bayes.template.real) / bayes.fs
            hitilde = np.fft.rfft(bayes.template.imag) / bayes.fs
            stilde = np.fft.rfft(bayes.svalue) / bayes.fs
            freqs = np.fft.rfftfreq(len(bayes), d = 1./bayes.fs)
            power_vec = bayes.psd(freqs)
            df = freqs[1] - freqs[0]
            dt = 1./bayes.fs
            
            # Inner product basic
            inner_hr = 2.*(hrtilde * hrtilde.conjugate() / power_vec).sum() * df
            inner_hr = np.abs(inner_hr)
            inner_hi = 2.*(hitilde * hitilde.conjugate() / power_vec).sum() * df
            inner_hi = np.abs(inner_hi)
            
            corr_hr = 2.*(stilde * hrtilde.conjugate() / power_vec)
            corr_hr_time = np.fft.irfft(corr_hr) * bayes.fs
            corr_hr_time = corr_hr_time.real
            
            corr_hi = 2.*(stilde * hitilde.conjugate() / power_vec)
            corr_hi_time = np.fft.irfft(corr_hi) * bayes.fs
            corr_hi_time = corr_hi_time.real
            
            # tc integration cut
            dt_start = gps_trigger - bayes.epoch - self.T_max
            dt_end = gps_trigger - bayes.epoch + self.T_max
            index_trigger_slice = slice(int(dt_start / bayes.fs) , int(dt_end / bayes.fs))
            
            # Apply time integration
            fac_hr_part1 = 2*corr_hr_time[index_trigger_slice].sum() * dt
            fac_hr_part2 = inner_hr * (index_trigger_slice.stop - index_trigger_slice.start) * dt
            fac_hi_part1 = 2*corr_hi_time[index_trigger_slice].sum() * dt
            fac_hi_part2 = inner_hi * (index_trigger_slice.stop - index_trigger_slice.start) * dt
            
            # Apply angular & distance integration
            # Calculate Gpc Matrixes
            
            
            
        

class bayesian_sngl(object):
    def __init__(self, strain, tmpl, 
                 psd = 'set',
                 r_max = None,
                 T_max = None):
        if not isinstance(tmpl, template):
            raise TypeError('The type of tmpl should be strain.template.template.')
            
        if tmpl.fs != strain.fs:
            raise ValueError('Sample rate of template and strain data is not equal to each other.')
        
        Det = Detector(strain.ifo)
        self._ifo = strain.ifo
        self._amplitude_modulation = Det.amplitude_modulation
        self._time_delay_from_earth_center = Det.time_delay_from_earth_center_gmst
        
        if psd in (None, 'set',):
            self._psd = strain.psdfun_setted
        elif psd in ('self',):
            self._psd = strain.psdfun()
        elif hasattr(psd, '__call__'):
            self._psd = psd
        else:
            self._psd = lambda x : 1
        
        if T_max is None:
            loc = strain.ifo_location
            delay = np.linalg.norm(loc) / c_SI
            T_max = delay + 5e-3
        self._T_max = T_max
        
        if r_max is None:
            horizon = tmpl.get_horizon(psd = psd, ret_SI = False)
            r_max = horizon / 4
        self._r_max = r_max
        self._horizon = horizon
        
        hbar = (tmpl.distance / self._horizon) * tmpl.template
        if len(hbar) > strain.Nt:
            s, h = sgl.cutinsert(strain.value, hbar)
        elif len(hbar) < strain.Nt:
            s,h = sgl.padinsert(strain.value, hbar)
        else:
            s = strain.value
        
        self._template = h
        self._strain = s
        self._epoch = strain.epoch + tmpl.dtpeak
        self._fs = strain.fs
        
        self._detrsp = Det.response
    
    def __len__(self):
        return len(self._strain)
    
    @property
    def fs(self):
        return self._fs
        
    @property
    def epoch(self):
        return self._epoch
        
    @property
    def svalue(self):
        return self._strain
        
    @property
    def ifo(self):
        return self._ifo
        
    @property
    def fGpc(self):
        return self._amplitude_modulation
    
    @property
    def ftimedelay(self):
        return self._time_delay_from_earth_center
    
    @property
    def T_max(self):
        return self._T_max
    
    @property
    def r_max(self):
        return self._r_max
    
    def amp(self, r):
        return self._horizon / r
    
    @property
    def psd(self):
        return self._psd
    
    @property
    def template(self):
        return self._template
    
    def calc_angular_amplitude_modulation(self, 
                                          ra, dec, gmst,
                                          psi, iota, phic):
        # iota = cos(inclination)
        gha = gmst - ra
        D = self._detrsp
        cosgha = np.cos(gha)
        singha = np.sin(gha)
        cosdec = np.cos(dec)
        sindec = np.sin(dec)

        x0 = -singha
        x1 = -cosgha
        x2 = 0
        x = np.array([x0, x1, x2])

        dx = np.dot(D, x)

        y0 =  -cosgha * sindec
        y1 =  singha * sindec
        y2 =  cosdec
        y = np.array([y0, y1, y2])
        dy = np.dot(D, y)
        Gplus = (x * dx - y * dy).sum()
        Gcross = (x * dy + y * dx).sum()
        Gpc = np.array([Gplus, Gcross])
        
        angM = angular_matrix(psi, iota, phic)
        #npsi, niota, nphic, _, __ = angM.shape
        
        ret = np.dot(Gpc, angM)
        return ret
        
        
        
            
        
def angular_matrix(psi, iota, phic):
    # iota = cos(inclination)
    if hasattr(psi, '__len__'):
        npsi = len(psi)
    else:
        npsi = 1
        psi = np.array([psi])
    
    if hasattr(iota, '__len__'):
        niota = len(iota)
    else:
        niota = 1
        iota = np.array([iota])
        
    if hasattr(phic, '__len__'):
        nphic = len(phic)
    else:
        nphic = 1
        phic = np.array([phic])
        
    ret = np.zeros([npsi,niota,nphic,2,2])
    
    cos2psi = np.cos(2*psi)
    sin2psi = np.sin(2*psi)
    cosiota = iota
    cosiotapt = (1 + cosiota**2) / 2
    cosphic = np.cos(phic)
    sinphic = np.sin(phic)
    
    for i in range(npsi):
        for j in range(niota):
            ret[i,j,:,0,0] = cosiotapt[j] * cos2psi[i] * cosphic - cosiota[j] * sin2psi[i] * sinphic
            ret[i,j,:,0,1] = -cosiotapt[j] * cos2psi[i] * sinphic - cosiota[j] * sin2psi[i] * cosphic
            ret[i,j,:,1,0] = cosiotapt[j] * sin2psi[i] * cosphic + cosiota[j] * cos2psi[i] * sinphic
            ret[i,j,:,1,1] = -cosiotapt[j] * sin2psi[i] * sinphic + cosiota[j] * cos2psi[i] * cosphic
    return ret
 
    