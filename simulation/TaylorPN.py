#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:41:46 2019

@author: drizl
"""

import numpy as np
from . import Binary as BR

class TaylorF2(BR.BinaryParams):
    def __init__(self, q, Mtotal, chi1z, chi2z):
        super(TaylorF2, self).__init__(q, Mtotal, 0, 0, chi1z, 0, 0, chi2z)
        
    def calculate_phase(self, freqs, 
                        tc = 0, phic = 0):
        MChirp = self.MChirp_t
        Mtotal = self.Mtotal_t
        beta = self.beta
        sigma = self.sigma
        eta = self.eta
        eta2 = eta*eta
        phase0 = 2*np.pi*freqs*tc - phic - np.pi/4
        x = np.power(np.pi * Mtotal * freqs, 2./3.)
        x32 = np.power(x, 3./2.)
        x2 = x*x
        phasePre = (3./128.)*np.power(np.pi*MChirp*freqs, -5./3.)
        phase1 = 1 + (3715/756+55*eta/9)*x - 4*(4*np.pi-beta)*x32 + \
            (15293365/508032+27145*eta/504+3085*eta2/72 - 10*sigma)*x2
        return phase0 + phasePre*phase1
        
    def calculate_time(self, freqs, tc = 0):
        MChirp = self.MChirp_t
        beta = self.beta
        sigma = self.sigma
        eta = self.eta
        eta2 = eta*eta
        x = np.power(np.pi * self._Mtotal * freqs, 2./3.)
        x32 = np.power(x, 3./2.)
        x2 = x*x
        timePre = -(5/256)*MChirp*np.power(np.pi*MChirp*freqs, -5./3.)
        time1 = 1 + (4./3.)*(743/336+11*eta/4)*x - (8/5)*(4*np.pi-beta)*x32 +\
            2*(3058673/1016064 + 5429*eta/1008 + 617*eta2/144 - sigma)*x2
        return tc + timePre * time1

    def calculate_amplitude(self, freqs):
        MChirp = self.MChirp
        Amp = np.power(MChirp, 5./6.) / np.sqrt(30) / np.power(np.pi, 2./3.) / dL
        freq_m76 = np.power(freqs, -7./6.)
        return np.sqrt(3/4) * Amp * freq_m76
