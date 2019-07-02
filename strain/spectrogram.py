#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:09:17 2019

@author: drizl
"""

import numpy as np
from .utils import get_psdfun

def spec_STFT(data, Fs, 
              NFFT = None, 
              NOLP = None, 
              window = None,
              epoch = None,
              cut = None):
    # psdfun = get_psdfun(data, Fs)
    if NFFT is None:
        NFFT = int(Fs/4)
    if NOLP is None or NOLP >= NFFT:
        NOLP = int(NFFT * 19 / 20)
    if window is None:
        window = np.hamming(NFFT)
    if epoch is None:
        epoch = 0
    if cut is None:
        cut = [30, 1000]
    
    data = np.asarray(data)
    ndata = data.size
    wstep = NFFT - NOLP
    deltat = wstep / Fs
    nwind = int((ndata - NFFT) / wstep)
    tini = epoch + deltat
    tout = np.zeros(nwind)
    fout = np.arange(0,NFFT/2 + 1) * Fs / NFFT
    deltaf = fout[1] - fout[0]
    cutlowidx = int(cut[0] / deltaf)
    cuthighidx = int(cut[1] / deltaf)
    print(deltaf)
    cutslice = slice(cutlowidx, cuthighidx)
    fout = fout[cutslice]
    # psd = psdfun(fout)
    spec = np.zeros([tout.size, fout.size], dtype = np.complex)
    for i in range(nwind):
        tout[i] = tini + deltat * i
        index = slice(i * wstep, i * wstep + NFFT)
        spec[i,:] = np.fft.rfft(data[index] * window)[cutslice]
    return tout, fout, spec
    
