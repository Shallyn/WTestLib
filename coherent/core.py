#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:34:20 2019

@author: drizl
"""

import numpy as np
from scipy.interpolate import interp1d
from ..strain.detectors import Detector
import sys
from ..Utils import Progress, Progress_time
import time
try:
    from . import PyGWCOH as pg
    GWCOH = True
except:
    GWCOH = False
    sys.stderr.write('Warning: cannot import PyGWCOH')

# LIST = [SNR_H1, SNR_L1, SNR_V1]
# SNR_ifo = [channel, snrTimeSeries, sigma2, index]
def utdk_times(LIST, ra_pix, de_pix, times, verbose = False):
    npix = ra_pix.size
    ntime = times.size
    ndet = len(LIST)
    #LLikeh = np.zeros([ntime, npix], np.float)
    #NSNR = np.zeros([ntime, npix], np.float)
    #null_stream = np.zeros([ntime, npix], np.float)
    if verbose:
        sys.stderr.write('Calculating utdk...\n')
        sys.stderr.write('--Calculating Gpc matrix...\n')
    if GWCOH:
        ifo_list = []
        sigma2_list = []
        for data in LIST:
            ifo_list.append(data.ifo)
            sigma2_list.append(data.sigma2)
        Gpc_matrix = pg.Gpc_time_pix(ifo_list, ra_pix, de_pix, times, sigma2_list)
    else:
        Gpc_matrix = Gpc(LIST, ra_pix, de_pix, times, verbose)
    if verbose:
        sys.stderr.write('--Calculating Gpc matrix: Done\n')
        sys.stderr.write('--Do SVD on Gpc matrix...')
    u,s,v = np.linalg.svd(Gpc_matrix)
    if verbose:
        sys.stderr.write('Done\n')
        sys.stderr.write('--Calculating rho_vec ...\n')
    if GWCOH:
        rho = rho_vec_GWCOH(LIST, ra_pix, de_pix, times)
    else:
        rho = rho_vec(LIST, ra_pix, de_pix, times, verbose)
    if verbose:
        sys.stderr.write('--Calculating rho_vec: Done\n')
        sys.stderr.write('--Multiply u, rho...')
    utdk = np.zeros([ntime, npix, ndet, ndet], np.complex)
    for i in range(ndet):
        utdk[:,:,:,i] = np.multiply(u[:,:,:,i],rho[:,:,:])
    if verbose:
        sys.stderr.write('Done\n')
    utdk = np.sum(utdk,axis=2)
    if verbose:
        sys.stderr.write('Calculating utdk: Complete\n')
    return utdk


def Gpc(snr_set, ra, de, times, verbose = False):
    npix = len(ra)
    ntime = len(times)
    ndet = len(snr_set)
    Gpc_matrix = np.zeros([ntime,npix,ndet,2],np.float)
    #Gpc_sigma  = np.zeros([ntime,npix,ndet,2],np.float)
    if verbose:
        cumitr = 0
        time_ini = time.time()
        itr_tot = npix * ntime * ndet
    for k in range(npix):
        rak = ra[k]
        dek = de[k]
        for j in range(ntime):
            for i,snr in enumerate(snr_set):
                detector = Detector(snr.ifo)
                ar = detector.antenna_pattern(rak, dek, 0, times[j])
                Gpc_matrix[j,k,i,0] = ar[0] * np.sqrt(snr.sigma2)
                Gpc_matrix[j,k,i,1] = ar[1] * np.sqrt(snr.sigma2)
                #Gpc_sigma[j,k,i,0]  = ar[0]*np.sqrt(snr.sigma2)
                #Gpc_sigma[j,k,i,1]  = ar[1]*np.sqrt(snr.sigma2)
        if verbose:
            cumitr += ndet * ntime
            Progress_time((time.time() - time_ini) / cumitr, cumitr, itr_tot)
    if verbose:
        sys.stderr.write('\r')
    return Gpc_matrix


def rho_vec(snr_set, ra, de, times, verbose = False):
    npix = len(ra)
    ntime = len(times)
    ndet = len(snr_set)
    rho = np.zeros([ntime,npix,ndet],np.complex)
    rho_i = np.zeros([ntime],np.complex)
    if verbose:
        cumitr = 0
        itr_tot = npix * len(snr_set) * ntime

    for k in range(npix):
        rak = ra[k]
        dek = de[k]
        for i, snr in enumerate(snr_set):
            fsnr = snr.sfun
            detector = Detector(snr.ifo)
            gps_time = times[int(ntime/2)]
            dt  = detector.time_delay_from_earth_center(rak, dek, gps_time)
            for j in range(ntime):
                rho_i[j] = fsnr(times[j]+dt)
            if verbose:
                Progress(cumitr, itr_tot)
                cumitr += ntime
            rho[:,k,i] = rho_i
    if verbose:
        sys.stderr.write('\r')
    return rho

def rho_vec_GWCOH(snr_set, ra, de, times):
    npix = len(ra)
    ntime = len(times)
    ndet = len(snr_set)
    rho = np.zeros([ntime,npix,ndet],np.complex)
    for (i,data) in enumerate(snr_set):
        snr_real = data.real
        snr_imag = data.imag
        rho_real = pg.SNR_time_pix(data.ifo, ra, de, times, snr_real, data.time)
        if data.value.dtype != complex or np.allclose(snr_imag, np.zeros(snr_imag.size)):
            rho_imag = np.zeros(rho_real.shape)
        else:
            rho_imag = pg.SNR_time_pix(data.ifo, ra, de, times, snr_imag, data.time)
        rho[:,:,i] = rho_real + 1.j * rho_imag
    return rho

# Time - Freq Coherent Matrix
def cohTF(sLIST, ra, de, psi, times, freqs, verbose = False, **kwargs):
    ntime = times.size
    nfreq = freqs.size
    ndet = len(sLIST)
    if verbose:
        sys.stderr.write('Calculating cohTF...\n')
        sys.stderr.write('--Calculating Gpc matrix...\n')
    if GWCOH:
        Gpc_matrix = np.zeros([ntime, nfreq, ndet, 2])
        for (i, data) in enumerate(sLIST):
            psd = data.psdfun(freqs)
            Gpc_matrix[:,:,i,:] = pg.Gpc_time_freq(data.ifo, ra, de, times, freqs, psd)
            
    else:
        Gpc_matrix = Gpc_tf(sLIST, ra, de, psi, times, freqs, verbose)
    if verbose:
        sys.stderr.write('--Calculating Gpc matrix: Done\n')
        sys.stderr.write('--Do SVD on Gpc matrix...')
    u,s,v = np.linalg.svd(Gpc_matrix)
    if verbose:
        sys.stderr.write('Done\n')
        sys.stderr.write('--Calculating rho_vec ...\n')
    rho = rhotf_vec(sLIST, ra, de, psi, times, freqs, verbose, **kwargs)
    if verbose:
        sys.stderr.write('--Calculating rho_vec: Done\n')
        sys.stderr.write('--Multiply u, rho...')
    coh = np.zeros([ntime, nfreq, ndet, ndet], np.complex)
    for i in range(ndet):
        coh[:,:,:,i] = np.multiply(u[:,:,:,i],rho[:,:,:])
    if verbose:
        sys.stderr.write('Done\n')
    coh = np.sum(coh,axis=2)
    if verbose:
        sys.stderr.write('Calculating cohTF: Complete\n')
    return coh

    

def Gpc_tf(sLIST, ra, de, psi, times, freqs, verbose = False):
    ntime = times.size
    nfreq = freqs.size
    ndet = len(sLIST)
    Gpc = np.zeros([ntime, nfreq, ndet, 2], np.float)
    if verbose:
        cumitr = 0
        itr_tot = nfreq * ndet * ntime
    for k, series in enumerate(sLIST):
        psdfun = series.psdfun
        for j in range(nfreq):
            for i in range(ntime):
                sigmaf = np.sqrt(psdfun(freqs[j]))
                ar = series.ifo_antenna_pattern(ra, de, psi, times[i])
                Gpc[i,j,k,0] = ar[0] / sigmaf
                Gpc[i,j,k,1] = ar[1] / sigmaf
            if verbose:
                cumitr += ntime
                Progress(cumitr, itr_tot)
    if verbose:
        sys.stderr.write('\r')
    return Gpc
    

def rhotf_vec(sLIST, ra, de, psi, times, freqs, verbose = False, **kwargs):
    ntime = times.size
    nfreq = freqs.size
    ndet = len(sLIST)
    rho = np.zeros([ntime, nfreq, ndet], np.complex)
    gps_time = times[int(ntime/2)]
    dt = np.zeros(3)
    for i,series in enumerate(sLIST):
        dt[i]  = series.ifo_delay(ra, de, gps_time)
        
    if verbose:
        cumitr = 0
        itr_tot = nfreq * ndet * ntime
    for i in range(ntime):
        for j in range(nfreq):
            for k, series in enumerate(sLIST):
                if not hasattr(series, "q_energyfunc"):
                    series.q_scan(outseg = times, foutseg = freqs, norm = True, ret_complex = True, **kwargs)
                rho[i,j,k] = series.q_energyfunc(times[i] + dt[k], freqs[j])
        if verbose:
            cumitr += ndet * nfreq
            Progress(cumitr, itr_tot)
    if verbose:
        sys.stderr.write('\r')
    return rho

# SNR - Time - Freq Coherent Matrix
def snr_cohTF(sLIST, ra, de, psi, times, freqs, tmpl, verbose = False, **kwargs):
    ntime = times.size
    nfreq = freqs.size
    ndet = len(sLIST)
    if verbose:
        sys.stderr.write('Calculating cohTF...\n')
        sys.stderr.write('--Calculating Gpc matrix...\n')
    if GWCOH:
        Gpc_matrix = np.zeros([ntime, nfreq, ndet, 2])
        for (i, data) in enumerate(sLIST):
            psd = np.ones(nfreq)
            Gpc_matrix[:,:,i,:] = pg.Gpc_time_freq(data.ifo, ra, de, times, freqs, psd) * np.sqrt(data.sigma2)
            
    else:
        Gpc_matrix = snr_Gpc_tf(sLIST, ra, de, psi, times, freqs, verbose)
    if verbose:
        sys.stderr.write('--Calculating Gpc matrix: Done\n')
        sys.stderr.write('--Do SVD on Gpc matrix...')
    u,s,v = np.linalg.svd(Gpc_matrix)
    if verbose:
        sys.stderr.write('Done\n')
        sys.stderr.write('--Calculating rho_vec ...\n')
    rho = snr_rhotf_vec(sLIST, ra, de, psi, times, freqs, tmpl, verbose, **kwargs)
    if verbose:
        sys.stderr.write('--Calculating rho_vec: Done\n')
        sys.stderr.write('--Multiply u, rho...')
    coh = np.zeros([ntime, nfreq, ndet, ndet], np.complex)
    for i in range(ndet):
        coh[:,:,:,i] = np.multiply(u[:,:,:,i],rho[:,:,:])
    if verbose:
        sys.stderr.write('Done\n')
    coh = np.sum(coh,axis=2)
    if verbose:
        sys.stderr.write('Calculating cohTF: Complete\n')
    return coh

def snr_Gpc_tf(sLIST, ra, de, psi, times, freqs, verbose = False):
    ntime = times.size
    nfreq = freqs.size
    ndet = len(sLIST)
    Gpc = np.zeros([ntime, nfreq, ndet, 2], np.float)
    if verbose:
        cumitr = 0
        itr_tot = nfreq * ndet * ntime
    for k, series in enumerate(sLIST):
        for j in range(nfreq):
            for i in range(ntime):
                ar = series.ifo_antenna_pattern(ra, de, psi, times[i])
                Gpc[i,j,k,0] = ar[0] * np.sqrt(series.sigma2)
                Gpc[i,j,k,1] = ar[1] * np.sqrt(series.sigma2)
            if verbose:
                cumitr += ntime
                Progress(cumitr, itr_tot)
    if verbose:
        sys.stderr.write('\r')
    return Gpc

def snr_rhotf_vec(sLIST, ra, de, psi, times, freqs, tmpl, verbose = False, **kwargs):
    ntime = times.size
    nfreq = freqs.size
    ndet = len(sLIST)
    rho = np.zeros([ntime, nfreq, ndet], np.complex)
    gps_time = times[int(ntime/2)]
    dt = np.zeros(3)
    for i,series in enumerate(sLIST):
        dt[i]  = series.ifo_delay(ra, de, gps_time)
        
    if verbose:
        cumitr = 0
        itr_tot = nfreq * ndet * ntime
    for i in range(ntime):
        for j in range(nfreq):
            for k, series in enumerate(sLIST):
                if not hasattr(series, "q_snrfunc"):
                    series.snr_q_scan(tmpl = tmpl, toutseg = times, foutseg = freqs, psd = 'set', **kwargs)
                rho[i,j,k] = series.q_snrfunc(times[i] + dt[k], freqs[j])
        if verbose:
            cumitr += ndet * nfreq
            Progress(cumitr, itr_tot)
    if verbose:
        sys.stderr.write('\r')
    return rho
