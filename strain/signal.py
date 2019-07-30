#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 10:46:58 2019

@author: drizl
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
from scipy import signal
from pathlib import Path


def whiten(strain, interp_psd, fs):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, 1./fs)
    df = freqs[1] - freqs[0]
    # freqs1 = np.linspace(0,2048.,Nt/2+1)
    
    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    #norm = 1./np.sqrt(fs/2)
    white_hf = hf / np.sqrt(interp_psd(np.abs(freqs)))# * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    sigmasq = 4 * (hf * hf.conjugate() / interp_psd(freqs)).sum() * df
    return white_ht, sigmasq

def get_psdfun(data, fs, NFFT = None, NOVL = None, window = False):
    data_psd, freqs = get_psd(data, fs = fs, NFFT = NFFT, window=window, NOVL=NOVL)
    return interp1d(freqs, data_psd)

def get_psd(data, fs, NFFT = None, NOVL = None, window = False):
    if NFFT is None:
        NFFT = 4*fs
    if window:
        psd_window = np.blackman(NFFT)
    else:
        psd_window = None
    # and a 50% overlap:
    if NOVL is None:
        NOVL = NFFT/2
    data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)
    return data_psd, freqs

def matched_filter(s, h, fs, psdfun = None, cut = None, window = True, ret_complex = False):
    if window:
        try:   
            dwindow = signal.tukey(h.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
        except: 
            dwindow = signal.blackman(h.size)     
    else:
        dwindow = 1
    stilde = np.fft.fft(s * dwindow) / fs
    htilde = np.fft.fft(h * dwindow) / fs
    datafreq = np.fft.fftfreq(h.size)*fs
    df = abs(datafreq[1] - datafreq[0])

    if psdfun is None:
        NFFT = 4*fs
        psd_window = np.blackman(NFFT)
        # and a 50% overlap:
        NOVL = NFFT/2
        data_psd, freqs = mlab.psd(s, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)
        power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
    else:
        power_vec = psdfun(np.abs(datafreq))
    
    sf = np.zeros(stilde.size, dtype = stilde.dtype)
    hf = np.zeros(htilde.size, dtype = htilde.dtype)

    if cut is not None:
        fmin, fmax = cut
        if fmin < min(abs(datafreq)):
            fmin = min(abs(datafreq))
        if fmax > max(abs(datafreq)):
            fmax = max(abs(datafreq))
        kmin_m = np.where( np.abs(datafreq - fmin) < df)[0][0]
        kmin_p = np.where( np.abs(datafreq + fmin) < df)[0][0]
        kmax_m = np.where( np.abs(datafreq - fmax) < df)[0][0]
        kmax_p = np.where( np.abs(datafreq + fmax) < df)[0][0]
        
        sf[kmin_m:kmax_m] = stilde[kmin_m:kmax_m]
        sf[kmin_p:kmax_p] = stilde[kmin_p:kmax_p]
        hf[kmin_m:kmax_m] = htilde[kmin_m:kmax_m]
        hf[kmin_p:kmax_p] = htilde[kmin_p:kmax_p]
    else:
        sf[:] = stilde[:]
        hf[:] = htilde[:]
    
    
    optimal = sf * hf.conjugate() / power_vec
    optimal_time = 2 * np.fft.ifft(optimal) * fs
    sigmasq = 1 * (hf * hf.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    sigma_time = optimal_time/sigma
    if ret_complex:
        return sigma_time, sigmasq.real
    else:
        return np.abs(sigma_time), sigmasq.real

def matched_filter_Real(s, h, fs, psdfun = None, cut = None, window = True):
    if window:
        try:   
            dwindow = signal.tukey(h.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
        except: 
            dwindow = signal.blackman(h.size)     
    else:
        dwindow = 1
    stilde = np.fft.fft(s * dwindow) / fs
    hrtilde = np.fft.fft(h.real * dwindow) / fs
    hitilde = np.fft.fft(h.imag * dwindow) / fs
    htilde = np.fft.fft(h * dwindow) / fs
    datafreq = np.fft.fftfreq(h.size, 1./fs)
    df = abs(datafreq[1] - datafreq[0])

    if psdfun is None:
        NFFT = 4*fs
        psd_window = np.blackman(NFFT)
        # and a 50% overlap:
        NOVL = NFFT/2
        data_psd, freqs = mlab.psd(s, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)
        power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
    else:
        power_vec = psdfun(np.abs(datafreq))
    
    if cut is not None:
        fmin, fmax = cut
        if fmin < min(abs(datafreq)):
            fmin = min(abs(datafreq))
        if fmax > max(abs(datafreq)):
            fmax = max(abs(datafreq))
        kmin = np.where( np.abs(datafreq - fmin) < df)[0][0]
        kmax = np.where( np.abs(datafreq - fmax) < df)[0][0]
        stilde[:kmin] = 0
        stilde[kmax:] = 0
        hrtilde[:kmin] = 0
        hrtilde[kmax:] = 0
        hitilde[:kmin] = 0
        hitilde[kmax:] = 0
    sig2 = 1 * (htilde * htilde.conjugate() / power_vec).sum() * df

    op_r = 2 * stilde * hrtilde.conjugate() / power_vec
    op_r_time = np.fft.ifft(op_r) * fs
    
    op_i = 2 * stilde * hitilde.conjugate() / power_vec
    op_i_time = np.fft.ifft(op_i) * fs
    op = (op_r_time + 1.j*op_i_time) / np.sqrt(np.abs(sig2))
    return op


def get_track(th, ht, epoch_peak, extra_index = 5):
    phi = np.unwrap(np.angle(ht))
    freq = np.gradient(phi) / np.gradient(th) / (2 * np.pi)
    idx_peak = np.argmax(np.abs(ht))
    lth = len(ht)
    idx_end = min(lth, idx_peak + extra_index)
    time = th - th[idx_peak] + epoch_peak
    return time[:idx_end], freq[:idx_end]

def check_increasing(y, eps = 1e-5):
    dy = y[:-1] - y[1:]
    if min(dy) <= eps:
        return False
    else:
        return True

def get_idx(x, var):
    return np.where(abs(x - var) == min(np.abs(x - var)))[0][0]

def resample(t, h, fs):
    dt_new = 1./fs
    t -= t[0]
    h_itp = interp1d(t, h)
    lth = t[-1] - t[0]
    t_new = np.arange(0, lth * 1.1, dt_new)
    idx_end = get_idx(t_new, lth)
    lth_new = len(t_new)
    h_new = np.zeros(lth_new, dtype = h.dtype)
    h_new[:idx_end] = h_itp(t_new[:idx_end])
    return t_new, h_new

def padinsert(a,b,length = None):
    if length is None:
        length = max(len(a), len(b))
    if length < max(len(a), len(b)):
        raise ValueError('pad length should not less than max(len(a), len(b))')
    ax = np.pad(a, (0, length - len(a)), mode = 'constant')
    bx = np.pad(b, (0, length - len(b)), mode = 'constant')
    return ax, bx

def cutinsert(a,b,cutpct = None):
    if cutpct is None or cutpct > 1 or cutpct < 0:
        cutpct = 0.5
    b_cut = b[:int(len(b)*cutpct)]
    return padinsert(a, b_cut)

def recwindow(M, pct = 0.1):
    lb = int(M * pct)
    if lb == 0:
        return np.ones(M)
    M1 = M - 2*lb
    one = np.ones(M1)
    bd_left = np.zeros(lb)
    bd_right = np.zeros(lb)
    for i in range(lb):
        bd_left[i] = 1 - ((i - lb) / lb)**2
        bd_right[i] = 1 - ((i + 1) / lb)**2
    return np.concatenate([bd_left, one, bd_right])

def plotit(time, data, 
           xrange, yrange, 
           title, xlabel, ylabel, 
           save = None, epoch = None, figsize = None):
    if figsize is None:
        plt.figure(figsize = (10,5))
    else:
        plt.figure(figsize = figsize)
    if epoch is not None:
        ex = [epoch, epoch]
        ey = [0,1]
        plt.plot(ex, ey, '--', alpha = 0.5, color = 'grey')
    plt.plot(time, data)
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save:
        plt.savefig(save, dpi = 200)
    plt.close()
    
def sngl_load_file(fdir, channel = 'CALIB'):
    fdir = Path(fdir)
    fout = []
    for file in fdir.iterdir():
        if channel in file.name.split('_'):
            fout.append(file)
    if len(fout) == 0:
        raise ValueError('No such channel data file')
    out = dict()
    for file in fout:
        for ifo in ['H1', 'L1', 'V1']:
            if ifo in file.name.split('_'):
                out[ifo] = file
    return out
    
