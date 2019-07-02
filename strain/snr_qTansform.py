#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:20:28 2019

@author: drizl
"""

import numpy as np
from math import (log, ceil, pi, isinf, exp)
from .qTransform import QTiling, QPlane, QTile, interp2d_complex, interp1d_complex
from .signal import get_psdfun, padinsert, cutinsert
from scipy import signal
import warnings, sys
import matplotlib.pyplot as plt

DEFAULT_FRANGE = (30, 1200)
DEFAULT_MISMATCH = 0.2
DEFAULT_QRANGE = (4, 64)


class snrQTiling(QTiling):
    def __init__(self,duration, sampling,
                 qrange=DEFAULT_QRANGE,
                 frange=DEFAULT_FRANGE,
                 mismatch=DEFAULT_MISMATCH):
        super(snrQTiling, self).__init__(duration = duration, 
                     sampling = sampling, 
                     qrange = qrange,
                     frange = frange,
                     mismatch=mismatch)
        
    def __iter__(self):
        """Iterate over this `QTiling`

        Yields a `QPlane` at each Q value
        """
        for q in self._iter_qs():
            yield snrQPlane(q, self.frange, self.duration, self.sampling,
                         mismatch=self.mismatch)
            
    def transform(self, stilde, htilde, psd, epoch = None, **kwargs):
        weight = 1 + np.log10(self.qrange[1]/self.qrange[0]) / np.sqrt(2)
        nind, nplanes, peak, result = (0, 0, 0, None)
        for plane in self:
            nplanes += 1
            nind += sum([1 + row.ntiles * row.deltam for row in plane])
            result = plane.transform(stilde, htilde, psd, epoch = epoch, **kwargs)
            if result.peak['energy'] > peak:
                out = result
                peak = out.peak['energy']
        return (out, nind * weight / nplanes)


            
            
class snrQTile(QTile):
    def __init__(self, q, frequency, duration, sampling,
                 mismatch=DEFAULT_MISMATCH):
        super(snrQTile, self).__init__(q = q, frequency = frequency,
                             duration = duration, sampling = sampling,
                             mismatch = mismatch)
    
    def get_snr_window(self, NFFT):
        window = np.zeros(NFFT)
        sngl_window = self.get_window()
        freq_indices = int(self.frequency * self.duration)
        window[self._get_indices() + int(NFFT/2) - freq_indices] = sngl_window
        window[self._get_indices() + int(NFFT/2) + freq_indices] = sngl_window
        return window
            
    def transform(self, stilde, htilde, psd, epoch = None, **kwargs):
        hwindowed = htilde * self.get_snr_window(htilde.size)
        sigmasq = 1 * (hwindowed * hwindowed.conjugate() / psd).sum() * self.sampling / htilde.size
        optimal = stilde * hwindowed.conjugate()
        optimal = np.fft.ifftshift(optimal)
        snr = 2 * np.fft.ifft(optimal) * self.sampling / np.sqrt(np.abs(sigmasq))
        return epoch, 1./self.sampling, snr
        


class snrQPlane(QPlane):
    def __init__(self, q, frange, duration, sampling,
                 mismatch=DEFAULT_MISMATCH):
        super(snrQPlane, self).__init__(q = q, frange = frange,
                                         duration = duration,
                                         sampling = sampling,
                                         mismatch = mismatch)

    def __iter__(self):
        """Iterate over this `QPlane`

        Yields a `QTile` at each frequency
        """
        # for each frequency, yield a QTile
        for freq in self._iter_frequencies():
            yield snrQTile(self.q, freq, self.duration, self.sampling,
                           mismatch=self.mismatch)
            
    def transform(self, stilde, htilde, psd, epoch=None):
        out = []
        for qtile in self:
            # get energy from transform
            ret = qtile.transform(stilde, htilde, psd, epoch=epoch)
            out.append(ret)
        return snrQGram(self, out)



class snrQGram(object):
    def __init__(self, plane, LIST):
        self.plane = plane
        self._LIST = LIST
        self.peak = self._find_peak()
        
    def _find_peak(self):
        peak = {'energy': 0, 'snr': None, 'time': None, 'frequency': None}
        for freq, qtile in zip(self.plane.frequencies, self._LIST):
            t0, dt, snr = qtile
            maxidx = np.abs(snr).argmax()
            maxe = np.abs(snr)[maxidx]
            if maxe > peak['energy']:
                peak.update({
                    'energy': maxe,
                    'snr': (2 * maxe) ** (1/2.),
                    'time': t0 + dt * maxidx,
                    'frequency': freq,
                })
        return peak
    
    def interpolate(self, tres="<default>", fres="<default>",
                        toutseg = None, foutseg = None, retfunc = False):
        t0 = self._LIST[0][0]
        dt = self._LIST[0][1]
        if toutseg is None or \
        (not isinstance(toutseg, list) and not isinstance(toutseg, np.ndarray)):
            toutseg = (t0, t0 + self._LIST[0][2].size * dt)
        if tres == "<default>":
            if len(toutseg) == 2:
                tres = abs(toutseg[1] - toutseg[0]) / 1000.
                xout = np.arange(*toutseg, step=tres)
            else:
                xout = np.asarray(toutseg)
        else:
            xout = np.arange(toutseg[0], toutseg[-1], step=tres)
        
        fout = self.plane.frequencies
        sout = []
        for i, row in enumerate(self._LIST):
            t0, dt, snr = row
            duration = dt * snr.size
            xrow = np.arange(t0, t0 + duration, dt)
            interp = interp1d_complex(xrow, snr)
            sout.append(interp(xout))
        sout = np.array(sout)
        interp = interp2d_complex(xout, fout, sout, kind = 'cubic')
        if retfunc:
            return interp, (xout[0], xout[-1]), (fout[0], fout[-1])
        minf = self.plane.frange[0]
        maxf = self.plane.frange[1]
        if foutseg is None or (not isinstance(foutseg, list) and not isinstance(foutseg, np.ndarray)):
            foutseg = (minf, maxf)

        if fres == "<default>":
            if len(foutseg) == 2:
                fres = (foutseg[-1] - foutseg[0]) / 500
                yout = np.arange(foutseg[0], foutseg[-1], fres)
            else:
                yout = np.asarray(foutseg)
        else:
            yout = np.arange(foutseg[0], foutseg[-1], fres)
            
        sout = interp(xout, yout)
        return xout, yout, sout

        


    


        
        




#----------------------------------------------#
def snr_q_scanf(data, ht,
                sampling, epoch, cut = None,
                psd = None,
                qrange = DEFAULT_QRANGE,
                frange = DEFAULT_FRANGE,
                mismatch = DEFAULT_MISMATCH,
                toutseg = None,
                foutseg = None,
                window = None,
                retfunc = False,
                **kwargs):
    
    Nt = data.size
    duration = Nt / sampling
    if len(ht) > Nt:
        s,h = cutinsert(data, ht)
    elif len(ht) < Nt:
        s,h = padinsert(data, ht)
        
    if window:
        try:   
            dwindow = signal.tukey(h.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
        except: 
            dwindow = signal.blackman(h.size)     
    else:
        dwindow = 1

    stilde = np.fft.fft(s * dwindow) / sampling
    htilde = np.fft.fft(h * dwindow) / sampling
    datafreq = np.fft.fftfreq(h.size) * sampling
    df = datafreq[1] - datafreq[0]
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
    
    freqshift = np.fft.fftshift(datafreq)
    if psd is None:
        psd = get_psdfun(data, sampling)
    power_vec = psd(np.abs(datafreq))
    psd_calc = psd(np.abs(freqshift))
    sf /= power_vec
    # Shift
    hf = np.fft.fftshift(hf)
    sf = np.fft.fftshift(sf)
    qgram, N = snrQTiling(duration, sampling, 
                          mismatch=mismatch, 
                          qrange=qrange,
                          frange=frange).transform(sf, hf, psd_calc, epoch = epoch, **kwargs)
    far = 1.5 * N * np.exp(-qgram.peak['energy']) / duration
    if retfunc:
        return qgram.interpolate(toutseg = toutseg, foutseg = foutseg, retfunc = retfunc)
    x,y,z = qgram.interpolate(toutseg = toutseg, foutseg = foutseg, retfunc = retfunc)
    return ((x,y,z), far)
