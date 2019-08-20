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
from scipy.interpolate import InterpolatedUnivariateSpline as fitp


DEFAULT_FRANGE = (20, 1200)
DEFAULT_MISMATCH = 0.2
DEFAULT_QRANGE = (4, 64)


class snrQTiling(QTiling):
    def __init__(self,duration, sampling,
                 qrange=DEFAULT_QRANGE,
                 frange=DEFAULT_FRANGE,
                 mismatch=DEFAULT_MISMATCH,
                 fdelay = None):
        super(snrQTiling, self).__init__(duration = duration, 
                     sampling = sampling, 
                     qrange = qrange,
                     frange = frange,
                     mismatch=mismatch)
        if fdelay is None:
            fdelay = lambda x : 0
        self._fdelay = fdelay
        
    def __iter__(self):
        """Iterate over this `QTiling`

        Yields a `QPlane` at each Q value
        """
        for q in self._iter_qs():
            yield snrQPlane(q, self.frange, self.duration, self.sampling,
                         mismatch=self.mismatch, fdelay = self._fdelay)
            
    def transform(self, stilde, hrtilde, hitilde, psd, epoch = None, **kwargs):
        weight = 1 + np.log10(self.qrange[1]/self.qrange[0]) / np.sqrt(2)
        nind, nplanes, peak, result = (0, 0, 0, None)
        for plane in self:
            nplanes += 1
            nind += sum([1 + row.ntiles * row.deltam for row in plane])
            result = plane.transform(stilde, hrtilde, hitilde, psd, epoch = epoch, **kwargs)
            if result.peak['energy'] > peak:
                out = result
                peak = out.peak['energy']
        return (out, nind * weight / nplanes)


            
            
class snrQTile(QTile):
    def __init__(self, q, frequency, duration, sampling,
                 mismatch=DEFAULT_MISMATCH, shift = 0):
        super(snrQTile, self).__init__(q = q, frequency = frequency,
                             duration = duration, sampling = sampling,
                             mismatch = mismatch)
        self._shift = shift
    
    def get_snr_window(self, NFFT):
        window = np.zeros(NFFT)
        sngl_window = self.get_window()
        freq_indices = int(self.frequency * self.duration)
        window[self._get_indices() + freq_indices] = sngl_window
        return window
        
    def get_window(self):
        wfrequencies = self._get_indices() / self.duration
        xfrequencies = wfrequencies * self.qprime / self.frequency
        # normalize and generate bi-square window
        norm = self.ntiles / (self.duration * self.sampling) * (
            315 * self.qprime / (128 * self.frequency)) ** (1/2.)
        return (1 - xfrequencies ** 2) ** 2 * norm

        # window = np.zeros(NFFT)
        # sngl_window = self.get_window()
        # freq_indices = int(self.frequency * self.duration)
        # window[self._get_indices() + int(NFFT/2) - freq_indices] = sngl_window
        # window[self._get_indices() + int(NFFT/2) + freq_indices] = sngl_window
        # return window
    @property
    def padding(self):
        """The `(left, right)` padding required for the IFFT

        :type: `tuple` of `int`
        """
        pad = self.ntiles - self.windowsize
        return (int((pad - 1)/2.), int((pad + 1)/2.))
            
    def transform(self, stilde, hrtilde, hitilde, psd, epoch = None):
        hrwindowed = hrtilde * self.get_snr_window(hrtilde.size)
        # hrwindowed = np.pad(hrwindowed, self.padding, mode = 'constant')
        hiwindowed = hitilde * self.get_snr_window(hitilde.size)
        # hiwindowed = np.pad(hiwindowed, self.padding, mode = 'constant')
        
        sigmasqr = 1 * (hrwindowed * hrwindowed.conjugate() / psd).sum() * self.sampling / hrtilde.size
        sigmasqi = 1 * (hiwindowed * hiwindowed.conjugate() / psd).sum() * self.sampling / hitilde.size
        
        op_r = 2 * stilde * hrwindowed.conjugate()
        op_r_t = np.fft.irfft(op_r) / np.sqrt(np.abs(sigmasqr))
        
        op_i = 2 * stilde * hiwindowed.conjugate()
        op_i_t = np.fft.irfft(op_i) / np.sqrt(np.abs(sigmasqi))
        
        snr = (op_r_t + 1.j*op_i_t) * self.sampling
        return epoch + self._shift, 1./self.sampling, snr
        


class snrQPlane(QPlane):
    def __init__(self, q, frange, duration, sampling,
                 mismatch=DEFAULT_MISMATCH, fdelay = None):
        super(snrQPlane, self).__init__(q = q, frange = frange,
                                         duration = duration,
                                         sampling = sampling,
                                         mismatch = mismatch)
        if fdelay is None:
            fdelay = lambda x : 0
        self._fdelay = fdelay
    
    
    def freq_timeshift(self, freq):
        return self._fdelay(freq)
    
    def __iter__(self):
        """Iterate over this `QPlane`

        Yields a `QTile` at each frequency
        """        # for each frequency, yield a QTile
        for freq in self._iter_frequencies():
            yield snrQTile(self.q, freq, self.duration, self.sampling,
                           mismatch=self.mismatch, shift = self.freq_timeshift(freq))

    def _iter_frequencies(self):
        minf, maxf = self.frange
        fcum_mismatch = log(maxf / minf) * (2 + self.q**2)**(1/2.) / 2.
        nfreq = int(max(1, ceil(fcum_mismatch / self.deltam)))
        fstep = fcum_mismatch / nfreq
        fstepmin = 1 / self.duration
        # for each frequency, yield a QTile
        for i in range(nfreq):
            yield (minf *
                   exp(2 / (2 + self.q**2)**(1/2.) * (i + .5) * fstep) //
                   fstepmin * fstepmin)


    def transform(self, stilde, hrtilde, hitilde, psd, epoch=None, **kwargs):
        out = []
        for qtile in self:
            # get energy from transform
            ret = qtile.transform(stilde, hrtilde, hitilde, psd, epoch=epoch, **kwargs)
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
        tini = 0
        tend = np.inf
        for qtile in self._LIST:
            t0, dt, snr = qtile
            t_final = t0 + dt*len(snr)
            if t0 > tini:
                tini = t0
            
            if t_final < tend:
                tend = t_final
            
            
        dt = self._LIST[0][1]
        if toutseg is None or \
        (not isinstance(toutseg, list) and not isinstance(toutseg, np.ndarray)):
            toutseg = (tini, tend)
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

        


    


        
        


from . import template
from .signal import check_increasing, plotit
#----------------------------------------------#
def snr_q_scanf(data, tmpl,
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
    if not isinstance(tmpl, template.template):
        raise TypeError('Type of variable tmpl should be strain.template.template')
    # track_x, track_y = tmpl.get_track(0, extra_index = 0)
    # track_x -= track_x[0]
    # func_freq_delay = fitp(track_y, track_x)
    func_freq_delay = tmpl.get_time_shift
    ht = tmpl.template
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

    stilde = np.fft.rfft(s * dwindow) / sampling
    hrtilde = np.fft.rfft(h.real * dwindow) / sampling
    hitilde = np.fft.rfft(h.imag * dwindow) / sampling
    datafreq = np.fft.rfftfreq(h.size) * sampling
    df = datafreq[1] - datafreq[0]

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
    
    if psd is None:
        psd = get_psdfun(data, sampling)
    power_vec = psd(np.abs(datafreq))
    stilde /= power_vec
    # sigmasq = 1 * (hrtilde * hrtilde.conjugate() / power_vec).sum() * sampling / hrtilde.size

    qgram, N = snrQTiling(duration, sampling, 
                          mismatch=mismatch, 
                          qrange=qrange,
                          frange=frange,
                          fdelay = func_freq_delay).transform(stilde, hrtilde, hitilde, power_vec, 
                                                  epoch = epoch,
                                                  **kwargs)
    far = 1.5 * N * np.exp(-qgram.peak['energy']) / duration
    if retfunc:
        return qgram.interpolate(toutseg = toutseg, foutseg = foutseg, retfunc = retfunc)
    x,y,z = qgram.interpolate(toutseg = toutseg, foutseg = foutseg, retfunc = retfunc)
    return ((x,y,z), far)
