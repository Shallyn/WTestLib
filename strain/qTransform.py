#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:52:12 2019

@author: drizl
"""

import numpy as np
from math import (log, ceil, pi, isinf, exp)
import warnings
from scipy.interpolate import InterpolatedUnivariateSpline, interp2d, interp1d


#__all__ = ['QTiling', 'QPlane', 'QTile', 'QGram', 'q_scan']

DEFAULT_FRANGE = (0, float('inf'))
DEFAULT_MISMATCH = 0.2
DEFAULT_QRANGE = (4, 64)

class QObject(object):
    """Base class for Q-transform objects

    This object exists just to provide basic methods for all other
    Q-transform objects.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, duration, sampling, mismatch=DEFAULT_MISMATCH):
        self.duration = float(duration)
        self.sampling = float(sampling)
        self.mismatch = float(mismatch)

    @property
    def deltam(self):
        """Fractional mismatch between neighbouring tiles

        :type: `float`
        """
        return 2 * (self.mismatch / 3.) ** (1/2.)
    
class QBase(QObject):
    """Base class for Q-transform objects with fixed Q

    This class just provides a property for Q-prime = Q / sqrt(11)
    """
    def __init__(self, q, duration, sampling, mismatch=DEFAULT_MISMATCH):
        super(QBase, self).__init__(duration, sampling, mismatch=mismatch)
        self.q = float(q)

    @property
    def qprime(self):
        """Normalized Q `(q/sqrt(11))`
        """
        return self.q / 11**(1/2.)
    
class QTiling(QObject):
    def __init__(self, duration, sampling,
                 qrange=DEFAULT_QRANGE,
                 frange=DEFAULT_FRANGE,
                 mismatch=DEFAULT_MISMATCH):
        super(QTiling, self).__init__(duration, sampling, mismatch=mismatch)
        self.qrange = (float(qrange[0]), float(qrange[1]))
        self.frange = [float(frange[0]), float(frange[1])]

        qlist = list(self._iter_qs())
        if self.frange[0] == 0:  # set non-zero lower frequency
            self.frange[0] = 50 * max(qlist) / (2 * pi * self.duration)
        maxf = self.sampling / 2 / (1 + 11**(1/2.) / min(qlist))
        if isinf(self.frange[1]):
            self.frange[1] = maxf
        elif self.frange[1] > maxf:  # truncate upper frequency to maximum
            warnings.warn('upper frequency of %.2f is too high for the given '
                          'Q range, resetting to %.2f'
                          % (self.frange[1], maxf))
            self.frange[1] = maxf
            
    @property
    def qs(self):  # pylint: disable=invalid-name
        """Array of Q values for this `QTiling`

        :type: `numpy.ndarray`
        """
        return np.array(list(self._iter_qs()))
    

    def _iter_qs(self):
        """Iterate over the Q values
        """
        # work out how many Qs we need
        cumum = log(self.qrange[1] / self.qrange[0]) / 2**(1/2.)
        nplanes = int(max(ceil(cumum / self.deltam), 1))
        dq = cumum / nplanes  # pylint: disable=invalid-name
        for i in range(nplanes):
            yield self.qrange[0] * exp(2**(1/2.) * dq * (i + .5))

    def __iter__(self):
        """Iterate over this `QTiling`

        Yields a `QPlane` at each Q value
        """
        for q in self._iter_qs():
            yield QPlane(q, self.frange, self.duration, self.sampling,
                         mismatch=self.mismatch)
            
    def transform(self, fseries, epoch = None, **kwargs):
        weight = 1 + np.log10(self.qrange[1]/self.qrange[0]) / np.sqrt(2)
        nind, nplanes, peak, result = (0, 0, 0, None)
        # identify the plane with the loudest tile
        for plane in self:
            nplanes += 1
            nind += sum([1 + row.ntiles * row.deltam for row in plane])
            result = plane.transform(fseries, epoch = epoch, **kwargs)
            if result.peak['energy'] > peak:
                out = result
                peak = out.peak['energy']
        return (out, nind * weight / nplanes)



class QPlane(QBase):
    def __init__(self, q, frange, duration, sampling,
                 mismatch=DEFAULT_MISMATCH):
        super(QPlane, self).__init__(q, duration, sampling, mismatch=mismatch)
        self.frange = [float(frange[0]), float(frange[1])]

        if self.frange[0] == 0:  # set non-zero lower frequency
            self.frange[0] = 50 * self.q / (2 * pi * self.duration)
        if isinf(self.frange[1]):  # set non-infinite upper frequency
            self.frange[1] = self.sampling / 2 / (1 + 1/self.qprime)
        
    def __iter__(self):
        """Iterate over this `QPlane`

        Yields a `QTile` at each frequency
        """
        # for each frequency, yield a QTile
        for freq in self._iter_frequencies():
            yield QTile(self.q, freq, self.duration, self.sampling,
                        mismatch=self.mismatch)
    
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

    @property
    def frequencies(self):
        """Array of central frequencies for this `QPlane`

        :type: `numpy.ndarray`
        """
        return np.array(list(self._iter_frequencies()))
    
    def transform(self, fseries, norm=True, epoch=None):
        out = []
        for qtile in self:
            # get energy from transform
            ret = qtile.transform(fseries, norm=norm, epoch=epoch)
            out.append(ret)
        return QGram(self, out)


class QTile(QBase):
    def __init__(self, q, frequency, duration, sampling,
                 mismatch=DEFAULT_MISMATCH):
        super(QTile, self).__init__(q, duration, sampling, mismatch=mismatch)
        self.frequency = frequency
        
    @property
    def bandwidth(self):
        return 2 * pi ** (1/2.) * self.frequency / self.q
    
    @property
    def ntiles(self):
        tcum_mismatch = self.duration * 2 * pi * self.frequency / self.q
        return next_power_of_two(tcum_mismatch / self.deltam)
    
    @property
    def windowsize(self):
        return 2 * int(self.frequency / self.qprime * self.duration) + 1
    
    def _get_indices(self):
        half = int((self.windowsize - 1) / 2)
        return np.arange(-half, half + 1)
    
    def get_window(self):
        """Generate the bi-square window for this row

        Returns
        -------
        window : `numpy.ndarray`
        """
        # real frequencies
        wfrequencies = self._get_indices() / self.duration
        # dimensionless frequencies
        xfrequencies = wfrequencies * self.qprime / self.frequency
        # normalize and generate bi-square window
        norm = self.ntiles / (self.duration * self.sampling) * (
            315 * self.qprime / (128 * self.frequency)) ** (1/2.)
        return (1 - xfrequencies ** 2) ** 2 * norm

    def get_data_indices(self):
        """Returns the index array of interesting frequencies for this row
        """
        return np.round(self._get_indices() + 1 +
                           self.frequency * self.duration).astype(int)

    @property
    def padding(self):
        """The `(left, right)` padding required for the IFFT

        :type: `tuple` of `int`
        """
        pad = self.ntiles - self.windowsize
        return (int((pad - 1)/2.), int((pad + 1)/2.))
    
    def transform(self, fseries, norm=True, epoch=None):
        windowed = fseries[self.get_data_indices()] * self.get_window()
        # pad data, move negative frequencies to the end, and IFFT
        padded = np.pad(windowed, self.padding, mode='constant')
        wenergy = np.fft.ifftshift(padded)
        tdenergy = np.fft.ifft(wenergy)
        energy = tdenergy.real ** 2 + tdenergy.imag ** 2
        if norm:
            norm = norm.lower() if isinstance(norm, str) else norm
            if norm in (True, 'median'):
                energy /= np.median(energy)
            elif norm in ('mean',):
                energy /= np.mean(energy)
            else:
                raise ValueError("Invalid normalisation %r" % norm)

        dt = self.duration / tdenergy.size
        if isinstance(epoch, int) or isinstance(epoch, float):
            t0 = epoch
        else:
            t0 = 0
        return energy, (t0, dt), tdenergy



class QGram(object):
    def __init__(self, plane, energies):
        self.plane = plane
        self.energies = energies
        self.peak = self._find_peak()
        
    def _find_peak(self):
        peak = {'energy': 0, 'snr': None, 'time': None, 'frequency': None}
        for freq, energy_T in zip(self.plane.frequencies, self.energies):
            energy, Tls, datatf = energy_T
            t0, dt = Tls
            maxidx = energy.argmax()
            maxe = energy[maxidx]
            if maxe > peak['energy']:
                peak.update({
                    'energy': maxe,
                    'snr': (2 * maxe) ** (1/2.),
                    'time': t0 + dt * maxidx,
                    'frequency': freq,
                })
        return peak
    
    def interpolate(self, tres="<default>", fres="<default>", logf=False,
                    outseg=None, foutseg = None, retfunc = False, ret_complex = False):
        t0, dt = self.energies[0][1]
        if outseg is None or (not isinstance(outseg, list) and not isinstance(outseg, np.ndarray)):
            outseg = (t0 + dt, t0 + (self.energies[0][0].size-2) * dt)
        if tres == "<default>":
            if len(outseg) == 2:
                tres = abs(outseg[1] - outseg[0]) / 1000.
                xout = np.arange(*outseg, step=tres)
            else:
                xout = np.asarray(outseg)
        else:
            xout = np.arange(outseg[0], outseg[-1], step=tres)
        fout = self.plane.frequencies
        #print(self.plane.q)
        #nx = xout.size
        #ny = fout.size
        eout = []
        for i, row in enumerate(self.energies):
            energy = row[0]
            t0, dt = row[1]
            datatf = row[2]
            duration = dt * energy.size
            xrow = np.arange(t0, t0 + duration, dt)
            if ret_complex:
                interp = interp1d(xrow, datatf)
            else:
                interp = InterpolatedUnivariateSpline(xrow, energy)
            eout.append(interp(xout))
        eout = np.array(eout)
        if ret_complex:
            interp = interp2d_complex(xout, fout, eout, kind = 'cubic')
        else:
            interp = interp2d(xout, fout, eout, kind='cubic')
        if retfunc:
            return interp, (xout[0], xout[-1]), (fout[0], fout[-1])
        
        if not logf:
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

        else:
            if fres == "<default>":
                fres = 500
            minf = np.log10(self.plane.frange[0])
            maxf = np.log10(self.plane.frange[1])
            yout = np.logspace(minf, maxf, fres)
        
        zout = interp(xout, yout)
        return xout, yout, zout




#----------------------------------------------------#
class interp2d_complex(object):
    def __init__(self, x, y, z, kind = 'cubic'):
        zreal = z.real
        zimag = z.imag
        self._func_real = interp2d(x,y,zreal, kind = kind)
        self._func_imag = interp2d(x,y,zimag, kind = kind)
    def __call__(self, x, y):
        return self._func_real(x,y) + 1.j * self._func_imag(x,y)

class interp1d_complex(object):
    def __init__(self, x, y, w = None, bbox = [None]*2, k=3, ext = 0, check_finite = False):
        self._func_real = InterpolatedUnivariateSpline(x,y.real, 
                        w = w, bbox = bbox, k = k, ext = ext, check_finite = check_finite)
        self._func_imag = InterpolatedUnivariateSpline(x,y.imag, 
                        w = w, bbox = bbox, k = k, ext = ext, check_finite = check_finite)
        
    def __call__(self, x):
        return self._func_real(x) + 1.j*self._func_imag(x)
    
def next_power_of_two(x):
    return 2**(ceil(log(x, 2)))
        
        
def q_scan(data, time = None, psd = None,mismatch=DEFAULT_MISMATCH, qrange=DEFAULT_QRANGE,
           frange=DEFAULT_FRANGE, duration=None, sampling=None, window = False, outseg = None,
           **kwargs):
    nfft = data.size
    dataf = np.fft.rfft(data, n = nfft) / nfft
    
    if time is not None:
        duration = time[-1] - time[0]
        sampling = 1./(time[1] - time[0])
        epoch = time[0]
        if psd is not None:
            freq = np.fft.rfftfreq(n = nfft, d = 1./sampling)
            dataf /= psd(freq)
            
    else:
        epoch = None
        
    if window:
        dataf *= np.blackman(dataf.size)
    qgram, N = QTiling(duration, sampling, mismatch=mismatch, qrange=qrange,
                       frange=frange).transform(dataf, epoch = epoch, **kwargs)
    far = 1.5 * N * np.exp(-qgram.peak['energy']) / duration
    x,y,z = qgram.interpolate(outseg = outseg)
    if window:
        for i in range(y.size):
            z[i,:] *= np.blackman(x.size)
    return ((x,y,z), far)

def q_scanf(data, time, psd = None,mismatch=DEFAULT_MISMATCH, qrange=DEFAULT_QRANGE,
           frange=DEFAULT_FRANGE, duration=None, sampling=None, window = False, \
           outseg = None, foutseg = None, retfunc = False, ret_complex = False, \
           **kwargs):
    nfft = 2 * (data.size - 1)
    dataf = data / nfft
    
    duration = time[-1] - time[0]
    sampling = 1./(time[1] - time[0])
    epoch = time[0]
    if psd is not None:
        freq = np.fft.rfftfreq(n = nfft, d = 1./sampling)
        dataf /= np.sqrt(psd(freq))
                    
    if window:
        dataf *= np.blackman(dataf.size)
    qgram, N = QTiling(duration, sampling, mismatch=mismatch, qrange=qrange,
                       frange=frange).transform(dataf, epoch = epoch, **kwargs)
    far = 1.5 * N * np.exp(-qgram.peak['energy']) / duration
    if retfunc:
        return qgram.interpolate(outseg = outseg, foutseg = foutseg, retfunc = retfunc, ret_complex = ret_complex)
    x,y,z = qgram.interpolate(outseg = outseg, foutseg = foutseg, retfunc = retfunc, ret_complex = ret_complex)
    return ((x,y,z), far)

        
        