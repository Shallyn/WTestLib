#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:41:46 2019

@author: drizl
"""

import numpy as np
import sys, os, json
from .Utils import switch, CEV, LOG, WARNING, CEV_parse_value, MESSAGE, plot_compare_attach_any, plot_marker, interp1d_complex
from .Utils import SpinWeightedM2SphericalHarmonic
from .h22datatype import dim_h, dim_t, h22base, h22_alignment, ModeBase, Mode_alignment
from pathlib import Path
from .generator import Generator, self_adaptivor
import csv,codecs,h5py
from .psd import DetectorPSD
from .SXSlist import *
from .NRPhenom import calculate_NRPeakParams

DEFAULT_SRCLOC = Path('/Users/drizl/Documents/2018/SEOBNRE/Program_Test/SXS_Data_txt')
DEFAULT_SRCLOC_ALL = Path('/Users/drizl/Documents/2018/SEOBNRE/Program_Test/SXS_Data')
DEFAULT_TABLE = Path('/Users/drizl/Documents/2018/SEOBNRE/Program_Test/table_data.json')
DEFAULT_SXS_PLOT_LINESTYLE = '-'
DEFAULT_SXS_PLOT_COLOR = 'red'
DEFAULT_SXS_ALPHA = 0.5
DEFAULT_SXS_LINEWIDTH = 2.0

DEFAULT_FIT_PLOT_LINESTYLE = '--'
DEFAULT_FIT_PLOT_COLOR = 'seagreen'
DEFAULT_FIT_ALPHA = 1.0 
DEFAULT_FIT_LINEWIDTH = 1.0

#-------------Load File--------------#
def loadSXSh5data(fname, modeL, modeM):
    f = h5py.File(fname, 'r')
    for key in f.keys():
        if (key != 'OutermostExtraction.dir'):
            dir_key = key
            break
    time = None
    hreal = None
    himag = None
    modeL = int(modeL)
    modeM = int(modeM)
    modekey = f'Y_l{modeL}_m{modeM}.dat'
    if modekey in f[dir_key]:
        Data = f[dir_key][modekey][:,:]
        time = Data[:,0]
        time -= time[0]
        hreal = Data[:,1]
        himag = Data[:,2]
    f.close()
    return time, hreal, himag

def loadSXStxtdata(SXSnum, srcloc):
    srcpath = Path(srcloc)
    filename = srcpath / f'BBH_{SXSnum}.txt'
    data = np.loadtxt(filename)
    return data[:,0], data[:,1], data[:,2]

def loadSXSdataLM(SXSnum, srcloc, modeL, modeM):
    srcloc = Path(srcloc)
    file = srcloc / f'BBH_{SXSnum}.h5'
    f = h5py.File(file,'r')
    for key in f.keys():
        if (key != 'OutermostExtraction.dir'):
            dir_key = key
            break
    time = None
    hreal = None
    himag = None
    modeL = int(modeL)
    modeM = int(modeM)
    modekey = f'Y_l{modeL}_m{modeM}.dat'
    if modekey in f[dir_key]:
        Data = f[dir_key][modekey][:,:]
        time = Data[:,0]
        time -= time[0]
        hreal = Data[:,1]
        himag = Data[:,2]
    f.close()
    return time, hreal, himag
    

#-----------------GET WAVEFORM PARAMETERS----------------#    
def get_SXS_parameters(SXSnum, pmsName, 
                             srcloc = DEFAULT_SRCLOC,
                             table = DEFAULT_TABLE,
                             **kwargs):
    wfp = SXSh22(SXSnum, srcloc, table, **kwargs)
    return wfp.__dict__[pmsName]

def GET_PARAMS_FROM_TABLE(SXSnum, table):
    with open(table,"r") as load_f:
        load_dict = json.load(load_f)
    SXSnum = 'SXS:BBH:'+SXSnum
    #return load_dict
    for x in load_dict:
        if x['alternative_names'] is SXSnum or SXSnum in x['alternative_names']:
            return x
    sys.stderr.write('Error: No such SXS in table: {}\n'.format(SXSnum))
    return CEV.PMS_ERROR


#------------------CLASS----------------#

class SXSObject(object):
    def __init__(self, SXSnum, table, verbose = False):
        self._SXSnum = SXSnum
        self._table = table
        if verbose:
            sys.stderr.write(f'{LOG}:Loading SXS parameters from table...\n')
        # Parse parameters from table
        pms = GET_PARAMS_FROM_TABLE(SXSnum, table)
        if isinstance(pms, CEV):
            self.CEV_STATE = pms
            if verbose:
                sys.stderr.write(f'{WARNING}Warning: Failed to load parameters of SXS ID: {SXSnum}\n')
        else:
            if verbose:
                sys.stderr.write(f'{LOG}:Loading succeed..Starting parsing\n')
            self.CEV_STATE = CEV.NORMAL
            self._load_pms(pms)
        if verbose:
            sys.stderr.write(f'{LOG}:Loading SXS parameters from table...Done\n')
            
            
    def _load_pms(self, pms):
        # mass ratio - q
        mratio = max(1, pms['initial_mass_ratio'])
        self.q = float('%.6f'%mratio)
        # spin
        spin1 = pms['initial_dimensionless_spin1']
        spin2 = pms['initial_dimensionless_spin2']
        self.s1x = float('%.6f'%spin1[0])
        self.s1y = float('%.6f'%spin1[1])
        self.s1z = float('%.6f'%spin1[2])
        self.s2x = float('%.6f'%spin2[0])
        self.s2y = float('%.6f'%spin2[1])
        self.s2z = float('%.6f'%spin2[2])
        self.spin1 = np.sqrt(self.s1x**2 + self.s1y**2 + self.s1z**2)
        self.spin2 = np.sqrt(self.s2x**2 + self.s2y**2 + self.s2z**2)
        self.Norb = pms['number_of_orbits']
        # eccentricity
        self.Sf_ini = pms['initial_orbital_frequency'] / np.pi
        # Relaxed Eccentricity
        try:
            self.ecc = pms['reference_eccentricity']
        except:
            self.ecc = pms['relaxed_eccentricity']
        # remnant
        self.final_mass = pms['remnant_mass']
        fspin = pms['remnant_dimensionless_spin']
        self.final_spin = np.sqrt(fspin[0]**2 + fspin[1]**2 + fspin[2]**2)
        self.final_spinx = fspin[0]
        self.final_spiny = fspin[1]
        self.final_spinz = fspin[2]
        fkv = pms['remnant_velocity']
        self.final_kick_v = np.sqrt(fkv[0]**2 + fkv[1]**2 + fkv[2]**2)
        self.final_kick_vx = fkv[0]
        self.final_kick_vy = fkv[1]
        self.final_kick_vz = fkv[2]
        self.initial_ADM_energy = pms['initial_ADM_energy']
        
    @property
    def mQ1(self):
        return self.q / (1 + self.q)

    @property
    def mQ2(self):
        return 1 / (1 + self.q)

    @property
    def chi1Vec(self):
        return np.array([self.s1x, self.s1y, self.s1z])

    @property
    def chi2Vec(self):
        return np.array([self.s2x, self.s2y, self.s2z])
    
    @property
    def is_prec(self):
        s1 = self.chi1Vec
        s2 = self.chi2Vec
        if s1[0] != 0 or s1[1] != 0 or s2[0] !=0 or s2[1] != 0:
            return True
        return False
    
    @property
    def zero_odd_mode(self):
        q = self.q
        s1 = self.chi1Vec
        s2 = self.chi2Vec
        J = s1+s2
        if q==1 and s1[2] == s2[2] and J[0] == 0 and J[1] == 0:
            return True
        return False
    
    @property
    def chiSVec(self):
        return 0.5*(self.chi1Vec + self.chi2Vec)
    
    @property
    def chiAVec(self):
        return 0.5 * (self.chi1Vec - self.chi2Vec)

    @property
    def spin1Vec(self):
        return self.chi1Vec * np.power(self.mQ1, 2)
    
    @property
    def spin2Vec(self):
        return self.chi2Vec * np.power(self.mQ2, 2)

    @property
    def chiKerr(self):
        return self.spin1Vec + self.spin2Vec

    @property
    def eta(self):
        return self.q / (1+self.q) / (1+self.q)

    @property
    def dm(self):
        return np.sqrt(1 - self.eta * 4)

    @property
    def SXSnum(self):
        return self._SXSnum

    @property
    def chiX(self):
        return self.chiSVec[-1] + self.chiAVec[-1] * self.dm / (1-self.eta*2)

    def get_NRPeakParams(self):
        eta = self.eta
        chiS = self.chiSVec[-1]
        chiA = self.chiAVec[-1]
        chiX = chiS + chiA * self.dm / (1-eta*2)
        return calculate_NRPeakParams(self.eta, chiX)
    
    def save_info(self, fname):
        with open(fname,'w') as f:
            f.write(f'#q {self.q}\n')
            f.write(f'#spin1x {self.s1x}\n')
            f.write(f'#spin1y {self.s1y}\n')
            f.write(f'#spin1z {self.s1z}\n')
            f.write(f'#spin2x {self.s2x}\n')
            f.write(f'#spin2y {self.s2y}\n')
            f.write(f'#spin2z {self.s2z}\n')
            f.write(f'#e0 {self.ecc}\n')
            f.write(f'#f_ini {self.Sf_ini}\n')
            f.write(f'#final mass {self.final_mass}\n')
            f.write(f'#final spinx {self.final_spinx}\n')
            f.write(f'#final spiny {self.final_spiny}\n')
            f.write(f'#final spinz {self.final_spinz}\n')
            f.write(f'#initial ADM energy {self.initial_ADM_energy}\n')
            f.write(f'final kick vx {self.final_kick_vx}\n')
            f.write(f'final kick vy {self.final_kick_vy}\n')
            f.write(f'final kick vz {self.final_kick_vz}\n')
        return

    def ecc_range(self):
        return parse_ecc(self.ecc, 0, 0.7)

def find_SXSh5(srcloc, SXSnum):
    ftmp = Path(srcloc) / f'BBH_{SXSnum}.h5'
    if ftmp.exists():
        return ftmp
    ftmp = Path(srcloc) / f'SXS_BBH_{SXSnum}' / 'rhOverM_Asymptotic_GeometricUnits_CoM.h5'
    if ftmp.exists():
        return ftmp
    raise Exception(f'Could not find waveform file of SXS:BBH:{SXSnum} in {srcloc}')


class SXSAllMode(SXSObject):
    def __init__(self, SXSnum, table = DEFAULT_TABLE, srcloc = DEFAULT_SRCLOC_ALL, cutpct = 0):
        super(SXSAllMode, self).__init__(SXSnum, table, verbose = False)
        #self._file = Path(srcloc) / f'BBH_{SXSnum}.h5'
        self._file = find_SXSh5(srcloc, SXSnum)
        self._core = waveform_mode_collector(cutpct)
        f = h5py.File(self._file,'r')
        for key in f.keys():
            if (key != 'OutermostExtraction.dir'):
                dir_key = key
                break
        for modekey in f[dir_key]:
            if modekey.split('.')[-1] != 'dat':
                continue
            Data = f[dir_key][modekey][:,:]
            l,m = self._parse_mode(modekey)
            time = Data[:,0]
            time -= time[0]
            hreal = Data[:,1]
            himag = Data[:,2]
            self._core.append_mode(time, hreal, himag, l, m)
        f.close()

    def get_timeSI(self, Mtotal):
        return self._core.get_timeSI(Mtotal)

    def _parse_mode(self, key):
        l = 0
        m = 0
        try:
            spt = key.split('.')[0].split('_')
            l = int(spt[1].split('l')[1])
            m = int(spt[2].split('m')[1])
        except:
            sys.stderr.write(f'{WARNING}:Failed to parse modekey:{key}, unrecgnized format.\n')
        return l,m

    def get_mode(self, l, m):
        return self._core.get_mode(l,m)

    def calculate_EnergyFlux(self):
        ret = np.zeros(len(self.time))
        for (l,m), mode in self._core:
            modeabs = np.abs(mode.dot)
            ret += np.power(modeabs,2)
        return ret / 16 / np.pi

    def calculate_AngularMomentumFlux(self):
        ret = np.zeros(len(self.time)) + 1.j*np.zeros(len(self.time))
        for (l,m), mode in self._core:
            modedot = mode.dot
            ret += 1.j * m * modedot * np.conjugate(mode)
        return np.real(ret) / 16 / np.pi


    @property
    def file(self):
        return self._file
    
    def __iter__(self):
        for ele in self._core:
            yield ele

    @property
    def time(self):
        return self._core.time

    def get_time_peak(self, l, m, Mtotal = None):
        return self._core.get_time_peak(l, m, Mtotal)

    def iter_modeL(self):
        for l in self._core.iter_modeL():
            yield l

    def dumpMode(self, l, m, fname = None):
        if fname is None:
            fname = f'Mode_{l}_{m}_{self._SXSnum}'
        mode = self.get_mode(l, m)
        mode.dump(fname)

    def dumpAllModes(self, prefix = None):
        if prefix is None:
            prefix = Path(f'AllModes_{self._SXSnum}/Mode_')
        if not prefix.parent.exists():
            prefix.parent.mkdir(parents = True)
        Prt = prefix.parent
        fP = prefix.name
        for (l, m), mode in self:
            fname = Prt / f'{fP}{l}_{m}.dat'
            mode.dump(fname)

    def mode_resample(self, time):
        return self._core.resample(time)

class waveform_mode_collector(object):
    def __init__(self, cutpct = 1):
        self._modes = {}
        self._time = None
        self._icut = cutpct
    
    def __len__(self):
        return len(self._modes)
    
    @property
    def size(self):
        return len(self._time)
    
    def __iter__(self):
        for key in self._modes:
            yield self._get_mode_lm(key), self._modes[key]
            
    def iter_modeL(self):
        modeLs = [self._get_mode_lm(x)[0] for x in self._modes]
        minL = min(modeLs)
        maxL = max(modeLs)
        for l in range(minL, maxL + 1):
            yield l
    
    @property
    def time(self):
        return self._time
    
    def get_timeSI(self, Mtotal = 40):
        if self._time is not None:
            return self._time / dim_t(Mtotal)
        else:
            return self.time
    
    def append_mode(self, time, real, imag, l, m):
        iscut = int(0.01*self._icut * len(time))
        time = time[iscut:]
        real = real[iscut:]
        imag = imag[iscut:]
        if self._time is None:
            self._time = np.asarray(time.copy())
        mode = np.asarray(real) + 1.j*np.asarray(imag)
        if not np.allclose(self._time, time):
            sys.stderr.write(f'{WARNING}:Length of time is not compatible.')
            func = interp1d_complex(time, mode)
            if (time[-1] - time[0]) > (self._time[-1] - self._time[0]):
                mode = func(self._time)
            else:
                idx_cut = np.where(np.abs(self._time - time[-1]) == np.min(np.abs(self._time - time[-1])))[0][0]
                mode = np.zeros(self._time.size, dtype = np.complex)
                mode[:idx_cut] = func(self._time[:idx_cut])
        # NOTICE
        # mode /= 2
        key = self._get_key(l, m)
        self._modes[key] = ModeBase(self._time, mode.real, mode.imag)
    
    def _get_key(self, l, m):
        return f'Y_{l}_{m}'
    
    def _get_mode_lm(self, key):
        if key not in self._modes:
            raise TypeError(f'Invalid key:{key}')
        splt = key.split('_')
        l = int(splt[1])
        m = int(splt[2])
        return (l,m)
    
    def get_mode_list(self):
        return [self._get_mode_lm(key) for key in self._modes]

    def get_mode(self, l, m):
        key = self._get_key(l,m)
        if key not in self._modes:
            # sys.stderr.write(f'{WARNING}:You shave not appended such mode ({l},{m})\n')
            return None
        return self._modes[key]
    
    def __del__(self):
        del self._modes
        del self._time

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, np.integer):
            ret = {'time':self._time[key]}
            for (l,m), mode in self:
                ret[self._get_key(l, m)] = mode[key]
            return ret
        return self._getslice(key)

    def _getslice(self, index):
        if index.start is not None and index.start < 0:
            raise ValueError(('Negative start index ({}) is not supported').format(index.start)) 
        out = waveform_mode_collector(0)  
        time_cut = self._time[index]
        for (l,m), mode in self:
            mode_cut = mode[index]
            out.append_mode(time_cut, mode_cut.real, mode_cut.imag, l, m)
        return out

    def get_time_peak(self, l, m, Mtotal = None):
        mode = self.get_mode(l,m)
        if Mtotal is not None:
            return self.get_timeSI(Mtotal)[np.argmax(np.abs(mode))]
        else:
            return self.time[np.argmax(np.abs(mode))]

    def resample(self, time):
        out = waveform_mode_collector(0)
        time_new = time - time[0]
        for (l, m), mode in self:
            mode_new = mode.interpolate(time)
            # amp = np.abs(mode_new)
            # phase = np.unwrap(np.angle(mode_new))
            # if m < 0:
            #     phase = -np.abs(phase - phase[0])
            # else:
            #     phase = np.abs(phase - phase[0])
            # mode_new = amp * np.exp(1.j*phase)
            out.append_mode(time_new, mode_new.real, mode_new.imag, l, m)
        return out

    def construct_hpc(self, iota, phic, modelist = None, phaseFrom0 = False):
        out = np.zeros(len(self._time)) + 1.j*np.zeros(len(self._time))
        if modelist is None:
            modelist = self.get_mode_list()
        for (l, m) in modelist:
            if m == 0:
                continue
            mode = self.get_mode(l, m)
            if mode is None:
                continue
            Y = SpinWeightedM2SphericalHarmonic(iota, phic, l, m)
            if phaseFrom0:
                modeAbsm = mode.amp * np.exp(-1.j*mode.phaseFrom0)
                if m < 0:
                    this_mode = np.conjugate(modeAbsm)
                    if l%2:
                        this_mode = -this_mode
                    out += np.conjugate(Y * this_mode)
                else:
                    out += np.conjugate(Y * modeAbsm)
            else:
                out += np.conjugate(Y * mode.value)
        return ModeBase(self.time, out.real, out.imag)

    def pad(self, pad_width, padmode, deltaT, **kwargs):
        out = waveform_mode_collector(0)
        length = None
        for (l, m), mode in self:
            mode_pad = np.pad(mode, pad_width, padmode, **kwargs)
            if length is None:
                length = len(mode_pad)
                time_pad = np.arange(self._time[0], self._time[0] + length * deltaT, deltaT)
                if len(time_pad) > length:
                    time_pad = time_pad[:length]
                elif len(time_pad) < length:
                    dl = length - len(time_pad)
                    time_pad = np.arange(self._time[0], self._time[0] + (length+dl) * deltaT, deltaT)
            out.append_mode(time_pad, mode_pad.real, mode_pad.imag, l, m)
        return out

def ModeC_alignment(modeA, modeB, deltaT = None):
    tA = modeA.time
    tB = modeB.time
    dtA = tA[1] - tA[0]
    dtB = tB[1] - tB[0]
    if dtA != dtB:
        if deltaT is not None:
            dt_final = deltaT
        elif dtA < dtB:
            dt_final = dtB
        else:
            dt_final = dtA
        tA_new = np.arange(tA[0], tA[-1], dt_final)
        tB_new = np.arange(tB[0], tB[-1], dt_final)
        modeA = modeA.resample(tA_new)
        modeB = modeB.resample(tB_new)
    else:
        tA_new = tA
        tB_new = tB
        dt_final = dtA
    wf22A = modeA.get_mode(2,2)
    wf22B = modeB.get_mode(2,2)
    if len(wf22A) == len(wf22B):
        return modeA, modeB
    ipeak_A = wf22A.argpeak
    ipeak_B = wf22B.argpeak
    if ipeak_A > ipeak_B:
        idx_A = ipeak_A - ipeak_B
        idx_B = 0
    else:
        idx_A = 0
        idx_B = ipeak_B - ipeak_A
    # tmove = (ipeak_A - ipeak_B) * dt_final
    modeA = modeA[idx_A:]
    modeB = modeB[idx_B:]
    # print(f'size: {modeA.size}, {modeB.size}')
    lenA = modeA.size
    lenB = modeB.size
    lenFinal = max(lenA, lenB)
    tail_A = lenFinal - lenA
    tail_B = lenFinal - lenB
    # print(f'tail: {tail_A}, {tail_B}')
    if tail_A > 0:
        modeA = modeA.pad((0,tail_A), 'constant', dt_final)
    if tail_B > 0:
        modeB = modeB.pad((0,tail_B), 'constant', dt_final)
    if modeA.size != modeB.size:
        sys.stderr.write(f'lenA = {modeA.size} != lenB = {modeB.size}\n')
        raise Exception('Error!')
    return modeA, modeB        

class SXSparameters(SXSObject):
    def __init__(self, SXSnum, table = DEFAULT_TABLE, f_ini = 0, Mtotal = 40, D = 100, verbose = False, ishertz = False):
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize SXSObject...\n')
        super(SXSparameters, self).__init__(SXSnum, table, verbose = verbose)
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize SXSObject...Done\n')
        self._D = D
        self._Mtotal = Mtotal
        self._f_ini = f_ini
        self._ishertz = ishertz
    
    @property
    def ecc_float(self):
        try:
            ecc = float(self.ecc)
        except:
            ecc = 0
        return ecc
    
    @property
    def D(self):
        return self._D
    
    @property
    def Mtotal(self):
        return self._Mtotal
    
    @property
    def m1(self):
        return self._Mtotal * self.q / (1 + self.q)
    
    @property
    def m2(self):
        return self._Mtotal / (1 + self.q)

    @property
    def f_ini(self):
        if self._f_ini <= 0:
            return self.Sf_ini * dim_t(self._Mtotal)
        else:
            if self._ishertz:
                return self._f_ini
            else:
                return self._f_ini * dim_t(self._Mtotal)
        
    @property
    def f_ini_dimless(self):
        return self.f_ini / dim_t(self.m1 + self.m2)

    def CalculateAdjParamsV4(self):
        return SEOBHyperCoefficients_v4(self.eta, self.chiKerr[2])


class SXSh22(SXSparameters, h22base):
    # def __new__(cls, SXSnum, srcloc = DEFAULT_SRCLOC, table = DEFAULT_TABLE , 
    #              f_ini = 0, srate = 16384, Mtotal = 16, D = 100, verbose = False):
    #     SXSparameters.__new__(cls, SXSnum, table, f_ini, Mtotal, D, verbose = verbose)
    
    def __init__(self, SXSnum, modeL = None, modeM = None,
                 srcloc = DEFAULT_SRCLOC, srcloc_all = DEFAULT_SRCLOC_ALL, 
                 table = DEFAULT_TABLE , 
                 f_ini = 0, srate = 16384, Mtotal = 40, D = 100, verbose = True, ishertz = False, cutpct = 0, ymode = 22):
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize SXSparameters...\n')
        SXSparameters.__init__(self, SXSnum, table, f_ini, Mtotal, D, verbose = verbose, ishertz = ishertz)
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize SXSparameters...Done\n')
            sys.stderr.write(f'{LOG}:Loading SXS template from srcloc...')
        self._modeL = modeL
        self._modeM = modeM
        if modeL is not None and modeM is not None:
            t, hr, hi = loadSXSdataLM(SXSnum, srcloc_all, modeL, modeM)
            if t is None:
                raise ValueError(f'Mode {modeL} {modeM} is not available.')
        else:
            try:
                t, hr, hi = loadSXStxtdata(SXSnum, srcloc)
            except:
                fh5 = find_SXSh5(srcloc_all, SXSnum)
                ymodel = 2
                ymodem = 2
                if ymode != 22:
                    ymodel = int(ymode / 10)
                    ymodem = ymode % 10
                t, hr, hi = loadSXSh5data(fh5, ymodel, ymodem)
        icut = int(len(t) * cutpct)
        t = t[icut:]
        hr = hr[icut:]
        hi = hi[icut:]
        self._rawData = ModeBase(t.copy(), hr, hi)
        t /= dim_t(Mtotal)
        self._srcloc = srcloc
        self._srcloc_all = srcloc_all
        if verbose:
            sys.stderr.write('Done\n')
            sys.stderr.write(f'{LOG}:Initialize h22base...\n')
        h22base.__init__(self, t, hr, hi, srate, verbose = verbose) 
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize h22base...Done\n')
        self._verbose = verbose
    
    @property
    def rawData(self):
        return self._rawData
    
    def cut_ringdown(self):
        mode = self._mode
        time = self._time
        indpeak = self.argpeak
        return h22base(time[:indpeak], mode.real[:indpeak], mode.imag[:indpeak], self.srate)
        
    def construct_generator(self, approx, executable, psd = None):
        return SXSCompGenerator(approx, executable, self, psd = psd,
                                modeL = self._modeL, modeM = self._modeM, 
                                verbose = self._verbose)
    
    def copy(self):
        return SXSh22(self._SXSnum, modeL = self._modeL, modeM = self._modeM,
                      srcloc = self._srcloc, srcloc_all = self._srcloc_all,
                      table = self._table, f_ini = self._f_ini, 
                      srate = self._srate, Mtotal = self._Mtotal, D = self._D,
                      verbose = False)
        
    def get_h22(self):
        return h22base(self.time, self.real, self.imag, self.srate)
    
    @property
    def dim_t(self):
        return dim_t(self._Mtotal)
    
    @property
    def dim_h(self):
        return dim_h(self._D, self._Mtotal)
    
    @property
    def duration_dimM(self):
        return self.duration * self.dim_t
        
    
        
class SXSCompGenerator(Generator):
    def __init__(self, approx, executable, sxsh22, psd = None, modeL = None, modeM = None,verbose = False):
        #print(isinstance(sxsh22, SXSh22))
        if not isinstance(sxsh22, SXSh22):
            raise TypeError('Incorrect input type: {}'.format(type(sxsh22)))
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize Generator...\n')
        super(SXSCompGenerator, self).__init__(approx, executable)
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize Generator...Done\n')
        self._core = sxsh22
        self._verbose = verbose
        self._modeL = modeL
        self._modeM = modeM
        self._psd = psd
        if self._modeL is None:
            self._modeL = 2
        if self._modeM is None:
            self._modeM = 2
        self._fecc = lambda ecc : self._CMD(m1 = self._core.m1,
                                    m2 = self._core.m2,
                                    s1z = self._core.s1z,
                                    s2z = self._core.s2z,
                                    D = self._core.D,
                                    ecc = ecc,
                                    srate = self._core.srate,
                                    f_ini = self._core.f_ini,
                                    L = self._modeL,
                                    M = self._modeM)
    
    @property
    def SXS(self):
        return self._core
        
    def get_CMD(self, ecc = None, **kwargs):
        if self._core.CEV_STATE != CEV.NORMAL:
            sys.stderr.write(f'{WARNING}:Abnormal SXSh22...\n')
            return self._core.CEV_STATE
        if ecc is None:
            ecc = 0
        if ecc is not None and not self.allow_ecc:
            sys.stderr.write(f'{WARNING}: parameter ecc is unused.\n')
        ret = self.call(m1 = self._core.m1,
                            m2 = self._core.m2,
                            s1z = self._core.s1z,
                            s2z = self._core.s2z,
                            D = self._core.D,
                            ecc = ecc,
                            srate = self._core.srate,
                            f_ini = self._core.f_ini,
                            L = self._modeL,
                            M = self._modeM,
                            **kwargs)
        return ret

    
    def get_waveform(self, ecc = None, 
                     jobtag = 'test',
                     timeout = 60,
                     Mtotal = None,
                     verbose = None,
                     fini = None,
                     **kwargs):
        if verbose is None:
            verbose = self._verbose
        if verbose:
            sys.stderr.write(f'{LOG}:Checking SXSh22 status.\n')
        if self._core.CEV_STATE != CEV.NORMAL:
            sys.stderr.write(f'{WARNING}:Abnormal SXSh22...\n')
            return self._core.CEV_STATE
        if verbose:
            sys.stderr.write(f'{LOG}:Checking SXSh22 status...Normal\n')
        if ecc is None:
            ecc = 0
        if ecc is not None and not self.allow_ecc:
            sys.stderr.write(f'{WARNING}: parameter ecc is unused.\n')
        # add prec
        kwargs['s1x'] = self._core.s1x
        kwargs['s1y'] = self._core.s1y
        kwargs['s2x'] = self._core.s2x
        kwargs['s2y'] = self._core.s2y
        if verbose:
            sys.stderr.write(f'{LOG}:Calling Generator to generate waveform...\n')
            sys.stderr.write(f'{LOG}:CMD:{self.get_CMD(ecc = ecc, **kwargs)}\n')
        if Mtotal is None or hasattr(Mtotal, '__len__'):
            m1 = self._core.m1
            m2 = self._core.m2
        else:
            m1 = Mtotal * self._core.q / (1 + self._core.q)
            m2 = Mtotal / (1 + self._core.q)
        if fini is None:
            fini = self._core.f_ini
        ret = self.__call__(m1 = m1,
                            m2 = m2,
                            s1z = self._core.s1z,
                            s2z = self._core.s2z,
                            D = self._core.D,
                            ecc = ecc,
                            srate = self._core.srate,
                            f_ini = fini,
                            L = self._modeL,
                            M = self._modeM,
                            jobtag = jobtag,
                            timeout = timeout,
                            **kwargs)

        if verbose:
            sys.stderr.write(f'{LOG}:Generator runs out, check the results...\n')
        if isinstance(ret, CEV):
            if verbose:
                sys.stderr.write(f'{WARNING}:Fail to generate waveform...terminate\n')
            return ret
        if verbose:
            sys.stderr.write(f'{LOG}:Generation succeed, construct mode data.\n')
        t, hr, hi = ret[:,0], ret[:,1], ret[:,2]
        if verbose:
            sys.stderr.write(f'{LOG}:Pretreatment.\n')
        t, hr, hi = self._pretreat(t, hr, hi, self._core._D, self._core._Mtotal)
        h22_wf = h22base(t, hr, hi, self._core._srate)
        return h22_wf

    def get_lnprob(self, jobtag = 'test', no_RD = False, **kwargs):
        h22_wf = self.get_waveform(jobtag = jobtag, verbose = True, **kwargs)
        if isinstance(h22_wf, CEV):
            return -np.inf, -1
        if no_RD:
            NR = self._core.cut_ringdown()
        else:
            NR = self._core.copy()
        psdfunc = self._psd
        # Check sample rate
        Mtotal_init = self._core.Mtotal
        Mtotal_list = (10, 40, 70, 100, 130, 160, 190)
        wf_1, wf_2, _ = h22_alignment(NR, h22_wf)
        fs = wf_1.srate
        NFFT = len(wf_1)
        df_old = fs/NFFT
        htilde_1 = wf_1.h22f
        htilde_2 = wf_2.h22f

        idxPeak = min(wf_1.argpeak, wf_2.argpeak)
        idx_start = int(0.1*idxPeak)
        idx_end = int(0.9*idxPeak)

        trange = (wf_1.time[idx_end] - wf_1.time[idx_start]) * dim_t(Mtotal_init)
        dPhiCum = (wf_1.phase[idx_start:idx_end] - wf_1.phase[idx_start]) - (wf_2.phase[idx_start:idx_end] - wf_2.phase[idx_start])
        dAmpCum = (wf_1.amp[idx_start:idx_end] - wf_2.amp[idx_start:idx_end]) / wf_1.amp[idx_start:idx_end]
        if trange == 0:
            return -np.inf, -1
        dPhiCum = np.sum(np.power(dPhiCum, 2))
        dAmpCum = np.sum(np.power(dAmpCum, 2))
        lnprob = []
        FF = []
        # eps_lst = []
        for Mtotal in Mtotal_list:
            dimt = dim_t(Mtotal)
            df = df_old *  Mtotal_init / Mtotal
            fs = df * NFFT
            freqs = np.abs(np.fft.fftfreq(NFFT, 1./fs))
            power_vec = psdfunc(freqs)
            O11 = np.sum(htilde_1 * htilde_1.conjugate() / power_vec).real * df
            O22 = np.sum(htilde_2 * htilde_2.conjugate() / power_vec).real * df
            Ox = htilde_1 * htilde_2.conjugate() / power_vec
            Oxt = np.fft.ifft(Ox) * fs / np.sqrt(O11 * O22)
            Oxt_abs = np.abs(Oxt)
            idx = np.where(Oxt_abs == max(Oxt_abs))[0][0]
            lth = len(Oxt_abs)
            if idx == lth-1 or idx == 1:
                tc = 0
            else:
                if idx > lth / 2:
                    tc = (idx - lth) / fs
                else:
                    tc = idx / fs
            eps = 1 - max(Oxt_abs)
            FF.append(1-eps)
            tc_dephase = tc * dimt
            lnprob.append(-( pow(eps/0.01, 2) + pow(tc_dephase/5, 2)))
            # lnprob.append( -(pow(eps/0.01, 2)))
            # lnprob.append(eps)
            # eps_lst.append(eps)
        # return max(eps_lst)
        return min(lnprob), min(FF)
        

    
    def get_overlap(self, jobtag = 'test', minecc = 0, maxecc = 0, **kwargs):
        if self._verbose:
            sys.stderr.write(f'{LOG}:Checking ecc is allowed or not.\n')
        eccentricity = kwargs.get('ecc')
        preset = kwargs.get('Preset')
        if not self.allow_ecc or (minecc == 0 and maxecc == 0 and preset is not True) or eccentricity is not None:
            if self._verbose:
                sys.stderr.write(f'{LOG}:ecc is unused in approx: {self._approx}, now calculate overlap.\n')
            fini = kwargs.get('fini')
            Mtotal = kwargs.get('Mtotal')
            h22_wf = self.get_waveform(jobtag = jobtag, fini = fini, **kwargs)
            if hasattr(Mtotal, '__len__'):
                ret = self.__core_calculate_overlap_MtotalList(h22_wf, MtotalList = Mtotal, verbose = self._verbose)
                return ret
            else:
                ret = self.__core_calculate_overlap(h22_wf, Mtotal = Mtotal)
                wraper = [np.array([0]),np.asarray([ret])]
        else:
            if self._verbose:
                sys.stderr.write(f'{LOG}:ecc is allowed in approx: {self._approx}, now run self-adaptivor.\n')
            wraper = self.__core_scan_ecc_overlap(jobtag = jobtag, minecc = minecc, maxecc = maxecc, **kwargs)
        if self._verbose:
            sys.stderr.write(f'{LOG}:Calculation complete, construct result processor.\n')
        return CompResults(self, wraper, self._verbose, jobtag = jobtag)
        
    def __core_calculate_overlap(self, h22_wf, verbose = None, Mtotal = None):
        if verbose is None:
            verbose = self._verbose
        if verbose:
            sys.stderr.write(f'{LOG}:Checking input mode status...\n')
        if isinstance(h22_wf, CEV):
            if verbose:
                sys.stderr.write(f'{WARNING}:Abnormal mode...\n')
            return 0,0,-1,0,h22_wf.value
        SXS = self._core.copy()
        # Check sample rate
        if verbose:
            sys.stderr.write(f'{LOG}:Align data for comparison...\n')
        SXS, h22_wf, tmove = h22_alignment(SXS, h22_wf)
        if verbose:
            sys.stderr.write('Done\n')
            sys.stderr.write(f'{LOG}:Calculating overlap...')
        fs = SXS.srate
        NFFT = len(SXS)
        df = fs/NFFT
        if Mtotal is not None:
            df = df * Mtotal / self._core.Mtotal
            fs = df * NFFT
        freqs = np.abs(np.fft.fftfreq(NFFT, 1./fs))
        power_vec = self._psd(freqs)
        Stilde = SXS.h22f
        htilde = h22_wf.h22f
        O11 = np.sum(Stilde * Stilde.conjugate() / power_vec).real * df
        O22 = np.sum(htilde * htilde.conjugate() / power_vec).real * df
        Ox = Stilde * htilde.conjugate() / power_vec
        Oxt = np.fft.ifft(Ox) * fs
        Oxt_abs = np.abs(Oxt) / np.sqrt(O11 * O22)
        idx = np.where(Oxt_abs == max(Oxt_abs))[0][0]
        lth = len(Oxt_abs)
        if idx > lth / 2:
            tc = (idx - lth) / fs
        else:
            tc = idx / fs
        phic = np.angle(Oxt[idx])
        if verbose:
            sys.stderr.write('Done\n')
        return tc, phic, Oxt_abs[idx], tmove, CEV.SUCCESS.value
    
    def __core_calculate_overlap_MtotalList(self, h22_wf, MtotalList = None, verbose = None):
        if verbose is None:
            verbose = self._verbose
        if verbose:
            sys.stderr.write(f'{LOG}:Checking input mode status...\n')
        if MtotalList is None:
            MtotalList = (10, 40, 70, 100, 130, 160, 190)
        ret_tc = np.zeros(len(MtotalList))
        ret_phic = np.zeros(len(MtotalList))
        ret_FF = np.zeros(len(MtotalList)) - 1
        ret_tmove = np.zeros(len(MtotalList))

        if isinstance(h22_wf, CEV):
            if verbose:
                sys.stderr.write(f'{WARNING}:Abnormal mode...\n')
            return ret_tc, ret_phic, ret_FF, ret_tmove
        SXS = self._core.copy()
        # Check sample rate
        if verbose:
            sys.stderr.write(f'{LOG}:Align data for comparison...\n')
        SXS, h22_wf, tmove = h22_alignment(SXS, h22_wf)
        if verbose:
            sys.stderr.write('Done\n')
            sys.stderr.write(f'{LOG}:Calculating overlap...')
        fs = SXS.srate
        NFFT = len(SXS)
        df_old = fs/NFFT
        Stilde = SXS.h22f
        htilde = h22_wf.h22f
        for i, Mtotal in enumerate(MtotalList):
            print(f'{Mtotal}/{max(MtotalList)}')
            df = df_old *  self._core.Mtotal / Mtotal
            fs = df * NFFT
            freqs = np.abs(np.fft.fftfreq(NFFT, 1./fs))
            power_vec = self._psd(freqs)
            if np.isinf(np.min(power_vec)):
                ret_tc[i] = 0
                ret_phic[i] = 0
                ret_FF[i] = 2
                ret_tmove[i] = tmove
                continue
            O11 = np.sum(Stilde * Stilde.conjugate() / power_vec).real * df
            O22 = np.sum(htilde * htilde.conjugate() / power_vec).real * df
            Ox = Stilde * htilde.conjugate() / power_vec
            Oxt = np.fft.ifft(Ox) * fs
            Oxt_abs = np.abs(Oxt) / np.sqrt(O11 * O22)
            idx = np.where(Oxt_abs == max(Oxt_abs))[0][0]
            lth = len(Oxt_abs)
            if idx > lth / 2:
                tc = (idx - lth) / fs
            else:
                tc = idx / fs
            phic = np.angle(Oxt[idx])
            ret_tc[i] = tc
            ret_phic[i] = phic
            ret_FF[i] = Oxt_abs[idx]
            ret_tmove[i] = tmove
        if verbose:
            sys.stderr.write('Done\n')
        return ret_tc, ret_phic, ret_FF, ret_tmove


    def __core_scan_ecc_overlap(self, estep = 0.02, maxitr = None, verbose = False,
                                prec_x = 1e-6, prec_y = 1e-6, jobtag = 'test', Mtotal = None,
                                minecc = 0, maxecc = 0, timeout = 60, Preset = False, scan_mtotal = False, **kwargs):
        # Parse ecc
        if self._verbose:
            sys.stderr.write(f'{LOG}:Parsing ecc...')
        if Preset:
            Preset = self.SXS._SXSnum
        else:
            Preset = None
        ecc_range = parse_ecc(self._core.ecc, minecc = minecc, maxecc = maxecc, Preset = Preset)
        if self._verbose:
            sys.stderr.write('Done: Ecc Range: {}\n'.format(ecc_range))
            sys.stderr.write(f'{LOG}:Construct self-adaptivor...')
        if ecc_range[1] - ecc_range[0] < estep * 10:
            estep = (ecc_range[1] - ecc_range[0]) / 10
        if scan_mtotal:
            def ecc_wf(ecc):
                h22_wf = self.get_waveform(ecc = ecc, verbose = False, jobtag = jobtag, timeout = timeout)
                tcL, phicL, FFL, tmoveL = self.__core_calculate_overlap_MtotalList(h22_wf, verbose = False)
                idx = np.argmin(FFL)
                if FFL[idx] < 0:
                    status = CEV.GEN_FAIL
                else:
                    status = CEV.SUCCESS
                return tcL[idx], phicL[idx], FFL[idx], tmoveL[idx], status.value
        else:
            def ecc_wf(ecc):
                h22_wf = self.get_waveform(ecc = ecc, verbose = False, jobtag = jobtag, timeout = timeout)
                ret = self.__core_calculate_overlap(h22_wf, verbose = False, Mtotal = Mtotal)
                return ret
        SA = self_adaptivor(ecc_wf, ecc_range, estep, outindex = 2, verbose = verbose)
        if self._verbose:
            sys.stderr.write('Done\n')
            sys.stderr.write(f'{LOG}:Run self-adaptivor...\n')
        return SA.run(maxitr = maxitr, 
                      verbose = self._verbose,
                      prec_x = prec_x, 
                      prec_y = prec_y)

class CompResults(object):
    def __init__(self, generator, results, verbose = False, jobtag = 'test'):
        self._core = generator
        self._results = results
        self._verbose = verbose
        self._jobtag = jobtag
        if self._verbose:
            sys.stderr.write(f'{LOG}:Parsing results...')
        self._parse_results()
        if self._verbose:
            sys.stderr.write('Done\n')
            sys.stderr.write(f'{LOG}:Parsing waveform...\n')
        self._get_fit_waveform()
        if self._verbose:
            sys.stderr.write(f'{LOG}:Parsing waveform...Done\n')
    
    def _parse_results(self):
        ecc, olpout = self._results
        self._ecc = ecc
        self._tc = olpout[:,0]
        self._phic = olpout[:,1]
        self._FF = olpout[:,2]
        self._tmove = olpout[:,3]
        self._CEV_STATE = CEV_parse_value(olpout[:,4].astype(np.int))
        fitarg = self._FF.argmax()
        self._max_FF = self._FF[fitarg]
        self._fit_tc = self._tc[fitarg]
        self._fit_phic = self._phic[fitarg]
        self._fit_STATE = self._CEV_STATE[fitarg]
        self._fit_ecc = self._ecc[fitarg]
        self._fit_tmove = self._tmove[fitarg]
    
    def plot_waveform_fit(self, fname, 
                          FIT_linestyle = None, FIT_color = None, 
                          FIT_alpha = None, FIT_linewidth = None,
                          SXS_linestyle = None, SXS_color = None, 
                          SXS_alpha = None, SXS_linewidth = None,
                          **kwargs):
        if self.CEV_STATE_fit is not CEV.SUCCESS and self._h22_fit is not None:
            sys.stderr.write(f'{WARNING}:State is not success, skip waveform plotting.\n')
        else:
            filename = fname
            if self._verbose:
                sys.stderr.write(f'{LOG}:Plotting waveform, name as {filename}\n')

            if FIT_linestyle is None:
                FIT_linestyle = DEFAULT_FIT_PLOT_LINESTYLE
            if FIT_color is None:
                FIT_color = DEFAULT_FIT_PLOT_COLOR
            if FIT_alpha is None:
                FIT_alpha = DEFAULT_FIT_ALPHA
            if FIT_linewidth is None:
                FIT_linewidth = DEFAULT_FIT_LINEWIDTH
    
            if SXS_linestyle is None:
                SXS_linestyle = DEFAULT_SXS_PLOT_LINESTYLE
            if SXS_color is None:
                SXS_color = DEFAULT_SXS_PLOT_COLOR
            if SXS_alpha is None:
                SXS_alpha = DEFAULT_SXS_ALPHA
            if SXS_linewidth is None:
                SXS_linewidth = DEFAULT_SXS_LINEWIDTH
    
    
            SXSplot = dict()
            FITplot = dict()
            SXSplot['x'] = self.generator.SXS.time
            SXSplot['y'] = self.generator.SXS.real
            SXSplot['name'] = self.generator.SXS.SXSnum
            SXSplot['linestyle'] = SXS_linestyle
            SXSplot['color'] = SXS_color
            SXSplot['alpha'] = SXS_alpha
            SXSplot['linewidth'] = SXS_linewidth
            
            FITplot['x'] = self.h22_fit.time
            FITplot['y'] = self.h22_fit.real
            FITplot['name'] = self.generator.approx
            FITplot['linestyle'] = FIT_linestyle
            FITplot['color'] = FIT_color
            FITplot['alpha'] = FIT_alpha
            FITplot['linewidth'] = FIT_linewidth
    
            
            plot_compare_attach_any([SXSplot, FITplot], savefig = filename, tstart=0, **kwargs)
        
    
    def plot_results(self, fname, **kwargs):
        if not self.generator.allow_ecc:
            sys.stderr.write(f'{WARNING}:Parameter ecc is unused, skip sa plot.\n')
        else:
            filename = fname
            if self.CEV_STATE_fit is not CEV.SUCCESS:
                sys.stderr.write(f'{WARNING}:State is not success, skip sa plot.\n')
            else:
                if self._verbose:
                    sys.stderr.write(f'{LOG}:Plotting sa results, name as {filename}\n')
                plot_marker(self.ecc_out, self.FF_out, 
                            title = self.generator.SXS.SXSnum,
                            xlabel = 'eccentricity',
                            ylabel = 'FF',
                            fname = filename, **kwargs)
            
    def save_waveform_fit(self, fname, **kwargs):
        if self.h22_fit is not None:
            if self._verbose:
                sys.stderr.write(f'{LOG}:Saving fitting waveform to {fname}.\n')
            self._h22_fit.saveh22(fname, **kwargs)
    
    def save_results(self, fname, **kwargs):
        # if not isinstance(fname, str):
        #     filename = 'scan_results.txt'
        # else:
        #     filename = fname
        filename = fname
        if self._verbose:
            sys.stderr.write(f'{LOG}:Saving sa results to {filename}\n')
        data = np.stack([self.tc_out.astype(np.str), 
                         self.phic_out.astype(np.str),
                         self.ecc_out.astype(np.str),
                         self.FF_out.astype(np.str),
                         self.CEV_STATE_out.name], axis = 1)
        np.savetxt(filename, data, fmt = '%s', **kwargs)
    
    
    def save_fit(self, fname, add_tag = False, **kwargs):
        # if not isinstance(fname, str):
        #     filename = 'results.csv'
        # else:
        #     filename = fname
        
        filename = fname
        if self._verbose:
            sys.stderr.write(f'{LOG}:Saving SXS fitting results to {filename}\n')
        
        if add_tag:
            fsplit = fname.split('.')
            suffix = fsplit[-1]
            filename = '.'.join(fsplit[:-1])
            filename += f'_{self._jobtag}.{suffix}'
        
        data = [[self.generator.SXS.SXSnum,
                self.generator.approx,
                self.generator.SXS.q,
                self.generator.SXS.s1x,
                self.generator.SXS.s1y,
                self.generator.SXS.s1z,
                self.generator.SXS.s2x,
                self.generator.SXS.s2y,
                self.generator.SXS.s2z,
                self.generator.SXS.ecc,
                self.generator.SXS.f_ini_dimless,
                self.generator.SXS.duration_dimM,
                self.generator.SXS.tpeak * self.generator.SXS.dim_t,
                self.tc_fit * self.generator.SXS.dim_t,
                self.phic_fit,
                self.max_FF,
                self.ecc_fit,
                self.tmove_fit * self.generator.SXS.dim_t,
                self.CEV_STATE_fit.name]]
        file = codecs.open(filename, 'a+', "gbk")
        writer = csv.writer(file)

        writer.writerows(data)
        file.close()
    
    def _get_fit_waveform(self):
        if self.CEV_STATE_fit is not CEV.SUCCESS:
            msg = f'SXSnum: {self.generator.SXS.SXSnum}\napprox: {self.generator.approx}\nerrtype: {self.CEV_STATE_fit.name}\nCMD: {self.generator._fecc(self.ecc_fit)}'
            sys.stderr.write(f'{WARNING}:Failed Job: \n{MESSAGE}: \n{msg}\n')
            self._errmsg = msg
            self._h22_fit = None
        else:
            self._errmsg = f'{self.generator.SXS.SXSnum}:{self.generator.approx}:Success'
            fit = self.generator.get_waveform(ecc = self.ecc_fit,
                                              jobtag = self._jobtag,
                                              verbose = False,
                                              timeout = 86400)
            fit.apply(-self.tc_fit + self.tmove_fit, self.phic_fit)
            self._h22_fit = fit

    @property
    def h22_fit(self):
        return self._h22_fit
            
    @property
    def generator(self):
        return self._core
    
    @property
    def ecc_out(self):
        return self._ecc
    
    @property
    def tc_out(self):
        return self._tc
    
    @property
    def phic_out(self):
        return self._phic
    
    @property
    def FF_out(self):
        return self._FF
    
    @property
    def CEV_STATE_out(self):
        return self._CEV_STATE
    
    @property
    def ecc_fit(self):
        return self._fit_ecc
    
    @property
    def tc_fit(self):
        return self._fit_tc

    @property
    def dephase_fit(self):
        return self._fit_tc * self.generator.SXS.dim_t
    
    @property
    def phic_fit(self):
        return self._fit_phic
    
    @property
    def tmove_fit(self):
        return self._fit_tmove
    
    @property
    def max_FF(self):
        return self._max_FF
    
    @property
    def CEV_STATE_fit(self):
        return self._fit_STATE
    
    @property
    def ErrorMsg(self):
        return self._errmsg
    
    
#--------Utils-------#
def get_wf_caller(approx, executable, SXSnum, verbose = False, **kwargs):
    core = SXSh22(SXSnum, **kwargs)
    ge = core.construct_generator(approx, executable)
    return ge

def calculate_overlap(wf_1, wf_2, psd = None, flow = 0, verbose = False, fullreturn = True):
    if verbose:
        sys.stderr.write(f'{LOG}:Checking input mode status...\n')
    if isinstance(wf_1, CEV) or isinstance(wf_2, CEV):
        if verbose:
            sys.stderr.write(f'{WARNING}:Abnormal mode...\n')
        return -1,0,0,wf_1.value, wf_2.value
    psdfunc = DetectorPSD(psd, flow)
    # Check sample rate
    if verbose:
        sys.stderr.write(f'{LOG}:Align data for comparison...\n')
    wf_1, wf_2, tmove = h22_alignment(wf_1, wf_2)
    if verbose:
        sys.stderr.write('Done\n')
        sys.stderr.write(f'{LOG}:Calculating overlap...')
    fs = wf_1.srate
    NFFT = len(wf_1)
    df = fs/NFFT
    freqs = np.abs(np.fft.fftfreq(NFFT, 1./fs))
    power_vec = psdfunc(freqs)
    htilde_1 = wf_1.h22f
    htilde_2 = wf_2.h22f
    O11 = np.sum(htilde_1 * htilde_1.conjugate() / power_vec).real * df
    O22 = np.sum(htilde_2 * htilde_2.conjugate() / power_vec).real * df
    Ox = htilde_1 * htilde_2.conjugate() / power_vec
    Oxt = np.fft.ifft(Ox) * fs / np.sqrt(O11 * O22)
    if verbose:
        sys.stderr.write('Done\n')
    if fullreturn:
        return Oxt, tmove, fs, CEV.SUCCESS.value, CEV.SUCCESS.value
    else:
        Oxt_abs = np.abs(Oxt)
        idx = np.where(Oxt_abs == max(Oxt_abs))[0][0]
        lth = len(Oxt_abs)
        if idx > lth / 2:
            tc = (idx - lth) / fs
        else:
            tc = idx / fs
        phic = np.angle(Oxt[idx])
        return max(Oxt_abs), tc, phic, tmove

def compare_with_SXS(SXSnum, hLM, **kwargs):
    NR = SXSh22(SXSnum)
    dtNR = np.gradient(NR.rawData.time)
    dth = np.gradient(hLM.time)
    min_dtNR = min(dtNR)
    max_dtNR = max(dtNR)
    min_dth = min(dth)
    max_dth = max(dth)
    fs = 1./ max(min_dtNR, max_dtNR, min_dth, max_dth)
    NRs = h22base(NR.rawData.time, NR.rawData.real, NR.rawData.imag, fs)
    hLMs = h22base(hLM.time, hLM.real, hLM.imag, fs)
    return calculate_overlap(NRs, hLMs, **kwargs)
    
def plot_fit(wf_1, wf_2, fname, name1 = 'name1', name2 = 'name2',
             FIT_linestyle = None, FIT_color = None, 
             FIT_alpha = None, FIT_linewidth = None,
             SXS_linestyle = None, SXS_color = None, 
             SXS_alpha = None, SXS_linewidth = None,
             **kwargs):
    
    Oxt, tmove, fs, _1, _2 = calculate_overlap(wf_1, wf_2)
    Oxt_abs = np.abs(Oxt)
    print(max(Oxt_abs))
    idx = np.where(Oxt_abs == max(Oxt_abs))[0][0]
    lth = len(Oxt_abs)
    if idx > lth / 2:
        tc = (idx - lth) / fs
    else:
        tc = idx / fs
    phic = np.angle(Oxt[idx])
    wf_2.apply(-tc + tmove, phic)
    if _1 != CEV.SUCCESS.value or _2 != CEV.SUCCESS.value:
        sys.stderr.write(f'{WARNING}:State is not success, skip waveform plotting.\n')
    else:
        filename = fname

        if FIT_linestyle is None:
            FIT_linestyle = DEFAULT_FIT_PLOT_LINESTYLE
        if FIT_color is None:
            FIT_color = DEFAULT_FIT_PLOT_COLOR
        if FIT_alpha is None:
            FIT_alpha = DEFAULT_FIT_ALPHA
        if FIT_linewidth is None:
            FIT_linewidth = DEFAULT_FIT_LINEWIDTH

        if SXS_linestyle is None:
            SXS_linestyle = DEFAULT_SXS_PLOT_LINESTYLE
        if SXS_color is None:
            SXS_color = DEFAULT_SXS_PLOT_COLOR
        if SXS_alpha is None:
            SXS_alpha = DEFAULT_SXS_ALPHA
        if SXS_linewidth is None:
            SXS_linewidth = DEFAULT_SXS_LINEWIDTH


        SXSplot = dict()
        FITplot = dict()
        SXSplot['x'] = wf_1.time
        SXSplot['y'] = wf_1.real
        SXSplot['name'] = name1
        SXSplot['linestyle'] = SXS_linestyle
        SXSplot['color'] = SXS_color
        SXSplot['alpha'] = SXS_alpha
        SXSplot['linewidth'] = SXS_linewidth
        
        FITplot['x'] = wf_2.time
        FITplot['y'] = wf_2.real
        FITplot['name'] = name2
        FITplot['linestyle'] = FIT_linestyle
        FITplot['color'] = FIT_color
        FITplot['alpha'] = FIT_alpha
        FITplot['linewidth'] = FIT_linewidth

        
        plot_compare_attach_any([SXSplot, FITplot], savefig = filename, tstart=0, **kwargs)
        return -tc + tmove, phic

def preset_ecc(SXSnum, retMid = False):
    for case in switch(SXSnum):
        if case('1355'):
            mid = 0.279776
            break
        if case('1356'):
            mid = 0.3420896
            break
        if case('1357'):
            mid = 0.4339328
            break
        if case('1358'):
            mid = 0.4600928
            break
        if case('1359'):
            mid = 0.43984
            break
        if case('1360'):
            mid = 0.526576
            break
        if case('1361'):
            mid = 0.5274464
            break
        if case('1362'):
            mid = 0.59
            break
        if case('1363'):
            mid = 0.581776
            break
        if case('1364'):
            mid = 0.2803712
            break
        if case('1365'):
            mid = 0.3123872
            break
        if case('1366'):
            mid = 0.44
            break
        if case('1367'):
            mid = 0.394416
            break
        if case('1368'):
            mid = 0.45016576
            break
        if case('1369'):
            mid = 0.5876
            break
        if case('1370'):
            mid = 0.583648
            break
        if case('1371'):
            mid = 0.2576224
            break
        if case('1372'):
            mid = 0.4562912
            break
        if case('1373'):
            mid = 0.4802464
            break
        if case('1374'):
            mid = 0.5848128
            break
        if case('0320'):
            mid = 0.103616
            break
        if case('0321'):
            mid = 0.2601568
            break
        if case('0322'):
            mid = 0.30392
            break
        if case('0323'):
            mid = 0.4219776
            break
        if case('0324'):
            mid = 0.5685408
            break
        mid = None
    if retMid:
        return mid
    if mid is not None:
        ret = [max(0, mid - 0.2), min(0.9, mid + 0.2)]
    else:
        ret = None
    return ret
        

def parse_ecc(ecc, minecc, maxecc, Preset = None):
    if Preset is not None:
        ecc_range = preset_ecc(Preset)
        if ecc_range is not None:
            return ecc_range
    if type(ecc) is str:
        try:
            xecc = float(ecc[1:])
        except:
            xecc = 0.5
        ecc = -1
        if xecc > 0.5:
            elip_min = 0
            elip_max = 0.6
        else:
            elip_max = min(0.6, xecc + 0.2)
            elip_min = max(0, xecc - 0.1)
    else:
        elip_max = min(0.6, ecc + 0.2)
        elip_min = max(0, ecc - 0.1)

    ecc_range = [elip_min, elip_max]
    if maxecc > 0 and elip_max < 0.15:
        ecc_range[1] = min(elip_max * 2, maxecc)
    elif maxecc > 0:
        ecc_range[1] = maxecc
        if minecc > 0 and minecc < maxecc:
            ecc_range[0] = minecc

    return ecc_range
    

DEFAULT_NAMECOL = ['#SXS id',
            '#approx',
            '#mass ratio',
            '#spin1z',
            '#spin1y',
            '#spin1z',
            '#spin2x',
            '#spin2y',
            '#spin2z',
            '#relaxed ecc',
            '#f_ini',
            '#duration',
            '#NR_time_peak',
            '#tc_fit',
            '#phic_fit',
            '#FF',
            '#ecc_fit',
            '#tpeak_move_fit',
            '#Status']
def save_namecol(filename, data = None):
    file = codecs.open(filename, 'wb', "gbk")
    writer = csv.writer(file)
    if data is None:
        data = [DEFAULT_NAMECOL]
    writer.writerows(data)
    file.close()
    
def load_csv(filename):
    file = codecs.open(filename, 'r', "gbk")
    reader = csv.reader(file,dialect='excel')
    data = []
    for i,line in enumerate(reader):
        if i>0:
            data.append(line)
    file.close()
    return data

def add_csv(filename, data):
    file = codecs.open(filename, 'a+', "gbk")
    writer = csv.writer(file)
    writer.writerows(data)
    file.close()

def resave_results(prefix, target):
    from pathlib import Path
    pre_list = prefix.split('/')
    prefix = pre_list.pop(-1)
    loc = Path('/'.join(pre_list))
    print(loc)
    lth = len(prefix)
    for fname in loc.iterdir():
        if fname.name[:lth] == prefix:
            break
    with open(fname, 'r') as f:
        firstline = f.readline()
    
    save_namecol(target, data = [firstline.split(',')])
    for fname in loc.iterdir():
        if fname.name[:lth] == prefix:
            sys.stderr.write(f'{LOG}:Saving {fname.name} to {target}...\n')
            data = load_csv(fname)
            add_csv(target, data)

def parse_SXSeccV1(SXSnum):
    for case in switch(SXSnum):
        if case('1355'):
            e0 = 0.280016
            break
        if case('1356'):
            e0 = 0.350032
            break
        if case('1357'):
            e0 = 0.4338912
            break
        if case('1358'):
            e0 = 0.4601088
            break
        if case('1359'):
            e0 = 0.4397376
            break
        if case('1360'):
            e0 = 0.5266624
            break
        if case('1361'):
            e0 = 0.53544
            break
        if case('1362'):
            e0 = 0.59
            break
        if case('1363'):
            e0 = 0.59
            break
        if case('1364'):
            e0 = 0.2642848
            break
        if case('1365'):
            e0 = 0.320086211
            break
        if case('1366'):
            e0 = 0.4401536
            break
        if case('1367'):
            e0 = 0.394346368
            break
        if case('1368'):
            e0 = 0.450117606
            break
        if case('1369'):
            e0 = 0.5876
            break
        if case('1370'):
            e0 = 0.5837184
            break
        if case('1371'):
            e0 = 0.280352
            break
        if case('1372'):
            e0 = 0.4563264
            break
        if case('1373'):
            e0 = 0.4802336
            break
        if case('1374'):
            e0 = 0.5848224
            break
        else:
            e0 = 0
            break
    return e0

def SEOBHyperCoefficients_v1(eta, a):
    c0 = 1.4467
    c1 = -1.7152360250654402
    c2 = -3.246255899738242
    KK = c0 + c1 * eta + c2 * eta * eta
    dSO = -69.5
    dSS = 2.75
    dtPeak = 0
    return KK, dSO, dSS, dtPeak

def SEOBHyperCoefficients_v2(eta, a):
    c20 = 1.712
    c21 = -1.803949138004582
    c22 = -39.77229225266885
    c23 = 103.16588921239249
    KK = c20 + c21 * eta + c22 * eta * eta + c23 * eta * eta * eta
    dSO = -74.71 - 156. * eta + 627.5 * eta * eta
    dSS = 8.127 - 154.2 * eta + 830.8 * eta * eta
    dtPeak = 0
    return KK, dSO, dSS, dtPeak

def SEOBHyperCoefficients_v4(eta, a):
    coeff00K = 1.7336
    coeff01K = -1.62045
    coeff02K = -1.38086
    coeff03K = 1.43659
    coeff10K = 10.2573
    coeff11K = 2.26831
    coeff12K = 0
    coeff13K = -0.426958
    coeff20K = -126.687
    coeff21K = 17.3736
    coeff22K = 6.16466
    coeff23K = 0
    coeff30K = 267.788
    coeff31K = -27.5201
    coeff32K = 31.1746
    coeff33K = -59.1658

    coeff00dSO = -44.5324
    coeff01dSO = 0
    coeff02dSO = 0
    coeff03dSO = 66.1987
    coeff10dSO = 0
    coeff11dSO = 0
    coeff12dSO = -343.313
    coeff13dSO = -568.651
    coeff20dSO = 0
    coeff21dSO = 2495.29
    coeff22dSO = 0
    coeff23dSO = 147.481
    coeff30dSO = 0
    coeff31dSO = 0
    coeff32dSO = 0
    coeff33dSO = 0

    coeff00dSS = 6.06807
    coeff01dSS = 0
    coeff02dSS = 0
    coeff03dSS = 0
    coeff10dSS = -36.0272
    coeff11dSS = 37.1964
    coeff12dSS = 0
    coeff13dSS = -41.0003
    coeff20dSS = 0
    coeff21dSS = 0
    coeff22dSS = -326.325
    coeff23dSS = 528.511
    coeff30dSS = 706.958
    coeff31dSS = 0
    coeff32dSS = 1161.78
    coeff33dSS = 0.

    coeff00DT = 2.50499
    coeff01DT = 13.0064
    coeff02DT = 11.5435
    coeff03DT = 0
    coeff10DT = 45.8838
    coeff11DT = -40.3183
    coeff12DT = 0
    coeff13DT = -19.0538
    coeff20DT = 13.0879
    coeff21DT = 0
    coeff22DT = 0
    coeff23DT = 0.192775
    coeff30DT = -716.044
    coeff31DT = 0
    coeff32DT = 0
    coeff33DT = 0


    chi = a / (1. - 2. * eta)
    eta2 = eta * eta
    eta3 = eta2 * eta
    chi2 = chi * chi
    chi3 = chi2 * chi
    KK = \
        coeff00K + coeff01K * chi + coeff02K * chi2 + coeff03K * chi3 + \
        coeff10K * eta + coeff11K * eta * chi + coeff12K * eta * chi2 + \
        coeff13K * eta * chi3 + coeff20K * eta2 + coeff21K * eta2 * chi + \
        coeff22K * eta2 * chi2 + coeff23K * eta2 * chi3 + coeff30K * eta3 + \
        coeff31K * eta3 * chi + coeff32K * eta3 * chi2 + coeff33K * eta3 * chi3
    
    dSO = \
            coeff00dSO + coeff01dSO * chi + coeff02dSO * chi2 + coeff03dSO * chi3 + \
            coeff10dSO * eta + coeff11dSO * eta * chi + coeff12dSO * eta * chi2 + \
            coeff13dSO * eta * chi3 + coeff20dSO * eta2 + coeff21dSO * eta2 * chi + \
            coeff22dSO * eta2 * chi2 + coeff23dSO * eta2 * chi3 + coeff30dSO * eta3 + \
            coeff31dSO * eta3 * chi + coeff32dSO * eta3 * chi2 + coeff33dSO * eta3 * chi3
    
    dSS = \
            coeff00dSS + coeff01dSS * chi + coeff02dSS * chi2 + coeff03dSS * chi3 + \
            coeff10dSS * eta + coeff11dSS * eta * chi + coeff12dSS * eta * chi2 + \
            coeff13dSS * eta * chi3 + coeff20dSS * eta2 + coeff21dSS * eta2 * chi + \
            coeff22dSS * eta2 * chi2 + coeff23dSS * eta2 * chi3 + coeff30dSS * eta3 + \
            coeff31dSS * eta3 * chi + coeff32dSS * eta3 * chi2 + coeff33dSS * eta3 * chi3
    
    dtPeak = \
        coeff00DT + coeff01DT * chi + coeff02DT * chi2 + coeff03DT * chi3 + \
        coeff10DT * eta + coeff11DT * eta * chi + coeff12DT * eta * chi2 + \
        coeff13DT * eta * chi3 + coeff20DT * eta2 + coeff21DT * eta2 * chi + \
        coeff22DT * eta2 * chi2 + coeff23DT * eta2 * chi3 + coeff30DT * eta3 + \
        coeff31DT * eta3 * chi + coeff32DT * eta3 * chi2 + coeff33DT * eta3 * chi3
    return KK, dSO, dSS, dtPeak


from .h22datatype import G_SI, c_SI, MRSUN_SI, MTSUN_SI, M_Sun_SI
def T2Timing_0PNCoeff(M, eta):
    return -5 /dim_t(M/M_Sun_SI)/ 256 / eta

def T2Timing_2PNCoeff(eta):
    return 7.43/2.52 + 11./3. * eta

def T2Timing_4PNCoeff(eta):
    return 30.58673/5.08032 + 54.29/5.04*eta + 61.7/7.2*eta*eta

def getChirpTimeBound(fmin, m1SI, m2SI, chi1, chi2):
    m1 = m1SI / M_Sun_SI
    m2 = m2SI / M_Sun_SI
    M = m1SI + m2SI
    mu = m1*m2/(m1+m2)
    eta = mu / (m1 + m2)
    chi = np.abs(chi1) if np.abs(chi1) > np.abs(chi2) else np.abs(chi2)
    c0 = np.abs(T2Timing_0PNCoeff(M, eta))
    c2 = T2Timing_2PNCoeff(eta)
    c3 = (226/15) * chi
    c4 = T2Timing_4PNCoeff(eta)
    v = np.cbrt(np.pi * G_SI * M * fmin) / c_SI
    return c0 * np.power(v,-8) * (1 + (c2 + (c3+c4*v)*v)*v*v)

def getFinalBlackHoleSpinBound(chi1, chi2):
    s = 0.686 + 0.15 * (chi1 + chi2)
    if s < abs(chi1):
        s = abs(chi1)
    if s < abs(chi2):
        s = abs(chi2)
    if (s > 0.998):
        s = 0.998
    return s

def getMergeTimeBound(m1SI, m2SI):
    norbits = 1
    M = m1SI + m2SI
    r = 9.0 * M * MRSUN_SI / M_Sun_SI
    v = c_SI / 3.0
    return norbits*(2.*np.pi*r/v)

def getRingdownTimeBound(M, s):
    nefolds = 11
    f1 = +1.5251
    f2 = -1.1568
    f3 = +0.1292
    q1 = +0.7000
    q2 = +1.4187
    q3 = -0.4990
    omega = (f1 + f2 * np.power(1.0 - s, f3)) / (M * MTSUN_SI / M_Sun_SI)
    Q = q1 + q2 * np.power(1.0 - s, q3)
    tau = 2.0 * Q/omega
    return nefolds * tau

def EstimateChirpLen(m1SI, m2SI, chi1, chi2, fmin, extra_cycles=3):
    tchirp = getChirpTimeBound(fmin, m1SI, m2SI, chi1, chi2)
    s = getFinalBlackHoleSpinBound(chi1, chi2)
    tmerge = getMergeTimeBound(m1SI, m2SI) + getRingdownTimeBound(m1SI + m2SI, s)
    return tchirp + tmerge
