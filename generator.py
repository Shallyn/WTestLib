#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:58:29 2019

@author: drizl
"""
import numpy as np
from .Utils import switch, cmd_stdout_cev, CEV, LOG, WARNING
from .h22datatype import h22base, dim_h, dim_t, h22_alignment
import sys
from pathlib import Path
import csv,codecs,h5py

def add_csv(filename, data):
    file = codecs.open(filename, 'a+', "gbk")
    writer = csv.writer(file)
    writer.writerows(data)
    file.close()

DEFAULT_lalsim_inspiral = 'lalsim-inspiral'
DEFAULT_SEOBNREv1 = Path('/Users/drizl/Documents/2018/SEOBNRE/SEOBNRE_good/SEOBNRE')
DEFAULT_SEOBNREv4 = Path('/Users/drizl/Documents/2019/EOB_Test/Program/SEOBNRE/SEOBNRE')

#-------------Waveform Generator--------------#
# Type 01: For lalsim-inspiral
def CMD_lalsim_inspiral(exe,  
                        m1,
                        m2,
                        s1z,
                        s2z,
                        D,
                        srate,
                        f_ini,
                        approx,
                        s1x = 0,
                        s2x = 0,
                        s1y = 0,
                        s2y = 0,
                        **kwargs):
    CMD = f'{exe} --m1={m1} --m2={m2} \
            --spin1z={s1z} --spin2z={s2z} \
            --distance={D} --sample-rate={srate} \
            --f-min={f_ini} --approx={approx} --inclination=0 \
            --spin1x={s1x} --spin2x={s2x} \
            --spin1y={s1y} --spin2y={s2y}'
    return CMD

# Type 02: For SEOBNREv1:
def CMD_SEOBNREv1(exe,
                  m1,
                  m2,
                  s1z,
                  s2z,
                  D,
                  ecc,
                  srate,
                  f_ini,
                  **kwargs):
    CMD = f'{exe} --m1 {m1} --m2 {m2} \
            --spin1z {s1z} --spin2z {s2z} \
            --distance {D} --e0 {ecc} \
            --f-min {f_ini} --sample-rate {srate} \
            --output'
    return CMD

# Type 03: For SEOBNREv4:
def CMD_SEOBNREv4(exe,
                  m1,
                  m2,
                  s1z,
                  s2z,
                  D,
                  ecc,
                  srate,
                  f_ini,
                  **kwargs):
    CMD = f'{exe} --m1 {m1} --m2 {m2} \
            --spin1z {s1z} --spin2z {s2z} \
            --distance {D} --e0 {ecc} \
            --f-min {f_ini} --sample-rate {srate} \
            --orbit-frequency'
            
    return CMD

# Type 04: For new_SEOBNRE:
def CMD_new_SEOBNRE(exe,
                    m1,
                    m2,
                    s1z,
                    s2z,
                    D,
                    ecc,
                    srate,
                    f_ini,
                    approx,
                    **kwargs):
    CMD = f'{exe} --m1={m1} --m2={m2} \
            --spin1z={s1z} --spin2z={s2z} \
            --distance={D} --sample-rate={srate} \
            --f-min={f_ini} --eccentricity={ecc} \
            --approx={approx} --inclination=0'
    return CMD

def CMD_new_SEOBNREv4HM(exe,
                        m1,
                        m2,
                        s1z,
                        s2z,
                        D,
                        srate,
                        f_ini,
                        approx,
                        modeL,
                        modeM,
                        **kwargs):
    CMD = f'{exe} --m1={m1} --m2={m2} \
            --spin1z={s1z} --spin2z={s2z} \
            --distance={D} --sample-rate={srate} \
            --f-min={f_ini} --mode-only \
            --approx={approx} --model={modeL} \
            --modem={modeM} --inclination=0'
    return CMD

def CMD_pyEOBCal(exe,
                 m1,
                 m2,
                 s1z,
                 s2z,
                 ecc,
                 srate,
                 f_ini,
                 **kwargs):
    CMD = f'{exe} --m1={m1} --m2={m2} \
                --spin1z={s1z} --spin2z={s2z} \
                --sample-rate={srate} --f-min={f_ini} --eccentricity={ecc}'
    return CMD

def CMD_SEOBNRP(exe,  
                m1,
                m2,
                s1z,
                s2z,
                srate,
                f_ini,
                approx,
                s1x = 0,
                s2x = 0,
                s1y = 0,
                s2y = 0,
                version = 1,
                **kwargs):        
    CMD = f'{exe} --m1={m1} --m2={m2} \
            --spin1z={s1z} --spin2z={s2z} \
            --sample-rate={srate} \
            --f-min={f_ini} --inclination=0 \
            --spin1x={s1x} --spin2x={s2x} \
            --spin1y={s1y} --spin2y={s2y} --version={version}'
    if 'only22' in kwargs:
        CMD = CMD + ' --only22'
    return CMD

def CMD_SEOBNRPmode(exe,  
                    m1,
                    m2,
                    s1z,
                    s2z,
                    srate,
                    f_ini,
                    approx,
                    ecc = 0,
                    s1x = 0,
                    s2x = 0,
                    s1y = 0,
                    s2y = 0,
                    mode = -1,
                    version = 1,
                    **kwargs):   
    CMD = f'{exe} --m1={m1} --m2={m2} \
            --spin1z={s1z} --spin2z={s2z} \
            --sample-rate={srate} --eccentricity={ecc} \
            --f-min={f_ini} --inclination=0 \
            --spin1x={s1x} --spin2x={s2x} --version={version} \
            --spin1y={s1y} --spin2y={s2y} --mode={mode}'
    return CMD

def CMD_SEOBNREv5(exe,
                  q,
                  deltaT,
                  ecc,
                  s1z,
                  s2z,
                  f_ini,
                  version = 3,
                  KK = None,
                  dSO = None,
                  dSS = None,
                  dtPeak = None,
                  ret = 0,
                  dump = None,
                  s1x = 0,
                  s2x = 0,
                  s1y = 0,
                  s2y = 0,
                  **kwargs):
    adjpms = ''
    if KK is not None:
        adjpms += f' --KK={KK}'
    if dSO is not None:
        adjpms += f' --dso={dSO}'
    if dSS is not None:
        adjpms += f' --dss={dSS}'
    if dtPeak is not None:
        adjpms += f' --dtPeak={dtPeak}'
    if dump is not None:
        adjpms += f' --fileout={dump}'
    for kw in kwargs:
        op = kw.replace('_','-')
        adjpms += f' --{op}={kwargs[kw]}'
    CMD = f'{exe} --mass-ratio={q} --f-min={f_ini} \
        --delta-t={deltaT} --eccentricity={ecc} \
        --chi1={s1z} --chi2={s2z} --version={version} --return-mode={ret}' + adjpms
    return CMD


# Classifier
class Generator(object):
    def __init__(self, approx, executable, verbose = False, diy = None):
        self._approx = approx
        self._exe = executable
        # Get self._CMD i.e. waveform generator
        if verbose:
            sys.stderr.write(f'{LOG}:Parsing approx...')
        self._choose_CMD(diy)
        if verbose:
            sys.stderr.write('Done\n')
        self._verbose = verbose
    
    def _choose_CMD(self, diy = None):
        self._HM = False
        for case in switch(self._approx):
            if case('diy'):
                if diy is None:
                    raise Exception('no diy dict')
                self._CMD = diy['CMD']
                self._pretreat = diy['pretreat']
                self._allow_ecc = diy['allow_ecc']
                break
            if case('EOBNRv1') or \
                case('EOBNRv4') or \
                case('SEOBNRv1') or \
                case('SEOBNRv2') or \
                case('IMRPhenomD') or \
                case('SEOBNRv4P') or \
                case('SEOBNRv4PHM') or \
                case('SEOBNRv4'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs : \
                    CMD_lalsim_inspiral(exe = self._exe,
                                        m1 = m1,
                                        m2 = m2,
                                        s1z = s1z,
                                        s2z = s2z,
                                        D = D,
                                        srate = srate,
                                        f_ini = f_ini,
                                        approx = self._approx,
                                        **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    hr *= np.sqrt(4 * np.pi / 5) * dim_h(r, M)
                    hi *= -np.sqrt(4 * np.pi / 5) * dim_h(r, M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = False
                break
            if case('SEOBNRP'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs : \
                    CMD_SEOBNRP(exe = self._exe,
                                m1 = m1,
                                m2 = m2,
                                s1z = s1z,
                                s2z = s2z,
                                srate = srate,
                                f_ini = f_ini,
                                approx = self._approx,
                                only22 = True,
                                **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    hr *= np.sqrt(4 * np.pi / 5) * dim_h(r, M)
                    hi *= -np.sqrt(4 * np.pi / 5) * dim_h(r, M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = False
                break
            if case('SEOBNRPHM'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs : \
                    CMD_SEOBNRP(exe = self._exe,
                                m1 = m1,
                                m2 = m2,
                                s1z = s1z,
                                s2z = s2z,
                                srate = srate,
                                f_ini = f_ini,
                                approx = self._approx,
                                **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    hr *= np.sqrt(4 * np.pi / 5) * dim_h(r, M)
                    hi *= -np.sqrt(4 * np.pi / 5) * dim_h(r, M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = False
                break
            if case('SEOBNRPHMOld'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs : \
                    CMD_SEOBNRP(exe = self._exe,
                                m1 = m1,
                                m2 = m2,
                                s1z = s1z,
                                s2z = s2z,
                                srate = srate,
                                f_ini = f_ini,
                                approx = self._approx,
                                version = 0,
                                **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    hr *= np.sqrt(4 * np.pi / 5) * dim_h(r, M)
                    hi *= -np.sqrt(4 * np.pi / 5) * dim_h(r, M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = False
                break
            if case('SEOBNRPmode'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs : \
                    CMD_SEOBNRPmode(exe = self._exe,
                                    m1 = m1,
                                    m2 = m2,
                                    s1z = s1z,
                                    s2z = s2z,
                                    ecc = ecc,
                                    srate = srate,
                                    f_ini = f_ini,
                                    approx = self._approx,
                                    **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    t = t / dim_t(M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = True
                break
            if case('SEOBNRPmodeOld'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs : \
                    CMD_SEOBNRPmode(exe = self._exe,
                                    m1 = m1,
                                    m2 = m2,
                                    s1z = s1z,
                                    s2z = s2z,
                                    ecc = ecc,
                                    srate = srate,
                                    f_ini = f_ini,
                                    approx = self._approx,
                                    version = 0,
                                    **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    t = t / dim_t(M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = True
                break
            if case('SEOBNRPmodeV3'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs : \
                    CMD_SEOBNRPmode(exe = self._exe,
                                    m1 = m1,
                                    m2 = m2,
                                    s1z = s1z,
                                    s2z = s2z,
                                    ecc = ecc,
                                    srate = srate,
                                    f_ini = f_ini,
                                    approx = self._approx,
                                    version = 3,
                                    **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    t = t / dim_t(M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = True
                break
            if case('SEOBNRPE'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs : \
                    CMD_SEOBNRPmode(exe = self._exe,
                                    m1 = m1,
                                    m2 = m2,
                                    s1z = s1z,
                                    s2z = s2z,
                                    ecc = ecc,
                                    srate = srate,
                                    f_ini = f_ini,
                                    approx = self._approx,
                                    version = 2,
                                    **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    t = t / dim_t(M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = True
                break
            if case('SEOBNREv1'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs: \
                    CMD_SEOBNREv1(exe = self._exe, 
                                  m1 = m1, 
                                  m2 = m2,
                                  s1z = s1z,
                                  s2z = s2z,
                                  D = D,
                                  ecc = ecc,
                                  srate = srate,
                                  f_ini = f_ini,
                                  **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    hr *= dim_h(r, M)
                    hi *= dim_h(r, M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = True
                break
            if case('SEOBNREv4'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs: \
                    CMD_SEOBNREv4(exe = self._exe,
                                  m1 = m1,
                                  m2 = m2,
                                  s1z = s1z,
                                  s2z = s2z,
                                  D = D,
                                  ecc = ecc,
                                  srate = srate,
                                  f_ini = f_ini,
                                  **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    hr *= dim_h(r, M)
                    hi *= dim_h(r, M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = True
                break
            if case('new_SEOBNREv1') or \
                case('new_SEOBNREv2') or \
                case('new_SEOBNREv4'):
                self._approx = self._approx.split('_')[-1]
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs: \
                    CMD_new_SEOBNRE(exe = self._exe,
                                    m1 = m1,
                                    m2 = m2,
                                    s1z = s1z,
                                    s2z = s2z,
                                    D = D,
                                    ecc = ecc,
                                    srate = srate,
                                    f_ini = f_ini,
                                    approx = self._approx,
                                    **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    hr *= np.sqrt(4 * np.pi / 5) * dim_h(r, M)
                    hi *= -np.sqrt(4 * np.pi / 5) * dim_h(r, M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = True
                break
            
            if case('SEOBNREv4HM'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs: \
                    CMD_new_SEOBNREv4HM(exe = self._exe,
                                        m1 = m1,
                                        m2 = m2,
                                        s1z = s1z,
                                        s2z = s2z,
                                        D = D,
                                        srate = srate,
                                        f_ini = f_ini,
                                        approx = self._approx,
                                        modeL = L,
                                        modeM = M,
                                        **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    hr *= dim_h(r, M)
                    hi *= dim_h(r, M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = False
                self._HM = True
                break

            if case('pyEOBCal'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs:\
                    CMD_pyEOBCal(exe = self._exe,
                                 m1 = m1,
                                 m2 = m2,
                                 s1z = s1z,
                                 s2z = s2z,
                                 ecc = ecc,
                                 srate = srate,
                                 f_ini = f_ini,
                                 **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    #t = t / dim_t(M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = True
                break

            if case('SEOBNREv5'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs:\
                    CMD_SEOBNREv5(exe = self._exe,
                                  q = m1 / m2,
                                  deltaT = dim_t(m1+m2)/srate,
                                  ecc = ecc,
                                  s1z = s1z,
                                  s2z = s2z,
                                  f_ini = f_ini / dim_t(m1 + m2),
                                  **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    t = t / dim_t(M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = True
                break

            if case('SEOBNREv6'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini, L, M, **kwargs:\
                    CMD_SEOBNREv5(exe = self._exe,
                                  q = m1 / m2,
                                  deltaT = dim_t(m1+m2)/srate,
                                  ecc = ecc,
                                  s1z = s1z,
                                  s2z = s2z,
                                  f_ini = f_ini / dim_t(m1 + m2),
                                  version = -2,
                                  **kwargs)
                def _pretreat(t, hr, hi, r, M, **kwargs):
                    t = t / dim_t(M)
                    return t, hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = True
                break

            raise ValueError('Unsupported approx: {}'.format(self._approx))
    
    @property
    def allow_ecc(self):
        return self._allow_ecc 
    
    @property
    def approx(self):
        return self._approx

    @property
    def pretreat(self):
        return self._pretreat
    
    def call(self, 
                 m1, 
                 m2,
                 s1z, 
                 s2z,
                 D,
                 ecc,
                 srate,
                 f_ini,
                 L,
                 M,
                 **kwargs):
        EXE = self._CMD(m1 = m1,
                        m2 = m2,
                        s1z = s1z,
                        s2z = s2z,
                        D = D,
                        ecc = ecc,
                        srate = srate,
                        f_ini = f_ini,
                        L = L,
                        M = M,
                        **kwargs)
        return EXE
    
    @property
    def HM(self):
        return self._HM

    def get_CMD(self, 
                 m1, 
                 m2,
                 s1z, 
                 s2z,
                 D,
                 ecc,
                 srate,
                 f_ini,
                 L,
                 M,
                 timeout = 60,
                 jobtag = '_test',
                 **kwargs):
        EXE = self._CMD(m1 = m1,
                        m2 = m2,
                        s1z = s1z,
                        s2z = s2z,
                        D = D,
                        ecc = ecc,
                        srate = srate,
                        f_ini = f_ini,
                        L = L,
                        M = M,
                        **kwargs)
        return EXE
        
    def __call__(self, 
                 m1, 
                 m2,
                 s1z, 
                 s2z,
                 D,
                 ecc,
                 srate,
                 f_ini,
                 L,
                 M,
                 timeout = 60,
                 jobtag = '_test',
                 **kwargs):
        EXE = self._CMD(m1 = m1,
                        m2 = m2,
                        s1z = s1z,
                        s2z = s2z,
                        D = D,
                        ecc = ecc,
                        srate = srate,
                        f_ini = f_ini,
                        L = L,
                        M = M,
                        **kwargs)
        if self._verbose:
            sys.stderr.write(f'{LOG}:{jobtag}-> \n{EXE}\n')
        cev, ret =  cmd_stdout_cev(EXE, 
                            timeout = timeout,
                            name_out = jobtag)
        if cev is CEV.SUCCESS:
            if len(ret) == 0:
                ret = CEV.GEN_FAIL
            return ret
        else:
            return cev

from .h22datatype import get_fmin
class CompGenerator(object):
    def __init__(self, approx1, exe1, approx2, exe2, psd = None, verbose = False):
        self._verbose = verbose
        if verbose:
            sys.stderr.write(f'{LOG}:Construct CompGenerator...\n')
        gen1 = Generator(approx1, exe1, verbose)
        gen2 = Generator(approx2, exe2, verbose)
        self._psd = psd
        self._get_wf1 = gen1.__call__
        self._pretreat1 = gen1._pretreat
        self._get_wf2 = gen2.__call__
        self._pretreat2 = gen2._pretreat
        
    def compare(self,
                q,
                s1z,
                s2z,
                ecc,
                Mtotal = 16,
                D = 100,
                f_ini = 40,
                srate = 16384,
                timeout = 60,
                jobtag = '_test'):
        if self._verbose:
            sys.stderr.write(f'{LOG}:Initialize parameter...\n')
        if hasattr(q, '__len__'):
            nq = len(q)
        else:
            nq = 1
            q = [q]
        
        if hasattr(s1z, '__len__'):
            ns1z = len(s1z)
        else:
            ns1z = 1
            s1z = [s1z]
            
        if hasattr(s2z, '__len__'):
            ns2z = len(s2z)
        else:
            ns2z = 1
            s2z = [s2z]
            
        if hasattr(ecc, '__len__'):
            necc = len(ecc)
        else:
            necc = 1
            ecc = [ecc]
        if self._verbose:
            sys.stderr.write(f'{LOG}:Now calling...\n')
        ret = np.zeros([nq, ns1z, ns2z, necc])
        for i, qi in enumerate(q):
            for j, s1zj in enumerate(s1z):
                for k, s2zk in enumerate(s2z):
                    for l, eccl in enumerate(ecc):
                        m1 = Mtotal * qi / (1 + qi)
                        m2 = Mtotal / (1 + qi) 
                        ans = self._core_calcFF(m1, m2, 
                                               s1zj, s2zk, eccl,
                                               D, f_ini, 
                                               srate, timeout, jobtag)
                        ret[i,j,k,l] = ans
                        sys.stderr.write(f'PMS: m1 = {m1}, m2 = {m2}, s1z = {s1zj}, s2z = {s2zk} ecc = {eccl}\n\t FF = {ans}\n\n')
        
        return ret

    def comp_random_ecc(self,
                       min_q, 
                       max_q, 
                       min_s1z, 
                       max_s1z, 
                       min_s2z, 
                       max_s2z, 
                       fsave,
                       min_ecc = 0,
                       max_ecc = 0.6,
                       Num = 10,
                       NumEcc = 60,
                       min_Mtotal = None,
                       max_Mtotal = None,
                       D = 100,
                       f_ini = 0.002,
                       srate = 16384,
                       timeout = 60,
                       jobtag = '_CompareRandom',
                       mode = 22):
        if self._verbose:
            sys.stderr.write(f'{LOG}:Initialize parameter...\n')
        Num = int(Num)
        # q = np.random.uniform(min_q, max_q, Num)
        max_eta = min_q / (1+min_q) / (1+min_q)
        min_eta = max_q / (1+max_q) / (1+max_q)
        eta = np.random.uniform(min_eta, max_eta, Num)
        q = (1+np.sqrt(1-4*eta))/(2*eta) - 1
        if min_s1z is None or max_s1z is None:
            s1z = np.zeros(Num)
        else:
            s1z = np.random.uniform(min_s1z, max_s1z, Num)

        if min_s2z is None or max_s2z is None:
            s2z = np.zeros(Num)
        else:
            s2z = np.random.uniform(min_s2z, max_s2z, Num)

        ecc = np.linspace(min_ecc, max_ecc, NumEcc)
        
        if min_Mtotal is None or max_Mtotal is None:
            mtotal_list = np.linspace(10, 200, 30)
        else:
            mtotal_list = np.linspace(min_Mtotal, max_Mtotal, 30)

        # spin normalization
        for i in range(Num):
            s1Vec = np.array([0,0,s1z[i]])
            s2Vec = np.array([0,0,s2z[i]])
            ans = np.zeros(len(ecc))
            if (mode % 10) % 2 and q[i] == 1 and s1Vec[0] ==  s2Vec[0] and s1Vec[1] ==  s2Vec[1] and s1Vec[2] ==  s2Vec[2]:
                ans = np.ones(len(ecc))
            else:
                for j, e0 in enumerate(ecc):
                    ans[j] = self.core_calcFF(q[i], mtotal_list, 
                                            s1Vec[2], s2Vec[2], e0,
                                            D, f_ini, 
                                            srate, timeout, jobtag, mode = mode, 
                                            s1x = s1Vec[0], s2x = s2Vec[0],
                                            s1y = s1Vec[1], s2y = s2Vec[1])
            sys.stderr.write(f'PMS: q = {q[i]}, s1z = {s1z[i]}, s2z = {s2z[i]}\n\t FF = {np.min(ans)}\n\n')
            data = [[q[i], s1Vec[0], s1Vec[1], s1Vec[2], s2Vec[0], s2Vec[1], s2Vec[2], ecc[i]] + ans.tolist()]
            add_csv(fsave, data)
        return


    def compare_random(self,
                       min_q, 
                       max_q, 
                       min_s1z, 
                       max_s1z, 
                       min_s2z, 
                       max_s2z, 
                       min_ecc,
                       max_ecc, 
                       fsave,
                       Num = 10,
                       Mtotal = 16,
                       min_Mtotal = None,
                       max_Mtotal = None,
                       D = 100,
                       f_ini = 40,
                       srate = 16384,
                       timeout = 60,
                       jobtag = '_CompareRandom',
                       mode = 22,
                       use_prec = False,
                       use_fcut = None):
        if self._verbose:
            sys.stderr.write(f'{LOG}:Initialize parameter...\n')
        Num = int(Num)

        # q = np.random.uniform(min_q, max_q, Num)
        max_eta = min_q / (1+min_q) / (1+min_q)
        min_eta = max_q / (1+max_q) / (1+max_q)
        eta = np.random.uniform(min_eta, max_eta, Num)
        q = (1+np.sqrt(1-4*eta))/(2*eta) - 1
        if min_s1z is None or max_s1z is None:
            s1z = np.zeros(Num)
        else:
            s1z = np.random.uniform(min_s1z, max_s1z, Num)

        if min_s2z is None or max_s2z is None:
            s2z = np.zeros(Num)
        else:
            s2z = np.random.uniform(min_s2z, max_s2z, Num)

        if min_ecc is None or max_ecc is None:
            ecc = np.zeros(Num)
        else:
            ecc = np.random.uniform(min_ecc, max_ecc, Num)
        
        if min_Mtotal is None or max_Mtotal is None:
            mtotal_list = np.linspace(10, 200, 30)
        else:
            mtotal_list = np.linspace(min_Mtotal, max_Mtotal, 30)
        
        # use_prec

        # spin normalization
        for i in range(Num):
            if use_prec:
                s1Vec = np.random.randn(3)
                s1 = np.linalg.norm(s1Vec)
                s1Vec = s1Vec * s1z[i] / s1

                s2Vec = np.random.randn(3)
                s2 = np.linalg.norm(s2Vec)
                s2Vec = s2Vec * s2z[i] / s2
            else:
                s1Vec = np.array([0,0,s1z[i]])
                s2Vec = np.array([0,0,s2z[i]])
            if (mode % 10) % 2 and q[i] == 1 and s1Vec[0] ==  s2Vec[0] and s1Vec[1] ==  s2Vec[1] and s1Vec[2] ==  s2Vec[2]:
                ans = 1.
            else:
                ans = self.core_calcFF(q[i], mtotal_list, 
                                        s1Vec[2], s2Vec[2], ecc[i],
                                        D, f_ini, 
                                        srate, timeout, jobtag, mode = mode, 
                                        s1x = s1Vec[0], s2x = s2Vec[0],
                                        s1y = s1Vec[1], s2y = s2Vec[1],
                                        use_fcut = use_fcut)
            sys.stderr.write(f'PMS: q = {q[i]}, s1z = {s1z[i]}, s2z = {s2z[i]} ecc = {ecc[i]}\n\t FF = {ans}\n\n')
            if ans < 0:
                continue
            data = [[q[i], s1Vec[0], s1Vec[1], s1Vec[2], s2Vec[0], s2Vec[1], s2Vec[2], ecc[i], ans]]
            add_csv(fsave, data)   
        return
        
    
    def core_calcFF(self, q, mtotal_list, s1z, s2z, ecc,
                     D, f_ini, srate, timeout, jobtag, use_fcut = None, 
                     **kwargs):
        mtotal_base = 40
        mtotal_max = np.max(mtotal_list)
        m1 = mtotal_base * q / (1 + q)
        m2 = mtotal_base / (1 + q)
        f_ini_dim = f_ini * dim_t(mtotal_base)
        if use_fcut is not None:
            f_ini_dim = use_fcut * dim_t(mtotal_base)
        # f_ini_dim = 25
        data = self._get_wf1(m1 = m1, m2 = m2, s1z = s1z, s2z = s2z, 
                             D = D, ecc = ecc, srate = srate, f_ini = f_ini_dim, 
                             L = 2, M = 2,
                             timeout = timeout, jobtag = jobtag, **kwargs)
        if isinstance(data, CEV):
            return -1
        t1, hr1, hi1 = data[:,0], data[:,1], data[:,2]
        t1, hr1, hi1 = self._pretreat1(t1, hr1, hi1, D, mtotal_base)

        wf1 = h22base(t1, hr1, hi1, srate)

        data = self._get_wf2(m1 = m1, m2 = m2, s1z = s1z, s2z = s2z, 
                             D = D, ecc = ecc, srate = srate, f_ini = f_ini_dim, 
                             L = 2, M = 2,
                             timeout = timeout, jobtag = jobtag, **kwargs)
        if isinstance(data, CEV):
            return 0
        t2, hr2, hi2 = data[:,0], data[:,1], data[:,2]
        t2, hr2, hi2 = self._pretreat2(t2, hr2, hi2, D, mtotal_base)
        wf2 = h22base(t2, hr2, hi2, srate)
        wf1, wf2, tmove = h22_alignment(wf1, wf2)
        fs_old = wf1.srate
        NFFT = len(wf1)
        df_old = fs_old/NFFT
        FF_list = []
        for Mtotal in mtotal_list:
            df = df_old *  mtotal_base / Mtotal
            fs = df * NFFT
            freqs = np.abs(np.fft.fftfreq(NFFT, 1./fs))
            power_vec = self._psd(freqs)
            Stilde = wf1.h22f
            htilde = wf2.h22f
            O11 = np.sum(Stilde * Stilde.conjugate() / power_vec).real * df
            O22 = np.sum(htilde * htilde.conjugate() / power_vec).real * df
            Ox = Stilde * htilde.conjugate() / power_vec
            Oxt = np.fft.ifft(Ox) * fs
            Oxt_abs = np.abs(Oxt) / np.sqrt(O11 * O22)
            idx = np.where(Oxt_abs == max(Oxt_abs))[0][0]
            FF = Oxt_abs[idx]
            FF_list.append(FF)
        return np.min(FF_list)


        





#------------Utils-----------#

#-----------self adaptive-----------#
class self_adaptivor(object):
    def __init__(self, fun, xrg, xstep, outindex = 0, verbose = False):
        # The Output Of fun must be list
        self.fun = fun
        self.xrg = xrg
        self.outindex = outindex
        self.step = xstep
        self._verbose = verbose
        
        
    def run(self, maxitr = None, verbose = False, prec_x = 1e-6, prec_y = 1e-6):
        xmin = self.xrg[0]
        xmax = self.xrg[1]
        dx = self.step
        xout = []
        yout = []
        diff = np.inf
        ymax = -np.inf
        idx = self.outindex
        if maxitr is None:
            maxitr = 10
        icount = 0
        if verbose:
            sys.stderr.write('xrange: '+str(xmin)+'-'+str(xmax)+'\n')
            sys.stderr.write('YMAX         DIFF         X         DX\n')
        while(icount < maxitr):
            if abs(xmax - xmin) <= dx * 0.9:
                break
            sys.stderr.write('Xrange:[{xmin}, {xmax}]')
            xlist = np.arange(xmin, xmax, dx)
            ylist = self.__get_fx(xlist)
            xout += xlist.tolist()
            yout += ylist.tolist()
            new_ymax = np.max(ylist[:, idx])
            new_ymin = np.min(ylist[:, idx])
            if new_ymax == new_ymin:
                sys.stderr.write(f'{WARNING}: Cannot find maximum y\n')
                break
            idx_ymax = np.where(ylist[:, idx] == new_ymax)[0][0]
            xmin, xmax, dx, new_diff = self.__check_adaptive(xlist, ylist[:, idx], idx_ymax)
            if verbose:
                sys.stderr.write('%.6f'%(new_ymax) + \
                                 '    %.6f'%(new_diff) + \
                                 '    %.6f'%(xlist[idx_ymax]) + \
                                 '    %.6f'%(dx) + '\n')
            if new_ymax > ymax and abs(new_ymax - ymax) > prec_y and dx > prec_x:
                icount = 0
                diff = new_diff
                continue
            else:
                if new_diff < diff:
                    diff = new_diff
                    icount += 1
                    continue
                else:
                    break
        xout = np.asarray(xout)
        yout = np.asarray(yout)
        if len(yout.shape) is 1:
            yout = yout.reshape(1, xout.size)
        return xout, yout

            
    def __check_adaptive(self,xlist, ylist, idx_ymax):
        ymax = ylist[idx_ymax]
        idx_ymax2 = np.argsort(ylist)[-2]
        ymax2 = ylist[idx_ymax2]
        dx = xlist[1] - xlist[0]
        lmax = len(ylist) - 1
        diff_l = abs(ymax - ylist[max(0, idx_ymax - 1)])
        diff_r = abs(ymax - ylist[min(lmax,idx_ymax + 1)])
        if diff_l == 0:
            new_diff = diff_r
        else:
            if diff_r == 0:
                new_diff = diff_l
            else:
                new_diff = (diff_l + diff_r) / 2
        deltaIdx = max(3, 1 + abs(idx_ymax - idx_ymax2))
        xmin = max(0, xlist[max(0,idx_ymax - deltaIdx)] - dx / 6)
        xmax = min(1.0, xlist[min(lmax,idx_ymax + deltaIdx)] + dx/6)
        dx = dx / 5
        return xmin, xmax, dx, new_diff

    def __get_fx(self, xlist):
        ylist = []
        #Ntot = len(xlist)
        for (i,x) in enumerate(xlist):
            fret = self.fun(x)
            if self._verbose:
                sys.stderr.write(f'{LOG}: {fret}\n')
            ylist.append(fret)
            #Process_v2(i+1, Ntot)
        return np.asarray(ylist)

