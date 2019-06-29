#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:58:29 2019

@author: drizl
"""
import numpy as np
from .Utils import switch, cmd_stdout_cev, CEV, LOG, WARNING
from .h22datatype import h22base, dim_h, dim_t
import sys
from pathlib import Path

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
                        approx):
    CMD = f'{exe} --m1={m1} --m2={m2} \
            --spin1z={s1z} --spin2z={s2z} \
            --distance={D} --sample-rate={srate} \
            --f-min={f_ini} --approx={approx} --inclination=0'
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
                  f_ini):
    CMD = f'{exe} --m1 {m1} --m2 {m2} \
            --spin1z {s1z} --spin2z {s2z} \
            --distance {D} --e0 {ecc} \
            --f-min {f_ini} --sample-rate {srate} \
            --output --debug-e0 2'
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
                  f_ini):
    CMD = f'{exe} --m1 {m1} --m2 {m2} \
            --spin1z {s1z} --spin2z {s2z} \
            --distance {D} --e0 {ecc} \
            --f-min {f_ini} --sample-rate {srate} \
            --orbit-frequency'
            
    return CMD


# Classifier
class Generator(object):
    def __init__(self, approx, executable, verbose = False):
        self._approx = approx
        self._exe = executable
        # Get self._CMD i.e. waveform generator
        if verbose:
            sys.stderr.write(f'{LOG}:Parsing approx...')
        self._choose_CMD()
        if verbose:
            sys.stderr.write('Done\n')
        self._verbose = verbose
    
    def _choose_CMD(self):
        for case in switch(self._approx):
            if case('EOBNRv1') or \
                case('EOBNRv4') or \
                case('SEOBNRv1') or \
                case('SEOBNRv4'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini : \
                    CMD_lalsim_inspiral(exe = self._exe,
                                        m1 = m1,
                                        m2 = m2,
                                        s1z = s1z,
                                        s2z = s2z,
                                        D = D,
                                        srate = srate,
                                        f_ini = f_ini,
                                        approx = self._approx)
                def _pretreat(hr, hi, r, M):
                    hr *= np.sqrt(4 * np.pi / 5) * dim_h(r, M)
                    hi *= -np.sqrt(4 * np.pi / 5) * dim_h(r, M)
                    return hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = False
                break
            if case('SEOBNREv1'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini : \
                    CMD_SEOBNREv1(exe = self._exe, 
                                  m1 = m1, 
                                  m2 = m2,
                                  s1z = s1z,
                                  s2z = s2z,
                                  D = D,
                                  ecc = ecc,
                                  srate = srate,
                                  f_ini = f_ini)
                def _pretreat(hr, hi, r, M):
                    hr *= dim_h(r, M)
                    hi *= dim_h(r, M)
                    return hr, hi
                self._pretreat = _pretreat
                self._allow_ecc = True
                break
            if case('SEOBNREv4'):
                self._CMD = lambda m1, m2, s1z, s2z, D, ecc, srate, f_ini : \
                    CMD_SEOBNREv4(exe = self._exe,
                                  m1 = m1,
                                  m2 = m2,
                                  s1z = s1z,
                                  s2z = s2z,
                                  D = D,
                                  ecc = ecc,
                                  srate = srate,
                                  f_ini = f_ini)
                def _pretreat(hr, hi, r, M):
                    hr *= dim_h(r, M)
                    hi *= dim_h(r, M)
                    return hr, hi
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
        
    def __call__(self, 
                 m1, 
                 m2,
                 s1z, 
                 s2z,
                 D,
                 ecc,
                 srate,
                 f_ini,
                 timeout = 60,
                 jobtag = 'test'):
        cev, ret =  cmd_stdout_cev(self._CMD(m1 = m1,
                                    m2 = m2,
                                    s1z = s1z,
                                    s2z = s2z,
                                    D = D,
                                    ecc = ecc,
                                    srate = srate,
                                    f_ini = f_ini ), 
                            timeout = timeout,
                            name_out = jobtag)
        if cev is CEV.SUCCESS:
            if len(ret) == 0:
                ret = CEV.GEN_FAIL
            return ret
        else:
            return cev

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
        Num_x = len(xout)
        return np.asarray(xout), np.asarray(yout).reshape(-1, Num_x)

            
    def __check_adaptive(self,xlist, ylist, idx_ymax):
        ymax = ylist[idx_ymax]
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
        xmin = xlist[max(0,idx_ymax - 2)]
        xmax = xlist[min(lmax,idx_ymax + 2)]
        dx = dx / 5
        return xmin, xmax, dx, new_diff

    def __get_fx(self, xlist):
        ylist = []
        #Ntot = len(xlist)
        for (i,x) in enumerate(xlist):
            ylist.append(self.fun(x))
            #Process_v2(i+1, Ntot)
        return np.asarray(ylist)

