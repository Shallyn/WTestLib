#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 10:52:52 2019

@author: drizl
"""


import numpy as np
from optparse import OptionParser
from .h22datatype import h22base, get_Mtotal, h22_alignment, get_fini_dimless
from .generator import Generator
from .Utils import CEV, cmd_stdout_cev
import sys
#-----Parse args-----#
def parseargs(argv):
    # Input Parameters:
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .generator import DEFAULT_SEOBNREv1
    
    parser = OptionParser(description='Waveform Comparation With SXS')
    parser.add_option('--exe', type = 'str', default = DEFAULT_SEOBNREv1.name, help = 'Exe command')
    parser.add_option('--approx', type = 'str', default = 'SEOBNREv1', help = 'approx')
    parser.add_option('--jobtag', type = 'str', default = 'job', help = 'Jobtag for the code run')
    parser.add_option('--prefix', type = 'str', default = '.', help = 'dir for results saving.')
    parser.add_option('--verbose', action = 'store_true', help = 'If added, will print verbose message.')
    parser.add_option('--psd', type = 'str', help = 'Detector psd.')
    parser.add_option('--srate', type = 'float', default = 16384, help = 'Sample rate')

    parser.add_option('--max-q', type = 'float', default = 5, help = 'Mass ratio')
    parser.add_option('--max-s1z', type = 'float', default = 0.5, help = 'Spin1z')
    parser.add_option('--max-s2z', type = 'float', default = 0.5, help = 'Spin2z')
    parser.add_option('--randomecc', action = 'store_true')
    parser.add_option('--f-ini', type = 'float', default = 0.002, help = 'Initial orbital frequency[M]')
    parser.add_option('--f-min', type = 'float', default = 10, help = 'Initial orbital frequency[Hz]')
    parser.add_option('--Mtotal', type = 'float', default = 20, help = 'Total system mass[M_Sun]')
    parser.add_option('--nsample', type = 'int', default = 10000, help = 'Number for sample')
    parser.add_option('--seed', type = 'int', help = 'Seed for random generator')

    parser.add_option('--timeout', type = 'int', default = 60, help = 'Used for waveform generation')
    args, _ = parser.parse_args(argv)
    return args

#-----bank-----#
class WfGenerator(Generator):
    def __init__(self, approx, executable):
        super(WfGenerator, self).__init__(approx, executable)
    
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
                 jobtag = '_test'):
        EXE = self._CMD(m1 = m1,
                        m2 = m2,
                        s1z = s1z,
                        s2z = s2z,
                        D = D,
                        ecc = ecc,
                        srate = srate,
                        f_ini = f_ini,
                        L = L,
                        M = M)
        cev, ret =  cmd_stdout_cev(EXE, 
                            timeout = timeout,
                            name_out = jobtag)
        if cev is CEV.SUCCESS:
            if len(ret) == 0:
                return CEV.GEN_FAIL
            t, hi, hr = ret[:,0], ret[:,1], ret[:,2]
            hr, hi = self._pretreat(hr, hi, D, m1 + m2)
            return (t, hr, hi)
        else:
            return cev


#----------------------------------#
#               Main               #
#----------------------------------#

from itertools import product
from .SXS import save_namecol, add_csv, calculate_overlap
import time as pytime
from pathlib import Path
def main(argv = None):
    args = parseargs(argv)
    srate = args.srate
    approx = args.approx
    exe = args.exe
    Gfunc = WfGenerator(approx, exe)
    
    flcut = args.f_min
    fini = args.f_ini
    Mtotal = args.Mtotal
    timeout = args.timeout
    D = 100
    if Mtotal < 0:
        Mtotal = get_Mtotal(fini, flcut)
    else:
        fini = get_fini_dimless(flcut, Mtotal)
    
    N = args.nsample

    q_max = args.max_q
    s1z_max = args.max_s1z
    s2z_max = args.max_s2z

    randomecc = args.randomecc

    psd = args.psd
    seed = args.seed
    jobtag = args.jobtag
    ttag = int(pytime.time() % 10000)
    rdtag =int(np.random.uniform(0,1)*pytime.time())
    if seed:
        seedrd = int(pytime.time() % 100)
        np.random.seed(seed + seedrd)
    jobtag = f'_{jobtag}_{ttag}_{rdtag}.job'
    prefix = Path(args.prefix)
    if not prefix.exists():
        prefix.mkdir(parents=True)
        
    fout = prefix / f'results_{psd}_{jobtag}.csv'
    if not fout.exists():
        save_namecol(fout, data = [['#q', '#s1z','#s2z','#ecc', '#1-FF']])
    
    for i in range(N):
        q = np.random.uniform(1, q_max)
        s1z = np.random.uniform(-s1z_max, s1z_max)
        s2z = np.random.uniform(-s2z_max, s2z_max)
        m2 = Mtotal / (q+1)
        m1 = Mtotal - m2
        wfC = Gfunc(m1, m2, s1z, s2z, D, 0, srate, flcut, 2, 2, jobtag = jobtag, timeout = timeout)
        if isinstance(wfC, CEV):
            sys.stderr.write('Error: m1, m2, s1z, s2z = %.3f, %.3f, %.3f, %.3f\n'%(m1, m2, s1z, s2z))
            continue
        wfC = h22base(wfC[0], wfC[1], wfC[2], srate)
        if not randomecc:
            ecc_ls = [0.01*i for i in range(10)] + \
                    [(0.02 * i + 0.1) for i in range(5)] + \
                    [(0.04 * i + 0.2) for i in range(5)] + \
                    [(0.05 * i + 0.36) for i in range(4)]
        else:
            ecc_ls = np.abs(0.2*np.random.randn(25))
            ecc_ls = ecc_ls[ecc_ls<0.5]

        for ecc in ecc_ls:
            wfE = Gfunc(m1, m2, s1z, s2z, D, ecc, srate, flcut, 2, 2, jobtag = jobtag, timeout = timeout)
            if isinstance(wfE, CEV):
                sys.stderr.write('Error: m1, m2, s1z, s2z, ecc = %.3f, %.3f, %.3f, %.3f, %.3f\n'%(m1, m2, s1z, s2z, ecc))
                continue
            wfE = h22base(wfE[0], wfE[1], wfE[2], srate)
            FF, tmove, fs,  _1, _2 = calculate_overlap(wfC, wfE, psd = psd)
            if _1 != CEV.SUCCESS.value or _2 != CEV.SUCCESS.value:
                Eps = 1
            else:
                Eps = 1 - np.abs(FF).max()
            add_csv(fout, data = [[q, s1z, s2z, ecc, Eps]])
            sys.stderr.write('Results: m1, m2, s1z, s2z, ecc, eps = %.3f, %.3f, %.3f, %.3f, %.3f, %.3e\n'%(m1, m2, s1z, s2z, ecc, Eps))

    return 0

