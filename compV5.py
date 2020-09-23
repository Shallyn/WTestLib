#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:06:37 2019

@author: drizl
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from .psd import DetectorPSD
from .SXS import DEFAULT_TABLE, DEFAULT_SRCLOC, DEFAULT_SRCLOC_ALL, SXSh22, CEV
from .h22datatype import h22_alignment, dim_t
from .SXSlist import DEFAULT_ECC_ORBIT_DICT
from optparse import OptionParser
from .MultiGrid import MultiGrid1D
from pathlib import Path

def alignment(wfA, wfB):
    fs_A = wfA.srate
    fs_B = wfB.srate
    if fs_A != fs_B:
        return None, None
    ipeak_A = wfA.argpeak
    ipeak_B = wfB.argpeak
    if ipeak_A > ipeak_B:
        idx_A = ipeak_A - ipeak_B
        idx_B = 0
    else:
        idx_A = 0
        idx_B = ipeak_B - ipeak_A
    wfA = wfA[idx_A:]
    wfB = wfB[idx_B:]
    lenA = len(wfA)
    lenB = len(wfB)
    ipeak_A = ipeak_A - idx_A
    ipeak_B = ipeak_B - idx_B
    tail_A = lenA - ipeak_A
    tail_B = lenB - ipeak_B
    if tail_A > tail_B:
        lpad = tail_A - tail_B
        wfB.pad((0,lpad), 'constant')
    else:
        lpad = tail_B - tail_A
        wfA.pad((0,lpad), 'constant')
    return wfA, wfB

DEFAULT_EXEV5 = '/Users/drizl/Documents/2020/EccentricEOBProject/EOBCoreTest/EOBNR/MAIN'

def main(argv = None):
    parser = OptionParser(description='Waveform Comparation With SXS')

    parser.add_option('--executable', type = 'str', default = DEFAULT_EXEV5, help = 'Exe command')
    parser.add_option('--approx', type = 'str', default = 'SEOBNREv5', help = 'Version of the code')
    parser.add_option('--fini', type = 'float', default = 0, help = 'Initial orbital frequency')
    parser.add_option('--SXS', type = 'str', default = '1374', help = 'SXS template for comparision')
    parser.add_option('--mtotal', type = 'float', default = 40, help = 'Total mass')
    parser.add_option('--srate', type = 'float', default = 16384, help = 'Sample rate')

    parser.add_option('--prefix', type = 'str', default = '.', help = 'dir for results saving.')
    parser.add_option('--jobtag', type = 'str', default = '_lnprob', help = 'jobtag.')

    parser.add_option('--psd', type = 'str', help = 'Detector psd.')
    parser.add_option('--flow', type = 'float', default = 0, help = 'Lower frequency cut off for psd.')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')
    parser.add_option('--mode', type = 'str', default = 'all', help = 'Search mode.')

    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')

    parser.add_option('--k', type = 'float', help = 'Parameter K')

    parser.add_option('--num-ecc', type = 'int', default = 50, help = 'numbers for grid search')
    parser.add_option('--max-ecc', type = 'float', help = 'Upper bound of parameters 5')
    parser.add_option('--min-ecc', type = 'float', help = 'Lower bound of parameters 5')

    parser.add_option('--eps', type = 'float', default = 1e-6, help = 'Thresh of div')
    parser.add_option('--mag', type = 'float', default = 10, help = 'Thresh of dx_init / dx (>1)')
    parser.add_option('--filter-thresh', type = 'float', default = 0.4, help = 'Thresh of grid search (<1)')
    parser.add_option('--max-step', type = 'int', default = 100, help = 'Max iter depth')

    parser.add_option('--plot', action = 'store_true', help = 'recover results')
    args, _ = parser.parse_args(argv)

    exe = args.executable
    approx = args.approx
    SXSnum = args.SXS
    mtotal = args.mtotal
    fini = args.fini
    srate = args.srate
    table = args.table
    srcloc = args.srcloc
    srcloc_all = args.srcloc_all
    psd = DetectorPSD(args.psd, flow = args.flow)

    if SXSnum in DEFAULT_ECC_ORBIT_DICT:
        f0, e0 = DEFAULT_ECC_ORBIT_DICT[SXSnum]
    else:
        f0 = fini

    SNR = SXSh22(SXSnum = SXSnum,
                f_ini = f0,
                Mtotal = mtotal,
                srate = srate,
                srcloc = srcloc,
                table = table,
                srcloc_all = srcloc_all)
    ge = SNR.construct_generator(approx, exe, psd = psd)
    pms0 = SNR.CalculateAdjParamsV4()
    KK = args.k if args.k is not None else pms0[0]
    dSO = pms0[1]
    dSS = pms0[2]
    dtpeak = pms0[3]


    def get_lnprob(ecc):
        h22_wf = ge.get_waveform(jobtag = args.jobtag, timeout = args.timeout, verbose = True,
                        KK = KK, dSO = dSO, dSS = dSS, dtPeak = dtpeak, ecc = ecc, ret = -1)
        if isinstance(h22_wf, CEV):
            return -65536

        thpeak = np.abs(h22_wf.t0)
        ithpeak = int(thpeak * h22_wf.srate)
        NR = SNR.cut_ringdown()

        psdfunc = psd
        # Check sample rate
        Mtotal_init = SNR.Mtotal
        Mtotal_list = np.array([10, 40, 70, 100, 130, 160, 190])
        MtotalFac_list = 1 - Mtotal_list / np.max(Mtotal_list)
        wf_1, wf_2 = alignment(h22_wf, NR)

        fs = wf_1.srate
        NFFT = len(wf_1)
        df_old = fs/NFFT
        htilde_1 = wf_1.h22f
        htilde_2 = wf_2.h22f

        idxPeak = min(wf_1.argpeak, wf_2.argpeak)
        idx_start = int(0.1*idxPeak)
        idx_end = int(0.9*idxPeak)

        timeH = wf_1.time[idx_start:idx_end] * dim_t(Mtotal_init)
        trange = timeH[-1] - timeH[0]
        dPhiCum = (wf_1.phase[idx_start:idx_end] - wf_1.phase[idx_start]) - (wf_2.phase[idx_start:idx_end] - wf_2.phase[idx_start])
        dAmpCum = wf_1.amp[idx_start:idx_end] - wf_2.amp[idx_start:idx_end]
        Pre = 3. * np.power(timeH - timeH[-1], 2) / np.power(trange, 3)

        if trange == 0:
            return -65536

        dPhiCum = np.sum(Pre * np.power(dPhiCum, 2))
        dAmpCum = np.sum(Pre * np.power(dAmpCum, 2))
        lnprob = 0
        FF_list = []
        # eps_lst = []
        for i, Mtotal in enumerate(Mtotal_list):
            df = df_old *  Mtotal_init / Mtotal
            fs = df * NFFT
            freqs = np.abs(np.fft.fftfreq(NFFT, 1./fs))
            power_vec = psdfunc(freqs)
            O11 = np.sum(htilde_1 * htilde_1.conjugate() / power_vec).real * df
            O22 = np.sum(htilde_2 * htilde_2.conjugate() / power_vec).real * df
            Ox = htilde_1 * htilde_2.conjugate() / power_vec
            Oxt = np.fft.ifft(Ox) * fs / np.sqrt(O11 * O22)
            Oxt_abs = np.abs(Oxt)
            Mfac = MtotalFac_list[i]
            FF = max(Oxt_abs)
            eps = (1 - FF) * Mfac
            FF_list.append(FF * Mfac)
            lnprob += -pow(eps/0.03, 2)
        lnprob += -dPhiCum-dAmpCum
        return lnprob
    if e0 > 0:
        max_ecc = args.max_ecc if args.max_ecc is not None else 0.5
        min_ecc = args.min_ecc if args.min_ecc is not None else 0
    else:
        max_ecc = args.max_ecc if args.max_ecc is not None else 0
        min_ecc = args.min_ecc if args.min_ecc is not None else -0.5
    num_ecc = args.num_ecc
    ecc_range = (min_ecc, max_ecc)
    eps = args.eps
    mag = args.mag
    filter_thresh = args.filter_thresh
    max_step = args.max_step

    prefix = Path(args.prefix)
    fsave = str(prefix / f'grid_{SXSnum}.txt')
    if not prefix.exists():
        prefix.mkdir(parents = True)
    
    if not Path(fsave).exists():
        MG = MultiGrid1D(get_lnprob, ecc_range, num_ecc)
        MG.run(fsave, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
    data = np.loadtxt(fsave)
    ecc_list, lnp_list = data[:,0], data[:,1]
    ind = np.argmax(lnp_list)
    ecc = ecc_list[ind]
    print(f'best fit: {(ecc, lnp_list[ind])}')
    if args.plot:
        h22_wf = ge.get_waveform(jobtag = args.jobtag, timeout = args.timeout, verbose = True,
                        KK = KK, dSO = dSO, dSS = dSS, dtPeak = dtpeak, ecc = ecc, ret = -1, dump = str(prefix))
        if isinstance(h22_wf, CEV):
            return -1
        NR = SNR.cut_ringdown()
        wf_1, wf_2 = alignment(h22_wf, NR)
        idxPeak = min(wf_1.argpeak, wf_2.argpeak)
        idx_start = int(0.1*idxPeak)
        t1 = wf_1.time
        phase1 = wf_1.phase - wf_1.phase[idx_start]
        h1 = wf_1.amp * np.exp(1.j * phase1)
        t2 = wf_2.time
        phase2 = wf_2.phase - wf_2.phase[idx_start]
        h2 = wf_2.amp * np.exp(1.j * phase2)
        Window = 3. * np.power(t1 - t1[-1], 2) / np.power(t1[-1] - t1[0], 3)
        plt.figure(figsize = (14, 9))
        plt.title(f'ecc={ecc}')
        plt.subplot(411)
        plt.plot(t1, h1.real, label = 'EOB')
        plt.plot(t2, h2.real, label = SXSnum)
        plt.legend()
        plt.subplot(412)
        plt.plot(t1, wf_1.amp, label = 'EOB')
        plt.plot(t2, wf_2.amp, label = SXSnum)
        plt.legend()
        plt.subplot(413)
        plt.plot(t1, phase1, label = 'EOB')
        plt.plot(t2, phase2, label = SXSnum)
        plt.legend()
        plt.subplot(414)
        plt.plot(t1, Window, label = 'window')
        plt.legend()
        plt.savefig(prefix / 'waveform.png', dpi = 200)
        plt.close()

        plt.scatter(ecc_list, lnp_list, marker = '.')
        plt.xlabel('ecc')
        plt.ylabel('lnprob')
        plt.savefig(prefix / 'lnprob.png', dpi = 200)
        plt.close()
    return 0

