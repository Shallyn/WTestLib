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
from .MultiGrid import MultiGrid1D, MultiGrid
from pathlib import Path

def alignment(wfA, wfB, ithpeak = None):
    fs_A = wfA.srate
    fs_B = wfB.srate
    if fs_A != fs_B:
        return None, None
    if ithpeak is None:
        ipeak_A = wfA.argpeak
    else:
        ipeak_A = ithpeak
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

    parser.add_option('--per-start', type = 'float', default = 0.1, help = 'start index percent')
    parser.add_option('--per-end', type = 'float', default = 0.9, help = 'end index percent')

    parser.add_option('--num-ecc', type = 'int', default = 50, help = 'numbers for grid search')
    parser.add_option('--max-ecc', type = 'float', help = 'Upper bound of parameters 5')
    parser.add_option('--min-ecc', type = 'float', help = 'Lower bound of parameters 5')

    parser.add_option('--num-dtpeak', type = 'int', help = 'numbers for grid search')
    parser.add_option('--max-dtpeak', type = 'float', help = 'Upper bound of parameters 4')
    parser.add_option('--min-dtpeak', type = 'float', help = 'Lower bound of parameters 4')

    parser.add_option('--eps', type = 'float', default = 1e-6, help = 'Thresh of div')
    parser.add_option('--mag', type = 'float', default = 10, help = 'Thresh of dx_init / dx (>1)')
    parser.add_option('--filter-thresh', type = 'float', default = 0.4, help = 'Thresh of grid search (<1)')
    parser.add_option('--max-step', type = 'int', default = 100, help = 'Max iter depth')

    parser.add_option('--plot', action = 'store_true', help = 'recover results')
    parser.add_option('--full-waveform', action = 'store_true', help = 'compare full waveform')
    parser.add_option('--compare-ff', action = 'store_true', help = 'just compare FF')
    parser.add_option('--testecc', type = 'float', help = 'used for test')
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
    dtpeak_default = pms0[3]
    per_start = args.per_start
    per_end = args.per_end
    FIT2D = False
    if args.num_dtpeak is not None:
        FIT2D = True
        num_dtpeak = args.num_dtpeak
        max_dtpeak = args.max_dtpeak if args.max_dtpeak is not None else 100
        min_dtpeak = args.min_dtpeak if args.min_dtpeak is not None else -10
        dtpeak_range = (min_dtpeak, max_dtpeak)
    if per_start > per_end or np.abs(per_start - 0.5) > 0.5 or np.abs(per_end - 0.5) > 0.5:
        raise Exception('Invalid per_start or per_end')
    is_full = args.full_waveform
    if is_full:
        ProgRet = 0
    else:
        ProgRet = -1
    is_comp_FF = args.compare_ff
    def get_lnprob(ecc, dtpeak = dtpeak_default, is_test = False):
        h22_wf = ge.get_waveform(jobtag = args.jobtag, timeout = args.timeout, verbose = is_test,
                        KK = KK, dSO = dSO, dSS = dSS, dtPeak = dtpeak, ecc = ecc, ret = ProgRet)
        if isinstance(h22_wf, CEV):
            return -65536
        if not is_full:
            NR = SNR.cut_ringdown()
        else:
            NR = SNR.copy()
        psdfunc = psd
        # Check sample rate
        Mtotal_init = SNR.Mtotal
        Mtotal_list = np.array([10, 40, 70, 100, 130, 160, 190])
        MtotalFac_list = 1 - Mtotal_list / np.max(Mtotal_list)

        # if FIT2D:
        #     thpeak = np.abs(h22_wf.t0)
        #     ithpeak = int(thpeak * h22_wf.srate)
        # else:
        #     ithpeak = None
        wf_1, wf_2 = alignment(h22_wf, NR, None)


        fs = wf_1.srate
        NFFT = len(wf_1)
        df_old = fs/NFFT
        htilde_1 = wf_1.h22f
        htilde_2 = wf_2.h22f

        idxPeak = wf_2.argpeak
        idx_start = int(per_start*idxPeak)
        idx_end = int(per_end*idxPeak)

        timeH = wf_1.time[idx_start:idx_end]
        trange = timeH[-1] - timeH[0]
        dPhiCum = (wf_1.phaseFrom0[idx_start:idx_end] - wf_1.phaseFrom0[idx_start]) - (wf_2.phaseFrom0[idx_start:idx_end] - wf_2.phaseFrom0[idx_start])
        dAmpCum = (wf_1.amp[idx_start:idx_end] - wf_2.amp[idx_start:idx_end]) / wf_1.amp[idx_start:idx_end] / 0.05
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
            if is_comp_FF:
                idx = np.where(Oxt_abs == max(Oxt_abs))[0][0]
                lth = len(Oxt_abs)
                if idx == lth-1 or idx == 1:
                    tc = 0
                else:
                    if idx > lth / 2:
                        tc = (idx - lth) / fs
                    else:
                        tc = idx / fs
                FF = max(Oxt_abs)
                tc = tc*dim_t(Mtotal)
                eps = (1-FF)
                FF_list.append(-pow(eps/0.01,2) - pow(tc/5, 2))
                lnprob += -pow(eps/0.01, 2)
            else:
                Mfac = MtotalFac_list[i]
                FF = max(Oxt_abs)
                eps = (1 - FF) * Mfac
                FF_list.append(FF * Mfac)
                lnprob += -pow(eps/0.02, 2)
        if is_test:
            return lnprob, FF_list, dPhiCum, dAmpCum
        if is_comp_FF:
            return min(FF_list)
        lnprob += -dPhiCum-dAmpCum
        return lnprob
    
    if args.testecc is not None:
        ret = get_lnprob(args.testecc, is_test = True)
        print(ret)
        return 0
    max_ecc = args.max_ecc if args.max_ecc is not None else 0.5
    min_ecc = args.min_ecc if args.min_ecc is not None else 0
    if e0 < 0 and min_ecc >= 0:
        _tmp = max_ecc
        max_ecc = -min_ecc
        min_ecc = -_tmp
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
        if FIT2D:
            MG = MultiGrid(get_lnprob, ecc_range, dtpeak_range, num_ecc, num_dtpeak)
            MG.run(fsave, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
        else:
            MG = MultiGrid1D(get_lnprob, ecc_range, num_ecc)
            MG.run(fsave, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
    data = np.loadtxt(fsave)
    if FIT2D:
        ecc_list, dtpeak_list, lnp_list = data[:,0], data[:,1], data[:,2]
        ind = np.argmax(lnp_list)
        ecc_fit = ecc_list[ind]
        dtpeak_fit = dtpeak_list[ind]
        print(f'best fit: {(ecc_fit, dtpeak_fit, lnp_list[ind])}')
    else:
        ecc_list, lnp_list = data[:,0], data[:,1]
        ind = np.argmax(lnp_list)
        ecc_fit = ecc_list[ind]
        dtpeak_fit = dtpeak_default
        print(f'best fit: {(ecc_fit, lnp_list[ind])}')
    if args.plot:
        h22_wf = ge.get_waveform(jobtag = args.jobtag, timeout = args.timeout, verbose = True,
                        KK = KK, dSO = dSO, dSS = dSS, dtPeak = dtpeak_fit, ecc = ecc_fit, ret = ProgRet, dump = str(prefix))
        if isinstance(h22_wf, CEV):
            return -1
        
        if FIT2D:
            thpeak = np.abs(h22_wf.t0)
            ithpeak = int(thpeak * h22_wf.srate)
        else:
            ithpeak = None
        fwfname = prefix / 'bestfitwaveform.dat'
        np.savetxt(fwfname, np.stack([h22_wf.time + h22_wf.t0, h22_wf.real, h22_wf.imag], axis = 1))
        if not is_full:
            NR = SNR.cut_ringdown()
        else:
            NR = SNR.copy()
        wf_1, wf_2 = alignment(h22_wf, NR, ithpeak)
        idxPeak = wf_2.argpeak
        idx_start = int(per_start*idxPeak)
        idx_end = int(per_end*idxPeak)
        comp_slice = slice(idx_start, idx_end)
        t1 = wf_1.time
        phase1 = wf_1.phaseFrom0 - wf_1.phaseFrom0[idx_start]
        h1 = wf_1.amp * np.exp(1.j * phase1)
        t2 = wf_2.time
        phase2 = wf_2.phaseFrom0 - wf_2.phaseFrom0[idx_start]
        h2 = wf_2.amp * np.exp(1.j * phase2)
        tW = t1[comp_slice]
        Window = 3. * np.power(tW - tW[-1], 2) / np.power(tW[-1] - tW[0], 3)
        dAmpW = (wf_1.amp[comp_slice] - wf_2.amp[comp_slice]) / wf_1.amp[comp_slice]
        dPhiW = (wf_1.phase[comp_slice] - wf_1.phase[idx_start]) - (wf_2.phase[comp_slice] - wf_2.phase[idx_start])

        ampNQC = wf_2.amp[comp_slice] / wf_1.amp[comp_slice]
        phaseNQC = dPhiW

        fnqcname = prefix / 'nqc.dat'
        np.savetxt(fnqcname, np.stack([tW, ampNQC, phaseNQC], axis = 1))

        fcutname = prefix / 'waveform_cut.dat'
        np.savetxt(fcutname, np.stack([tW, wf_1.amp[comp_slice], phase1[comp_slice], wf_2.amp[comp_slice], phase2[comp_slice]], axis = 1))
        xmin = t1[0] - 0.05 * t1[-1]
        xmax = t1[-1] + 0.05 * t1[-1]

        plt.figure(figsize = (14, 9))
        plt.subplot(411)
        if FIT2D:
            plt.title(f'ecc={ecc_fit},dt={dtpeak_fit}')
        else:
            plt.title(f'ecc={ecc_fit}')
        plt.plot(t1, h1.real, label = 'EOB')
        plt.plot(t2, h2.real, label = SXSnum)
        plt.xlim([xmin, xmax])
        plt.legend()

        plt.subplot(412)
        plt.plot(t1, wf_1.amp, label = 'EOB')
        plt.plot(t2, wf_2.amp, label = SXSnum)
        plt.vlines(tW[0], np.min(wf_2.amp), np.max(wf_2.amp), linestyle = '--')
        plt.vlines(tW[-1], np.min(wf_2.amp), np.max(wf_2.amp), linestyle = '--')
        plt.xlim([xmin, xmax])
        plt.legend()

        plt.subplot(413)
        plt.plot(t1, phase1, label = 'EOB')
        plt.plot(t2, phase2, label = SXSnum)
        plt.vlines(tW[0], np.min(phase1), np.max(phase1), linestyle = '--')
        plt.vlines(tW[-1], np.min(phase1), np.max(phase1), linestyle = '--')
        plt.xlim([xmin, xmax])
        plt.legend()

        plt.subplot(414)
        plt.plot(tW, Window, label = 'window')
        plt.plot(tW, np.power(dPhiW, 2), label = 'dPhi')
        plt.plot(tW, np.power(dAmpW, 2), label = 'dAmp')
        plt.yscale('log')
        plt.xlim([xmin, xmax])
        plt.legend()
        plt.savefig(prefix / 'waveform.png', dpi = 200)
        plt.close()

        plt.scatter(ecc_list, lnp_list, marker = '.')
        plt.xlabel('ecc')
        plt.ylabel('lnprob')
        plt.savefig(prefix / 'lnprob.png', dpi = 200)
        plt.close()
    return 0

#-----Recover EOB vs SXS-----#
def GridSearch_KK_noecc(argv = None):
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL
    from .generator import self_adaptivor

    parser = OptionParser(description='Waveform Comparation With SXS')

    parser.add_option('--executable', type = 'str', default = 'lalsim-inspiral', help = 'Exe command')
    parser.add_option('--approx', type = 'str', default = 'SEOBNREv5', help = 'Version of the code')
    parser.add_option('--fini', type = 'float', default = 0, help = 'Initial orbital frequency')
    parser.add_option('--SXS', type = 'str', default = '0071', help = 'SXS template for comparision')
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

    parser.add_option('--num-k', type = 'int', default = 50, help = 'numbers for grid search')
    parser.add_option('--max-k', type = 'float', help = 'Upper bound of parameter')
    parser.add_option('--min-k', type = 'float', help = 'Lower bound of parameter')
    parser.add_option('--dtpeak', type = 'float', default = 0, help = 'dtpeak')
    parser.add_option('--ecc', type = 'float', default = 0, help = 'ecc')
    parser.add_option('--eps', type = 'float', default = 1e-6, help = 'Thresh of div')
    parser.add_option('--mag', type = 'float', default = 10, help = 'Thresh of dx_init / dx (>1)')
    parser.add_option('--filter-thresh', type = 'float', default = 0.4, help = 'Thresh of grid search (<1)')
    parser.add_option('--max-step', type = 'int', default = 100, help = 'Max iter depth')
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

    max_k = args.max_k if args.max_k is not None else 10
    min_k = args.min_k if args.min_k is not None else -10
    k_range = (min_k, max_k)
    num_k = args.num_k
    eps = args.eps
    mag = args.mag
    filter_thresh = args.filter_thresh
    max_step = args.max_step
    NR = SXSh22(SXSnum = SXSnum,
                f_ini = fini,
                Mtotal = mtotal,
                srate = srate,
                srcloc = srcloc,
                table = table,
                srcloc_all = srcloc_all)
    ge = NR.construct_generator(approx, exe, psd = psd)
    pms0 = NR.CalculateAdjParamsV4()
    dSO_default = pms0[1]
    dSS_default = pms0[2]
    def get_lnprob(k):
        ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout,
                    KK = k, dSO = dSO_default, dSS = dSS_default, dtPeak = args.dtpeak, ecc = args.ecc)
        return ret[0]
    prefix = Path(args.prefix)
    fsave = str(prefix / f'grid_{SXSnum}.txt')
    if not prefix.exists():
        prefix.mkdir(parents = True)
    MG = MultiGrid1D(get_lnprob, k_range, num_k)
    MG.run(fsave, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
    return 0

#-----Recover EOB vs SXS-----#
def GridSearch_dt_noecc(argv = None):
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL
    from .generator import self_adaptivor

    parser = OptionParser(description='Waveform Comparation With SXS')

    parser.add_option('--executable', type = 'str', default = 'lalsim-inspiral', help = 'Exe command')
    parser.add_option('--approx', type = 'str', default = 'SEOBNREv5', help = 'Version of the code')
    parser.add_option('--fini', type = 'float', default = 0, help = 'Initial orbital frequency')
    parser.add_option('--SXS', type = 'str', default = '0071', help = 'SXS template for comparision')
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

    parser.add_option('--num-dtpeak', type = 'int', default = 50, help = 'numbers for grid search')
    parser.add_option('--max-dtpeak', type = 'float', help = 'Upper bound of parameter')
    parser.add_option('--min-dtpeak', type = 'float', help = 'Lower bound of parameter')
    parser.add_option('--eps', type = 'float', default = 1e-6, help = 'Thresh of div')
    parser.add_option('--mag', type = 'float', default = 10, help = 'Thresh of dx_init / dx (>1)')
    parser.add_option('--filter-thresh', type = 'float', default = 0.4, help = 'Thresh of grid search (<1)')
    parser.add_option('--max-step', type = 'int', default = 100, help = 'Max iter depth')
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

    max_dtpeak = args.max_dtpeak if args.max_dtpeak is not None else 100
    min_dtpeak = args.min_dtpeak if args.min_dtpeak is not None else -10
    dtpeak_range = (min_dtpeak, max_dtpeak)
    num_dtpeak = args.num_dtpeak
    eps = args.eps
    mag = args.mag
    filter_thresh = args.filter_thresh
    max_step = args.max_step
    NR = SXSh22(SXSnum = SXSnum,
                f_ini = fini,
                Mtotal = mtotal,
                srate = srate,
                srcloc = srcloc,
                table = table,
                srcloc_all = srcloc_all)
    ge = NR.construct_generator(approx, exe, psd = psd)
    pms0 = NR.CalculateAdjParamsV4()
    KK_default = pms0[0]
    dSO_default = pms0[1]
    dSS_default = pms0[2]
    def get_lnprob(dtpeak):
        ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout,
                    KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak, ecc = 0)
        return ret[0]
    prefix = Path(args.prefix)
    fsave = str(prefix / f'grid_{SXSnum}.txt')
    if not prefix.exists():
        prefix.mkdir(parents = True)
    MG = MultiGrid1D(get_lnprob, dtpeak_range, num_dtpeak)
    MG.run(fsave, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
    
    return 0
