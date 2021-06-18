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
from .Utils import switch, SpinWeightedM2SphericalHarmonic
from .SXS import DEFAULT_TABLE, DEFAULT_SRCLOC, DEFAULT_SRCLOC_ALL, SXSh22, CEV, SXSAllMode, Generator, waveform_mode_collector, LOG
from .h22datatype import h22_alignment, dim_t, ModeBase, Mode_alignment, calculate_ModeFF
from .SXSlist import DEFAULT_ECC_ORBIT_DICT, DEFAULT_ECC_ORBIT_DICT_V5
from optparse import OptionParser
from .MultiGrid import MultiGrid1D, MultiGrid
from pathlib import Path
from itertools import product
import multiprocessing as mp

class V5Dynamics(object):
    def __init__(self, dydata):
        #time #r #phi #dr #dphi #prT #pphi #dprT #dpphi #ham
        self._time = dydata[:,0]
        self._r = dydata[:,1]
        self._phi = dydata[:,2]
        self._dr = dydata[:,3]
        self._dphi = dydata[:,4]
        self._prT = dydata[:,5]
        self._pphi = dydata[:,6]
        self._dprT = dydata[:,7]
        self._dpphi = dydata[:,8]
        self._ham = dydata[:,9]
    @property
    def time(self):
        return self._time
    @property
    def r(self):
        return self._r
    @property
    def phi(self):
        return self._phi
    @property
    def dr(self):
        return self._dr
    @property
    def dphi(self):
        return self._dphi
    @property
    def xx(self):
        return np.power(self._dphi, 2./3.)
    @property
    def prT(self):
        return self._prT
    @property
    def pphi(self):
        return self._pphi
    @property
    def dprT(self):
        return self._dprT
    @property
    def dpphi(self):
        return self._dpphi
    @property
    def ham(self):
        return self._ham

def alignment(wfA, wfB, ithpeak = None, ret_tmove = False):
    #if cut wfA, tmove > 0, else tmove < 0
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
    tmove = (ipeak_A - ipeak_B) / fs_A
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
    if ret_tmove:
        return wfA, wfB, tmove
    return wfA, wfB

def get_new_dtpeak_nospin_Nv1(eta):
    return 2.50373124 + 166.24461103 * eta -1097.73967883*np.power(eta,2) + 1753.20870987 * np.power(eta,3)
def get_new_dtpeak_nospin_Nv1nhm(eta):
    return 2.42676826  +  18.46995727 *eta +  188.80152615*np.power(eta,2) -1060.72511485 *np.power(eta,3)
def get_new_dtpeak_nospin_Nv3(eta):
    return 2.45459198 + 33.91443053 * eta + 186.55163593*np.power(eta,2) -1168.50878255 * np.power(eta,3)
def calc_dt_nsNv5Av2(eta):
    return 2.45368344 + 49.77169908 *eta + 48.94536063*np.power(eta,2) -702.77321007*np.power(eta,3)
def get_ecc_from_SXSid_Nv1A2_dtV4(SXSid):
    Sdict = {'0066': 0.0, '0168': 0.0, '0056': 0.0, '0070': 0.0, '0183': 0.0, '0166': 0.0, '0071': 0.0, '0167': 0.0, '0169': 0.0, '0182': 0.0, '0298': 0.0, '0184': 0.0, '0055': 0.0, '0063': 0.0, '0301': 0.0, '1365': 0.07612456747404844, '1366': 0.11764705882352941, '1360': 0.16991926182237602, '1370': 0.24447520184544405, '1367': 0.11888888888888888, '1371': 0.06920415224913494, '1368': 0.11568627450980393, '1372': 0.11411764705882352, '1355': -0.06443829296424453, '1361': 0.17151095732410612, '1373': 0.11227220299884658, '1356': 0.11456747404844289, '1369': 0.22047289504036907, '1362': 0.2226066897347174, '1357': 0.1237985390234525, '1374': 0.21116493656286045, '1363': 0.22346020761245675, '1358': 0.14648212226066898, '1364': 0.05843906189926951, '1359': 0.12265667051134181}
    if SXSid in Sdict:
        return Sdict[SXSid]
    else:
        return 0.0

def get_ecc_range(SXSnum, min_ecc = None, max_ecc = None, fini = None):
    if SXSnum not in DEFAULT_ECC_ORBIT_DICT:
        return None, -0.05, 0.05
    f0, e0 = DEFAULT_ECC_ORBIT_DICT[SXSnum]
    min_e = e0 - 0.12
    max_e = e0 + 0.12
    if e0 > 0 and min_e < 0:
        min_e = 0
    if e0 < 0 and max_e > 0:
        max_e = 0
    min_e = min_ecc if min_ecc is not None else min_e
    max_e = max_ecc if max_ecc is not None else max_e
    f0 = fini if fini is not None else f0
    return f0, min_e, max_e

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
    parser.add_option('--delta-ecc', type = 'float', help = 'delta ecc around e0')
    parser.add_option('--num-dtpeak', type = 'int', help = 'numbers for grid search')
    parser.add_option('--max-dtpeak', type = 'float', help = 'Upper bound of parameters 4')
    parser.add_option('--min-dtpeak', type = 'float', help = 'Lower bound of parameters 4')
    parser.add_option('--delta-dtpeak', type = 'float', help = 'delta dtpeak around dt_v4')

    parser.add_option('--eps', type = 'float', default = 1e-6, help = 'Thresh of div')
    parser.add_option('--mag', type = 'float', default = 10, help = 'Thresh of dx_init / dx (>1)')
    parser.add_option('--filter-thresh', type = 'float', default = 0.4, help = 'Thresh of grid search (<1)')
    parser.add_option('--max-step', type = 'int', default = 100, help = 'Max iter depth')

    parser.add_option('--plot', action = 'store_true', help = 'plot')

    parser.add_option('--full-waveform', action = 'store_true', help = 'compare full waveform')
    parser.add_option('--compare-ff', action = 'store_true', help = 'just compare FF')
    parser.add_option('--testecc', type = 'float', help = 'used for test')

    parser.add_option('--version', type = 'str', default = 'default', help = 'code version')
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

    if SXSnum in DEFAULT_ECC_ORBIT_DICT_V5:
        f0, e0 = DEFAULT_ECC_ORBIT_DICT_V5[SXSnum]
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
    version = args.version.lower()
    if args.num_dtpeak is not None:
        FIT2D = True
        num_dtpeak = args.num_dtpeak
        max_dtpeak = args.max_dtpeak if args.max_dtpeak is not None else 100
        min_dtpeak = args.min_dtpeak if args.min_dtpeak is not None else -10
        if args.delta_dtpeak:
            for case in switch(version):
                if case('nv1'):
                    dt_v4 = get_new_dtpeak_nospin_Nv1(SNR.eta)
                    break
                if case('nv1nohm'):
                    dt_v4 = get_new_dtpeak_nospin_Nv1nhm(SNR.eta)
                    break
                if case('nv5av2'):
                    dt_v4 = calc_dt_nsNv5Av2(SNR.eta)
                    break
                else:
                    dt_v4 = SNR.CalculateAdjParamsV4()[3]
                    break
            min_dtpeak = max(0, dt_v4 - args.delta_dtpeak)
            max_dtpeak = dt_v4 + args.delta_dtpeak
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
        h22_wf = ge.get_waveform(jobtag = args.jobtag, timeout = args.timeout, verbose = True,
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
        try:
            trange = timeH[-1] - timeH[0]
            dPhiCum = (wf_1.phaseFrom0[idx_start:idx_end] - wf_1.phaseFrom0[idx_start]) - (wf_2.phaseFrom0[idx_start:idx_end] - wf_2.phaseFrom0[idx_start])
            dAmpCum = (wf_1.amp[idx_start:idx_end] - wf_2.amp[idx_start:idx_end]) / wf_1.amp[idx_start:idx_end] / 0.05
            Pre = 3. * np.power(timeH - timeH[-1], 2) / np.power(trange, 3)
        except:
            sys.stderr.write(f'len_wf = {len(wf_1)}, idxPeak = {idxPeak}\n')
            trange = 0
            dPhiCum = 0.0
            dAmpCum = 0.0
            Pre = 0.0

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
    if args.delta_ecc is not None:
        max_ecc = e0 + args.delta_ecc
        min_ecc = e0 - args.delta_ecc
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

#-----Recover EOB vs SXS-----#

def GridSearch_eccV2(argv = None):
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL
    from .SXS import save_namecol, add_csv
    from .generator import self_adaptivor

    parser = OptionParser(description='Waveform Comparation With SXS')

    parser.add_option('--executable', type = 'str', default = DEFAULT_EXEV5, help = 'Exe command')
    parser.add_option('--approx', type = 'str', default = 'SEOBNREv5', help = 'Version of the code')
    parser.add_option('--fini', type = 'float', help = 'Initial orbital frequency')
    parser.add_option('--SXS', type = 'str', default = '1374', help = 'SXS template for comparision')
    parser.add_option('--mtotal', type = 'float', default = 40, help = 'Total mass')
    parser.add_option('--srate', type = 'float', default = 16384, help = 'Sample rate')
 
    parser.add_option('--prefix', type = 'str', default = '.', help = 'dir for results saving.')
    parser.add_option('--jobtag', type = 'str', default = '_lnprob', help = 'jobtag.')

    parser.add_option('--psd', type = 'str', help = 'Detector psd.')
    parser.add_option('--flow', type = 'float', default = 0, help = 'Lower frequency cut off for psd.')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')
    parser.add_option('--oldecc', type = 'str', help = 'use old ecc or not')

    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')

    parser.add_option('--num-ecc', type = 'int', default = 50, help = 'numbers for grid search')
    parser.add_option('--max-ecc', type = 'float', help = 'Upper bound of parameter')
    parser.add_option('--min-ecc', type = 'float', help = 'Lower bound of parameter')
    parser.add_option('--ecc', type = 'float', help = 'The specific ecc')
    parser.add_option('--eps', type = 'float', default = 1e-6, help = 'Thresh of div')
    parser.add_option('--mag', type = 'float', default = 10, help = 'Thresh of dx_init / dx (>1)')
    parser.add_option('--filter-thresh', type = 'float', default = 0.4, help = 'Thresh of grid search (<1)')
    parser.add_option('--max-step', type = 'int', default = 100, help = 'Max iter depth')
    parser.add_option('--version', type = 'str', default = 'default', help = 'code version')
    parser.add_option('--only22', action = 'store_true', help = 'only use 22 mode')
    parser.add_option('--circ', action = 'store_true', help = 'force ecc = 0')
    parser.add_option('--cutpct', type = 'float', default = 0, help = 'cut the NR waveform')
    parser.add_option('--plot', action = 'store_true', help = 'code version')
    parser.add_option('--plot-thresh', type = 'float', help = 'plot when FF < ..')
    args, _ = parser.parse_args(argv)

    exe = args.executable
    approx = args.approx
    SXSid = args.SXS
    mtotal = args.mtotal
    fini = args.fini
    srate = args.srate
    table = args.table

    srcloc_all = args.srcloc_all
    psd = DetectorPSD(args.psd, flow = args.flow)
    eps = args.eps
    mag = args.mag
    filter_thresh = args.filter_thresh
    max_step = args.max_step


    prefix_all = Path(args.prefix)
    if not prefix_all.exists():
        prefix_all.mkdir(parents=True)

    return 0

def GridSearch_ecc(argv = None):
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL
    from .SXS import save_namecol, add_csv
    from .generator import self_adaptivor

    parser = OptionParser(description='Waveform Comparation With SXS')

    parser.add_option('--executable', type = 'str', default = DEFAULT_EXEV5, help = 'Exe command')
    parser.add_option('--approx', type = 'str', default = 'SEOBNREv5', help = 'Version of the code')
    parser.add_option('--fini', type = 'float', help = 'Initial orbital frequency')
    parser.add_option('--SXS', type = 'str', action = 'append', default = [], help = 'SXS template for comparision')
    parser.add_option('--mtotal', type = 'float', default = 40, help = 'Total mass')
    parser.add_option('--srate', type = 'float', default = 16384, help = 'Sample rate')
 
    parser.add_option('--prefix', type = 'str', default = '.', help = 'dir for results saving.')
    parser.add_option('--jobtag', type = 'str', default = '_lnprob', help = 'jobtag.')

    parser.add_option('--psd', type = 'str', help = 'Detector psd.')
    parser.add_option('--flow', type = 'float', default = 0, help = 'Lower frequency cut off for psd.')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')
    parser.add_option('--oldecc', type = 'str', help = 'use old ecc or not')

    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc22', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc21', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc33', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc44', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')

    parser.add_option('--num-ecc', type = 'int', default = 50, help = 'numbers for grid search')
    parser.add_option('--max-ecc', type = 'float', help = 'Upper bound of parameter')
    parser.add_option('--min-ecc', type = 'float', help = 'Lower bound of parameter')
    parser.add_option('--ecc', type = 'float', help = 'The specific ecc')
    parser.add_option('--eps', type = 'float', default = 1e-6, help = 'Thresh of div')
    parser.add_option('--mag', type = 'float', default = 10, help = 'Thresh of dx_init / dx (>1)')
    parser.add_option('--filter-thresh', type = 'float', default = 0.4, help = 'Thresh of grid search (<1)')
    parser.add_option('--max-step', type = 'int', default = 100, help = 'Max iter depth')
    parser.add_option('--version', type = 'str', default = 'default', help = 'code version')
    parser.add_option('--only22', action = 'store_true', help = 'only use 22 mode')
    parser.add_option('--circ', action = 'store_true', help = 'force ecc = 0')
    parser.add_option('--cutpct', type = 'float', default = 0, help = 'cut the NR waveform')
    parser.add_option('--plot', action = 'store_true', help = 'code version')
    parser.add_option('--plot-thresh', type = 'float', help = 'plot when FF < ..')
    args, _ = parser.parse_args(argv)

    exe = args.executable
    approx = args.approx
    SXSnum_list = args.SXS
    if len(SXSnum_list) == 0:
        SXSnum_list.append('0001')
    mtotal = args.mtotal
    fini = args.fini
    srate = args.srate
    table = args.table
    srcloc22 = args.srcloc22
    srcloc21 = args.srcloc21
    srcloc33 = args.srcloc33
    srcloc44 = args.srcloc44

    srcloc_all = args.srcloc_all
    psd = DetectorPSD(args.psd, flow = args.flow)
    eps = args.eps
    mag = args.mag
    filter_thresh = args.filter_thresh
    max_step = args.max_step


    prefix_all = Path(args.prefix)
    if not prefix_all.exists():
        prefix_all.mkdir(parents=True)
    ymodeDict = {22: (srcloc22, prefix_all / 'collect_22.csv', prefix_all / 'MtotalFF_22'),
                 21: (srcloc21, prefix_all / 'collect_21.csv', prefix_all / 'MtotalFF_21'),
                 33: (srcloc33, prefix_all / 'collect_33.csv', prefix_all / 'MtotalFF_33'),
                 44: (srcloc44, prefix_all / 'collect_44.csv', prefix_all / 'MtotalFF_44')}
    collect_title = [['#SXSid', '#q', '#chi1x', '#chi1y', '#chi1z', '#chi2x', '#chi2y', '#chi2z', '#chiX', '#ecc', '#fini', '#FF', '#lnp']]
    for ymode in ymodeDict:
        _, fname_collect, prefixM = ymodeDict[ymode]
        if not fname_collect.exists():
            save_namecol(fname_collect, collect_title)
        if not prefixM.exists():
            prefixM.mkdir(parents = True)
    for SXSnum in SXSnum_list:
        sys.stderr.write(f'NOW SXS:BBH:{SXSnum}:\n\n')
        oldecc = None
        fini_use = None
        prefixSXS = prefix_all / SXSnum
        if not prefixSXS.exists():
            prefixSXS.mkdir(parents = True)
        with open(prefixSXS / 'mdebug.txt', 'w') as f:
            f.write('DEBUG MESSAGE\n')
        # if not prefixSXS.exists():
        #     prefixSXS.mkdir(parents=True)
        mDebug = f'SXS_BBH_{SXSnum}:\n'
        for ymode in ymodeDict:
            if args.only22 and ymode != 22:
                continue
            mDebug += f'mode: Y{ymode}:\n'
            srcloc, fname_collect, prefixM = ymodeDict[ymode]
            f0, min_e, max_e = get_ecc_range(SXSnum, args.min_ecc, args.max_ecc)
            mDebug += f'f0 = {f0}\nmin_e={min_e}\nmax_e={max_e}\n'
            if f0 is not None:
                fini = f0
                max_ecc = max_e
                min_ecc = min_e
                ecc_range = (min_ecc, max_ecc)
            if fini is None:
                fini = 0
            num_ecc = args.num_ecc
            NR = SXSh22(SXSnum = SXSnum,
                        f_ini = fini,
                        Mtotal = mtotal,
                        srate = srate,
                        srcloc = srcloc,
                        table = table,
                        srcloc_all = srcloc_all, cutpct = args.cutpct, ymode = ymode)
            if (ymode % 10) % 2 and NR.q == 1 and NR.s1z ==  NR.s2z:
                continue
            ge = NR.construct_generator(approx, exe, psd = psd)
            pms0 = NR.CalculateAdjParamsV4()
            KK_default = pms0[0]
            dSO_default = pms0[1]
            dSS_default = pms0[2]
            version = args.version.lower()
            for case in switch(version):
                if case('nv1'):
                    dtpeak_fit = get_new_dtpeak_nospin_Nv1(NR.eta)
                    break
                if case('nv1nohm'):
                    dtpeak_fit = get_new_dtpeak_nospin_Nv1nhm(NR.eta)
                    break
                if case('nv5av2'):
                    dtpeak_fit = calc_dt_nsNv5Av2(NR.eta)
                    break
                else:
                    dtpeak_fit = pms0[3]
                    break

            def get_lnprob(ecc):
                ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = ymode,
                            KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak_fit, ecc = ecc)
                return ret[0]
            def get_lnprob_fini(fini):
                ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = ymode, fini = fini,
                            KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak_fit, ecc = 0.0)
                return ret[0]
            def get_lnprob_ecc_fini(ecc, fini):
                ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = ymode, fini = fini,
                            KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak_fit, ecc = ecc)
                return ret[0]

            if args.ecc is not None:
                ecc = args.ecc
            elif SXSnum not in DEFAULT_ECC_ORBIT_DICT or args.circ:
                ecc = 0.0
                if ge.SXS.is_prec and fini_use is None:
                    fini_range = (ge.SXS.f_ini*0.75, ge.SXS.f_ini*1.25)
                    MG = MultiGrid1D(get_lnprob_fini, fini_range, 20)
                    data = MG.run(None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
                    fini_grid, lnp_grid = data[:,0], data[:,1]
                    fini_use = fini_grid[np.argmax(lnp_grid)]
            elif oldecc is not None:
                ecc = oldecc
            else:
                if ge.SXS.is_prec and fini_use is None:
                    prefix = prefixSXS / f'mode_{ymode}'
                    if not prefix.exists():
                        prefix.mkdir(parents=True)
                    fsave = str(prefix / f'grid_{SXSnum}.txt')
                    fini_range = (ge.SXS.f_ini*0.75, ge.SXS.f_ini*1.25)
                    MG = MultiGrid(get_lnprob_ecc_fini, ecc_range, fini_range, num_ecc, 20)
                    MG.run(fsave, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
                    data = np.loadtxt(fsave)
                    ecc_grid, fini_grid, lnp_grid = data[:,0], data[:,1], data[:,2]
                    ecc = ecc_grid[np.argmax(lnp_grid)]
                    fini_use = fini_grid[np.argmax(lnp_grid)]
                    oldecc = ecc
                elif not ge.SXS.is_prec:
                    prefix = prefixSXS / f'mode_{ymode}'
                    if not prefix.exists():
                        prefix.mkdir(parents=True)
                    fsave = str(prefix / f'grid_{SXSnum}.txt')
                    if not prefix.exists():
                        prefix.mkdir(parents = True)
                    MG = MultiGrid1D(get_lnprob, ecc_range, num_ecc)
                    MG.run(fsave, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
                    data = np.loadtxt(fsave)
                    ecc_grid, lnp_grid = data[:,0], data[:,1]
                    ecc = ecc_grid[np.argmax(lnp_grid)]
                    oldecc = ecc
            mDebug += f'fini_use : {fini_use}\necc : {ecc}\n'
            lnp, FF = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = ymode, fini = fini_use,
                            KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak_fit, ecc = ecc)
            if args.plot and (args.plot_thresh is None or FF < args.plot_thresh):
                # if not prefixSXS.exists():
                #     prefixSXS.mkdir(parents=True)
                prefix = prefixSXS / f'mode_{ymode}'
                if not prefix.exists():
                    prefix.mkdir(parents=True)
                h22_wf = ge.get_waveform(jobtag = args.jobtag, timeout = args.timeout, verbose = True,
                                KK = KK_default, dSO = dSO_default, dSS = dSS_default, fini = fini_use,
                                dtPeak = dtpeak_fit, ecc = ecc, mode = ymode, dump = str(prefixSXS))
                wf_1, wf_2, tmove = alignment(h22_wf, NR, ret_tmove = True)
                np.savetxt(prefix / 'waveform.dat', np.stack([wf_1.time, wf_1.real, wf_1.imag, wf_2.real, wf_2.imag], axis = 1))
                if tmove < 0:
                    tmove = 0.0
                dAmp = wf_1.amp - wf_2.amp
                dPhase = wf_1.phaseFrom0 - wf_2.phaseFrom0
                fig = plt.figure(figsize = (14, 7))
                ax1 = fig.add_subplot(211)
                ax1.set_title(f'lnp={lnp},FF={FF}')
                ax1.plot(wf_1.time, wf_1.amp, label = f'EOB_{ymode}')
                ax1.plot(wf_2.time, wf_2.amp, label = f'NR_{ymode}')
                ax2 = ax1.twinx()
                ax2.plot(wf_1.time, dAmp, color = 'black', linestyle = '--', alpha = 0.7)
                ax1.legend()
                ax1.grid()

                ax3 = fig.add_subplot(212)
                ax3.plot(wf_1.time, wf_1.phaseFrom0, label = f'EOB_{ymode}')
                ax3.plot(wf_2.time, wf_2.phaseFrom0, label = f'NR_{ymode}')
                ax4 = ax3.twinx()
                ax4.plot(wf_1.time, dPhase, color = 'black', linestyle = '--', alpha = 0.7)
                ax3.legend()
                ax3.grid()
                plt.savefig(prefix / f'AmpPhase.png', dpi = 200)
                plt.close()

                ipeak = wf_2.argpeak
                h1 = wf_1.amp * np.exp(-1.j*(wf_1.phase - wf_1.phase[ipeak]))
                h2 = wf_2.amp * np.exp(-1.j*(wf_2.phase - wf_2.phase[ipeak]))
                plt.figure(figsize = (14, 7))
                plt.subplot(211)
                plt.title(f'eta={NR.eta}, chi={NR.chiX}')
                plt.plot(wf_1.time, h1.real, label = f'EOB_{ymode}')
                plt.plot(wf_2.time, h2.real, label = f'NR_{ymode}')
                plt.legend()
                plt.grid()
                plt.subplot(212)
                plt.title(f'chi1={NR.s1z}, chi2={NR.s2z}')
                plt.plot(wf_1.time, h1.imag, label = f'EOB_{ymode}')
                plt.plot(wf_2.time, h2.imag, label = f'NR_{ymode}')
                plt.legend()
                plt.grid()
                plt.savefig(prefix / f'RealImag.png', dpi = 200)
                plt.close()

                # plot dynamics & nqc
                fLowNQC = prefixSXS / 'waveformLowNQCWindow.dat'
                fLowhNoNQC = prefixSXS / f'waveformLowSRnoNQC_{int(ymode)}.dat'
                fLowDy = prefixSXS / 'dynamics.dat'
                fHigh = prefixSXS / f'waveformHiNoNQC_{int(ymode)}.dat'
                fHiDy = prefixSXS / 'dynamicsHi.dat'
                fRD = prefixSXS / f'RingDown_{int(ymode)}.dat'
                fHighN = prefixSXS / f'waveformHiWithNQC_{int(ymode)}.dat'
                dimt = dim_t(NR.Mtotal)
                if fLowNQC.exists() and fLowhNoNQC.exists() and fLowDy.exists() and fHigh.exists() and fHiDy.exists():
                    data = np.loadtxt(fLowNQC)
                    tNQC, nWind = data[:,0], data[:,1],
                    data = np.loadtxt(fLowhNoNQC)
                    tLow, hrL, hiL = data[:,0], data[:,1], data[:,2]
                    hLow = ModeBase(tLow, hrL, hiL)
                    dataLDY = np.loadtxt(fLowDy)
                    dyLow = V5Dynamics(dataLDY)
                    data = np.loadtxt(fHigh)
                    tHi, hrHi, hiHi = data[:,0], data[:,1], data[:,2]
                    hHi = ModeBase(tHi, hrHi, hiHi)
                    fig = plt.figure(figsize = (16, 12))
                    ax5 = fig.add_subplot(311)
                    ax5.set_title(f'lnp={lnp}, FF={FF}')
                    ax5_ln1 = ax5.plot((wf_1.time + tmove)*dimt, wf_1.amp, label = f'EOB_{ymode}', linestyle = '--', alpha = 0.7)
                    ax5_ln2 = ax5.plot((wf_2.time + tmove)*dimt, wf_2.amp, label = f'NR_{ymode}', alpha = 0.6, color ='black')
                    ax5_ln3 = ax5.plot(tLow, hLow.amp, label = 'ampLowNoNQC')
                    ax6 = ax5.twinx()
                    ax6_ln1 = ax6.plot(tNQC, nWind, label = 'nqcWind', color = 'purple', linestyle = '--', alpha = 0.5)
                    ax56_lns = ax5_ln1 + ax5_ln2 + ax5_ln3 + ax6_ln1
                    ax56_labs = [l.get_label() for l in ax56_lns]
                    ax5.legend(ax56_lns, ax56_labs)
                    ax5.grid()
                    ax5.set_ylabel('h')
                    ax5.set_ylabel('nqcWindow')

                    ax1 = fig.add_subplot(312)
                    ax1.set_title(f'eta={NR.eta}, chi={NR.chiX}')
                    ax1_ln1 = ax1.plot((wf_1.time + tmove)*dimt, wf_1.amp, label = f'EOB_{ymode}', linestyle = '--', alpha = 0.7)
                    ax1_ln2 = ax1.plot((wf_2.time + tmove)*dimt, wf_2.amp, label = f'NR_{ymode}', alpha = 0.6, color ='black')
                    ax1_ln3 = ax1.plot(tLow, hLow.amp, label = 'ampLowNoNQC')
                    ax2 = ax1.twinx()
                    ax2_ln1 = ax2.plot(tNQC, nWind, label = 'nqcWindow', color = 'purple', linestyle = '--', alpha = 0.5)
                    ax12_lns = ax1_ln1 + ax1_ln2 + ax1_ln3 + ax2_ln1
                    ax12_labs = [l.get_label() for l in ax12_lns]
                    ax1.legend(ax12_lns, ax12_labs)
                    ax1.grid()
                    ax1.set_ylabel('h')
                    ax2.set_ylabel('hNQC')

                    ax3 = fig.add_subplot(313)
                    ax3.set_title(f'chi1={NR.s1z}, chi2={NR.s2z}')
                    ax3_ln1 = ax3.plot(dyLow.time, dyLow.r, label = r'$r$', color = 'red')
                    ax4 = ax3.twinx()
                    ax4_ln1 = ax4.plot(tNQC, nWind, label = 'nqcPreO')
                    ax34_lns = ax3_ln1 + ax4_ln1
                    ax34_labs = [l.get_label() for l in ax34_lns]
                    ax3.legend(ax34_lns, ax34_labs)
                    ax4.grid()
                    ax3.set_xlabel('time[M]')
                    ax3.set_ylabel('nqcP')
                    ax4.set_ylabel('r')
                    ax3.set_xlim([tHi[0]*0.99, tHi[-1]*1.005])

                    plt.savefig(prefix / 'dyNQCLow.png', dpi = 200)
                    plt.close()
                    os.system(f'rm {fLowDy}')
                    os.system(f'rm {fLowhNoNQC}')
                    os.system(f'rm {fLowNQC}')

                    data = np.loadtxt(fHiDy)
                    dyHi = V5Dynamics(data)
                    
                    fig = plt.figure(figsize = (10, 10))
                    ax1 = fig.add_subplot(211)
                    ax1.set_title(f'ecc={ecc}')
                    ax1_ln1 = ax1.plot((wf_1.time+tmove)*dimt, wf_1.amp, label = f'EOB_{ymode}', linestyle = '--', alpha = 0.7)
                    ax1_ln2 = ax1.plot((wf_2.time+tmove)*dimt, wf_2.amp, label = f'NR_{ymode}', color = 'black', alpha = 0.6)
                    ax1_ln3 = ax1.plot(tHi, hHi.amp, label = 'ampNoNQC')
                    ax2 = ax1.twinx()
                    ax2_ln1 = ax2.plot(dyHi.time, dyHi.r, label = r'$r$', color = 'red')
                    ax1.set_xlim([tHi[0]*0.999, tHi[-1]*1.001])
                    ax12_lns = ax1_ln1 + ax1_ln2 + ax1_ln3 + ax2_ln1
                    ax12_labs = [l.get_label() for l in ax12_lns]
                    ax1.legend(ax12_lns, ax12_labs)
                    ax2.grid()

                    ax3 = fig.add_subplot(212)
                    ax3_ln1 = ax3.plot((wf_1.time+tmove)*dimt, wf_1.amp, label = f'EOB_{ymode}', linestyle = '--', alpha = 0.7)
                    ax3_ln2 = ax3.plot((wf_2.time+tmove)*dimt, wf_2.amp, label = f'NR_{ymode}', color = 'black', alpha = 0.6)
                    ax3_ln3 = ax3.plot(tHi, hHi.amp, label = 'ampNoNQC')
                    ax4 = ax3.twinx()
                    ax4_ln1 = ax4.plot(dyHi.time, dyHi.xx, label = r'$x$', color = 'red')
                    ax3.set_xlim([tHi[0]*0.999, tHi[-1]*1.001])
                    ax34_lns = ax3_ln1 + ax3_ln2 + ax3_ln3 + ax4_ln1
                    ax34_labs = [l.get_label() for l in ax34_lns]
                    ax3.legend(ax34_lns, ax34_labs)
                    ax4.grid()

                    plt.savefig(prefix / 'dyNQCHigh.png', dpi = 200)
                    plt.close()
                    os.system(f'rm {fHigh}')
                    os.system(f'rm {fHiDy}')
                    os.system(f'rm {fRD}')
                    os.system(f'rm {fHighN}')
            # plot end

            Mtotal_list = np.linspace(10, 200, 500)
            # Setting saveing prefix
            fresults = prefixM / f'results_{SXSnum}.csv'
            # Setting Results savimg filename.
            save_namecol(fresults, data = [['#q', '#chi1', '#chi2', '#Mtotal', '#FF', f'#ecc={ecc}']])
            ret = ge.get_overlap(jobtag = args.jobtag, minecc = 0, maxecc = 0, eccentricity = ecc,
                                timeout = args.timeout, verbose = True, Mtotal = Mtotal_list, 
                                KK = KK_default, dSO = dSO_default, dSS = dSS_default, 
                                dtPeak = dtpeak_fit, ecc = ecc, mode = ymode)
            length = len(Mtotal_list)
            q_list = NR.q*np.ones(len(Mtotal_list)).reshape(1,length)
            s1z_list = NR.s1z*np.ones(len(Mtotal_list)).reshape(1,length)
            s2z_list = NR.s2z*np.ones(len(Mtotal_list)).reshape(1,length)
            FF_list = ret[2].reshape(1,length)
            Mtotal_list_out = Mtotal_list.reshape(1, length)
            data = np.concatenate((q_list, s1z_list, s2z_list, Mtotal_list_out, FF_list), axis = 0)
            add_csv(fresults, data.T.tolist())
            add_csv(fname_collect, [[SXSnum, NR.q, NR.s1x, NR.s1y, NR.s1z, NR.s2x, NR.s2y, NR.s2z, NR.chiX, ecc, fini_use, FF, lnp]])
            with open(prefixSXS / 'mdebug.txt', 'a') as f:
                f.write(mDebug)
    return 0

def Fitting_EOB_vs_NR_HM(argv = None):
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL
    from .SXS import save_namecol, add_csv, ModeC_alignment
    from .generator import self_adaptivor

    parser = OptionParser(description='Waveform Comparation With SXS')

    parser.add_option('--executable', type = 'str', default = DEFAULT_EXEV5, help = 'Exe command')
    parser.add_option('--approx', type = 'str', default = 'SEOBNREv5', help = 'Version of the code')
    parser.add_option('--fini', type = 'float', default = 0, help = 'Initial orbital frequency')
    parser.add_option('--SXS', type = 'str', action = 'append', default = [], help = 'SXS template for comparision')
    parser.add_option('--mtotal-base', type = 'float', default = 1, help = 'Total mass')
    parser.add_option('--srate', type = 'float', default = 16384, help = 'Sample rate')
 
    parser.add_option('--prefix', type = 'str', default = '.', help = 'dir for results saving.')
    parser.add_option('--jobtag', type = 'str', default = '_lnprob', help = 'jobtag.')

    parser.add_option('--psd', type = 'str', help = 'Detector psd.')
    parser.add_option('--flow', type = 'float', default = 0, help = 'Lower frequency cut off for psd.')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')

    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')

    parser.add_option('--num-ecc', type = 'int', default = 50, help = 'numbers for grid search')
    parser.add_option('--max-ecc', type = 'float', help = 'Upper bound of parameter')
    parser.add_option('--min-ecc', type = 'float', help = 'Lower bound of parameter')
    parser.add_option('--modeest-forecc', type = 'int', default = 22, help = 'mode for ecc estimation')

    parser.add_option('--num-fini', type = 'int', default = 20, help = 'numbers for grid search')
    parser.add_option('--max-finifac', type = 'float', default = 1.05, help = 'Upper bound of parameter')
    parser.add_option('--min-finifac', type = 'float', default = 0.95, help = 'Lower bound of parameter')
    parser.add_option('--modeest-forfini', type = 'int', default = 22, help = 'mode for fini estimation')

    parser.add_option('--eps', type = 'float', default = 1e-6, help = 'Thresh of div')
    parser.add_option('--mag', type = 'float', default = 10, help = 'Thresh of dx_init / dx (>1)')
    parser.add_option('--filter-thresh', type = 'float', default = 0.4, help = 'Thresh of grid search (<1)')
    parser.add_option('--max-step', type = 'int', default = 100, help = 'Max iter depth')
    parser.add_option('--version', type = 'str', default = 'default', help = 'code version')
    parser.add_option('--cutpct', type = 'float', default = 1.5, help = 'cut the NR waveform')
    parser.add_option('--plot', action = 'store_true', help = 'code version')
    parser.add_option('--plot-thresh', type = 'float', default = 1, help = 'plot when FF < ..')
    args, _ = parser.parse_args(argv)

    exe = args.executable
    approx = args.approx
    SXSnum_list = args.SXS
    if len(SXSnum_list) == 0:
        SXSnum_list.append('0001')
    fini = args.fini
    srate = args.srate
    table = args.table
    srcloc_all = args.srcloc_all
    psd = DetectorPSD(args.psd, flow = args.flow)
    eps = args.eps
    mag = args.mag
    filter_thresh = args.filter_thresh
    max_step = args.max_step
    cutpct = args.cutpct
    jobtag = args.jobtag

    prefix_all = Path(args.prefix)
    mode_list = ((2,2), (2,1), (3,3), (4,4))
    if not prefix_all.exists():
        prefix_all.mkdir(parents=True)
    collect_title = [['#SXSid', '#q', '#chi1x', '#chi1y', '#chi1z', '#chi2x', '#chi2y', '#chi2z', '#ecc', '#fini', '#minFF', '#maxFF', '#minlnp', '#maxlnp']]
    for (modeL, modeM) in mode_list:
        ymodeTag = int(10*modeL + modeM)
        fname_collect = prefix_all / f'collect_{ymodeTag}'
        save_namecol(fname_collect, collect_title)
        prefixM = prefix_all / f'MtotalFF_{ymodeTag}'
        if not prefixM.exists():
            prefixM.mkdir(parents = True)

    for ind, SXSnum in enumerate(SXSnum_list):
        debugMessage = f'SXS:BBH:{SXSnum}\n'
        prefix_SXS = prefix_all / SXSnum
        NR = SXSAllMode(SXSnum, table = table, srcloc = srcloc_all, cutpct = cutpct)
        h22 = NR.get_mode(2,2)
        h21 = NR.get_mode(2,1)
        h33 = NR.get_mode(3,3)
        h44 = NR.get_mode(4,4)
        # step 1 Estimate Ecc & fini
        max_freq = max(h22.frequency.max(), h21.frequency.max(), h33.frequency.max(), h44.frequency.max())
        deltaT = np.pi / max_freq
        debugMessage += f'max_freq :\t{max_freq}'
        NRModetime = np.arange(h22.time[0], h22.time[-1], deltaT)
        NRModes = NR.mode_resample(NRModetime)
        debugMessage += f'NRModeTime :\t{NRModetime}'
        m1 = NR.mQ1 * args.mtotal_base
        m2 = NR.mQ2 * args.mtotal_base
        s1x = NR.s1x
        s1y = NR.s1y
        s1z = NR.s1z
        s2x = NR.s2x
        s2y = NR.s2y
        s2z = NR.s2z
        debugMessage += f'm1 :\t{m1}\nm2 :\t{m2}\nchi1 :\t({s1x},{s1y},{s1z})\nchi2 :\t({s2x},{s2y},{s2z})\n'
        srate = dim_t(m1 + m2) / deltaT
        debugMessage += f'fs :\t{srate}\n'
        ge = Generator(approx = approx, executable = exe, verbose = args.verbose)
        IS_CIRC = False
        IS_PREC = NR.is_prec
        f0, min_e, max_e = get_ecc_range(SXSnum, args.min_ecc, args.max_ecc)
        if f0 is None:
            f0 = NR.Sf_ini
            IS_CIRC = True
        debugMessage += f'IS_CIRC :\t{IS_CIRC}\nIS_PREC :\t{IS_PREC}\n'
        fini = f0 * dim_t(m1 + m2)
        debugMessage += f'preFreq[M] :\t{f0}\npreFreq[Hz] :\t{fini}\n'
        max_ecc = max_e
        min_ecc = min_e
        ecc_range = (min_ecc, max_ecc)
        num_ecc = args.num_ecc
        num_fini = args.num_fini
        mode_ecc_est = args.modeest_forecc
        modeL_ecc_est = int(mode_ecc_est/10)
        modeM_ecc_est = mode_ecc_est%10
        max_finifac = args.max_finifac
        min_finifac = args.min_finifac
        mode_fini_est = args.modeest_forfini
        modeL_fini_est = int(mode_fini_est/10)
        modeM_fini_est = mode_fini_est%10

        # Functions to Estimate initial eccentricity and frequency
        def estimate_ecc(ecc, Mtotal = None):
            ret = ge(m1 = m1, m2 = m2, s1x = s1x, s1y = s1y, s1z = s1z, 
                    s2x = s2x, s2y = s2y, s2z = s2z, D = 100, 
                    ecc = ecc, srate = srate, f_ini = fini, L = 2, M = 2,
                    timeout = 3600, jobtag = jobtag, mode = mode_ecc_est)
            if isinstance(ret, CEV):
                return -np.inf
            t, hr, hi = ret[:,0], ret[:,1], ret[:,2]
            hEOB = ModeBase(t, hr, hi)
            hNR = NRModes.get_mode(modeL_ecc_est, modeM_ecc_est)
            MtotalList_ecc = (20, 40, 70, 100, 130, 160, 190)
            if Mtotal is not None:
                MtotalList_ecc = Mtotal
            FFL, _, tcL = calculate_ModeFF(hEOB, hNR, Mtotal = MtotalList_ecc, psd = psd)
            lnp = -np.power((1-FFL)/0.01, 2) - np.power(tcL/5, 2)
            if args.verbose:
                sys.stderr.write(f'ecc = {ecc}: lnp = {lnp}\n')
            best = np.min(lnp)
            if np.isnan(best):
                return -np.inf
            return best
        
        def estimate_fini(fini_input, Mtotal = None):
            ret = ge(m1 = m1, m2 = m2, s1x = s1x, s1y = s1y, s1z = s1z, 
                    s2x = s2x, s2y = s2y, s2z = s2z, D = 100, 
                    ecc = 0.0, srate = srate, f_ini = fini_input, L = 2, M = 2,
                    timeout = 3600, jobtag = jobtag, mode = mode_fini_est)
            if isinstance(ret, CEV):
                return -np.inf
            t, hr, hi = ret[:,0], ret[:,1], ret[:,2]
            hEOB = ModeBase(t, hr, hi)
            hNR = NRModes.get_mode(modeL_fini_est, modeM_fini_est)
            MtotalList_ini = (20, 40, 70, 100, 130, 160, 190)
            if Mtotal is not None:
                MtotalList_ini = Mtotal
            FFL, _, tcL = calculate_ModeFF(hEOB, hNR, Mtotal = MtotalList_ini, psd = psd)
            lnp = -np.power((1-FFL)/0.01, 2) - np.power(tcL/5, 2)
            if args.verbose:
                sys.stderr.write(f'fini = {fini_input}: lnp = {lnp}\n')
            best = np.min(lnp)
            if np.isnan(best):
                return -np.inf
            return best
        
        def estimate_ecc_fini(ecc, fini_input, Mtotal = None):
            ret = ge(m1 = m1, m2 = m2, s1x = s1x, s1y = s1y, s1z = s1z, 
                    s2x = s2x, s2y = s2y, s2z = s2z, D = 100, 
                    ecc = ecc, srate = srate, f_ini = fini_input, L = 2, M = 2,
                    timeout = 3600, jobtag = jobtag, mode = 22)
            if isinstance(ret, CEV):
                return -np.inf
            t, hr, hi = ret[:,0], ret[:,1], ret[:,2]
            hEOB = ModeBase(t, hr, hi)
            hNR = NRModes.get_mode(2, 2)
            MtotalList_ini = (20, 40, 70, 100, 130, 160, 190)
            if Mtotal is not None:
                MtotalList_ini = Mtotal
            FFL, _, tcL = calculate_ModeFF(hEOB, hNR, Mtotal = MtotalList_ini, psd = psd)
            lnp = -np.power((1-FFL)/0.01, 2) - np.power(tcL/5, 2)
            if args.verbose:
                sys.stderr.write(f'fini = {fini_input}: lnp = {lnp}\n')
            best = np.min(lnp)
            if np.isnan(best):
                return -np.inf
            return best

        fini_use = fini
        ecc_use = 0
        if IS_CIRC and IS_PREC:
            # precession circular orbits
            fini_range = (fini*min_finifac, fini*max_finifac)
            debugMessage += f'finiSearchRange :\t{fini_range}, {num_fini}\n'
            MG = MultiGrid1D(estimate_fini, fini_range, num_fini)
            data = MG.run(None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
            fini_grid, lnp_grid = data[:,0], data[:,1]
            fini_use = fini_grid[np.argmax(lnp_grid)]
        elif not IS_CIRC and not IS_PREC:
            # spin-aligned elliptical orbits
            debugMessage += f'eccSearchRange :\t{ecc_range}, {num_ecc}\n'
            MG = MultiGrid1D(estimate_ecc, ecc_range, num_ecc)
            data = MG.run(None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
            ecc_grid, lnp_grid = data[:,0], data[:,1]
            ecc_use = ecc_grid[np.argmax(lnp_grid)]
        elif not IS_CIRC and IS_PREC:
            # precession elliptical orbits
            fini_range = (fini*min_finifac, fini*max_finifac)
            debugMessage += f'eccSearchRange :\t{ecc_range}, {num_ecc}\n'
            debugMessage += f'finiSearchRange :\t{fini_range}, {num_fini}\n'
            MG = MultiGrid(estimate_ecc_fini, ecc_range, fini_range, num_ecc, num_fini)
            data = MG.run(None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
            ecc_grid, fini_grid, lnp_grid = data[:,0], data[:,1], data[:,2]
            ecc_use = ecc_grid[np.argmax(lnp_grid)]
            fini_use = fini_grid[np.argmax(lnp_grid)]
        CMD = ge.get_CMD(m1 = m1, m2 = m2, s1x = s1x, s1y = s1y, s1z = s1z, 
                    s2x = s2x, s2y = s2y, s2z = s2z, D = 100, 
                ecc = ecc_use, srate = srate, f_ini = fini_use, L = 2, M = 2,
                timeout = 3600, jobtag = jobtag, mode = -1)
        debugMessage += f'CM :\n\t{CMD}\n'
        ret_f = ge(m1 = m1, m2 = m2, s1x = s1x, s1y = s1y, s1z = s1z, 
                    s2x = s2x, s2y = s2y, s2z = s2z, D = 100, 
                ecc = ecc_use, srate = srate, f_ini = fini_use, L = 2, M = 2,
                timeout = 3600, jobtag = jobtag, mode = -1)
        if isinstance(ret_f, CEV):
            continue
        t, h22r, h22i, h21r, h21i, h33r, h33i, h44r, h44i = \
            ret_f[:,0], ret_f[:,1], ret_f[:,2], ret_f[:,3], ret_f[:,4], ret_f[:,5], ret_f[:,6], ret_f[:,7], ret_f[:,8]
        EOBModes = waveform_mode_collector(0)
        EOBModes.append_mode(t, h22r, h22i, 2, 2)
        EOBModes.append_mode(t, h22r, -h22i, 2, -2)
        EOBModes.append_mode(t, h21r, h21i, 2, 1)
        EOBModes.append_mode(t, h21r, -h21i, 2, -1)
        EOBModes.append_mode(t, h33r, h33i, 3, 3)
        EOBModes.append_mode(t, -h33r, h33i, 3, -3)
        EOBModes.append_mode(t, h44r, h44i, 4, 4)
        EOBModes.append_mode(t, h44r, -h44i, 4, -4)
        EOBModes_C, NRModes_C = ModeC_alignment(EOBModes, NRModes)
        MtotalList = np.arange(20, 201, 1)
        for (modeL, modeM) in mode_list:
            if modeM % 2 and NR.zero_odd_mode:
                continue
            ymodeTag = int(10*modeL + modeM)
            debugMessage += f'Mode {ymodeTag}\n'
            hEOB = EOBModes_C.get_mode(modeL, modeM)
            hNR = NRModes_C.get_mode(modeL, modeM)
            FFlm, dflm, tclm = calculate_ModeFF(hEOB, hNR, Mtotal = MtotalList, psd = psd)
            lnp = -np.power((1-FFlm)/0.01, 2) - np.power(tclm/5, 2)
            maxFF = np.max(FFlm)
            minFF = np.min(FFlm)
            maxlnp = np.max(lnp)
            minlnp = np.min(lnp)
            tc_best = tclm[np.argmax(FFlm)]
            phic_best = dflm[np.argmax(FFlm)]
            debugMessage += f'\tFF :\t({minFF}, {maxFF})\nlnp :\t({minlnp}, {maxlnp})\n'
            length = len(MtotalList)
            data = np.concatenate((MtotalList.reshape(1, length), FFlm.reshape(1, length), lnp.reshape(1, length)), axis = 0)
            fresults = prefix_all / f'MtotalFF_{ymodeTag}' / f'results_{SXSnum}.csv'
            fname_collect = prefix_all / f'collect_{ymodeTag}.csv'
            add_csv(fresults, data.T.tolist())
            add_csv(fname_collect, [[SXSnum, NR.q, NR.s1x, NR.s1y, NR.s1z, NR.s2x, NR.s2y, NR.s2z, ecc_use, fini_use, minFF, maxFF, minlnp, maxlnp]])
            if args.plot and minFF < args.plot_thresh:
                fig = plt.figure(figsize = (14, 7))
                ax1 = fig.add_subplot(211)
                ax1.set_title(f'lnp={maxlnp},FF={maxFF}')
                ax1.plot(hEOB.time, hEOB.amp, label = f'EOB_{ymodeTag}')
                ax1.plot(hNR.time, hNR.amp, label = f'NR_{ymodeTag}')
                ax1.legend()
                ax1.grid()

                ax3 = fig.add_subplot(212)
                ax3.plot(hEOB.time, hEOB.phaseFrom0, label = f'EOB_{ymodeTag}')
                ax3.plot(hNR.time, hNR.phaseFrom0, label = f'NR_{ymodeTag}')
                ax3.legend()
                ax3.grid()
                plt.savefig(prefix_SXS / f'AmpPhase_{ymodeTag}.png', dpi = 200)
                plt.close()

                ipeak = hEOB.argpeak
                h1 = hEOB.amp * np.exp(-1.j*(hEOB.phaseFrom0 - hEOB.phaseFrom0[ipeak]))
                h2 = hNR.amp * np.exp(-1.j*(hNR.phaseFrom0 - hNR.phaseFrom0[ipeak]))
                plt.figure(figsize = (14, 7))
                plt.subplot(211)
                plt.title(f'eta={NR.eta}, chi={NR.chiX}')
                plt.plot(hEOB.time, h1.real, label = f'EOB_{ymodeTag}')
                plt.plot(hNR.time, h2.real, label = f'NR_{ymodeTag}')
                plt.legend()
                plt.grid()
                plt.subplot(212)
                plt.title(f'chi1={NR.s1z}, chi2={NR.s2z}')
                plt.plot(hEOB.time, h1.imag, label = f'EOB_{ymodeTag}')
                plt.plot(hNR.time, h2.imag, label = f'NR_{ymodeTag}')
                plt.legend()
                plt.grid()
                plt.savefig(prefix_SXS / f'RealImag_{ymodeTag}.png', dpi = 200)
                plt.close()
        with open(prefix_SXS / 'mdebug.txt', 'a') as f:
            f.write(debugMessage)
    return 0




import scipy.optimize as op
def Compare_ecc_HM(argv = None):
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL
    from .SXS import save_namecol, add_csv, ModeC_alignment
    from .generator import self_adaptivor

    parser = OptionParser(description='Waveform Comparation With SXS')

    parser.add_option('--executable', type = 'str', default = DEFAULT_EXEV5, help = 'Exe command')
    parser.add_option('--approx', type = 'str', default = 'SEOBNREv5', help = 'Version of the code')
    parser.add_option('--SXS', type = 'str', action = 'append', default = [], help = 'SXS template for comparision')
 
    parser.add_option('--prefix', type = 'str', default = '.', help = 'dir for results saving.')
    parser.add_option('--jobtag', type = 'str', default = '_lnprob', help = 'jobtag.')

    parser.add_option('--psd', type = 'str', default = 'advLIGO_zerodethp', help = 'Detector psd.')
    parser.add_option('--flow', type = 'float', default = 0, help = 'Lower frequency cut off for psd.')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')

    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')

    parser.add_option('--num-ecc', type = 'int', default = 50, help = 'numbers for grid search')
    parser.add_option('--max-ecc', type = 'float', help = 'Upper bound of parameter')
    parser.add_option('--min-ecc', type = 'float', help = 'Lower bound of parameter')
    parser.add_option('--mtotal-base', type = 'float', default = 1, help = 'base mtotal for wf generation')
    parser.add_option('--num-mtotal', type = 'int', default = 10, help = 'numbers for grid search')
    parser.add_option('--max-mtotal', type = 'float', default = 200, help = 'Upper bound of parameter')
    parser.add_option('--min-mtotal', type = 'float', default = 20, help = 'Lower bound of parameter')
    parser.add_option('--iota', type = 'float', help = 'inclination 0 [pi]')
    parser.add_option('--delta-ci', type = 'float', default = 0.05, help = 'cos iota step [0.1]')
    parser.add_option('--delta-phix', type = 'float', default = 0.125, help = 'cos iota step [0.1]')
    parser.add_option('--phix', type = 'float', help = 'phix 0 [pi]')
    parser.add_option('--kappa', type = 'float', help = 'kappa 0 [pi]')
    parser.add_option('--ecc', type = 'float', help = 'estimated ecc')
    parser.add_option('--mtotal', type = 'float', help = 'this mtotal')
    parser.add_option('--circ', action = 'store_true', help = 'force ecc = 0')
    parser.add_option('--save-all', action = 'store_true', help = 'dump ciota')
    parser.add_option('--usemp', action = 'store_true', help = 'use multi process')
    parser.add_option('--verbose', action = 'store_true', help = 'verbose output')
    parser.add_option('--num-mc', type = 'int', default = 0, help = 'if > 0 will use Monte-Carlo sample spherical integration for phix-iota')
    parser.add_option('--np', type = 'int', help = 'numbers of multi processes')

    parser.add_option('--eps', type = 'float', default = 1e-6, help = 'Thresh of div')
    parser.add_option('--mag', type = 'float', default = 10, help = 'Thresh of dx_init / dx (>1)')
    parser.add_option('--filter-thresh', type = 'float', default = 0.4, help = 'Thresh of grid search (<1)')
    parser.add_option('--max-step', type = 'int', default = 100, help = 'Max iter depth')
    parser.add_option('--only22', action = 'store_true', help = 'only22 for EOB')
    parser.add_option('--search-ecc', action = 'store_true', help = 'for test')
    parser.add_option('--search-ecc-mtotal', action = 'store_true', help = 'for test')
    args, _ = parser.parse_args(argv)

    exe = args.executable
    approx = args.approx
    SXSnum_list = args.SXS
    if len(SXSnum_list) == 0:
        SXSnum_list.append('0001')
    table = args.table
    jobtag = args.jobtag

    srcloc_all = args.srcloc_all
    psd = DetectorPSD(args.psd, flow = args.flow)
    eps = args.eps
    mag = args.mag
    filter_thresh = args.filter_thresh
    max_step = args.max_step
    prefix = Path(args.prefix)
    if not prefix.exists():
        prefix.mkdir(parents = True)

    if args.kappa is not None:
        kappaList = np.array([args.kappa * np.pi])
    else:
        kappaList = np.linspace(0, 2*np.pi, 10)

    if args.phix is not None:
        phiXList = np.array([args.phix * np.pi])
    else:
        phiXList = np.arange(0, 2, args.delta_phix) * np.pi
        np.savetxt(prefix / 'phiXList.dat', phiXList)
    if args.iota is not None:
        iotaList = np.array([args.iota * np.pi])
        cosiList = np.cos(iotaList)
    else:
        # iotaList = np.linspace(0, np.pi, 15)
        # iotaList = np.concatenate([np.linspace(0, 7, 15)[::-1], np.linspace(8, 28, 21)])*np.pi / 28
        cosiList = np.arange(0, 1, args.delta_ci)
        iotaList = np.arccos(cosiList)
        # iotaList = np.arccos(np.roll(cosiList, int(len(cosiList)/2)))
        np.savetxt(prefix / 'iotalist.dat', iotaList)
        np.savetxt(prefix / 'cosiotalist.dat', cosiList)

    for SXSnum in SXSnum_list:
        NR = SXSAllMode(SXSnum, table = table, srcloc = srcloc_all, cutpct = 1.5)
        h22 = NR.get_mode(2,2)
        h21 = NR.get_mode(2,1)
        h33 = NR.get_mode(3,3)
        h44 = NR.get_mode(4,4)

        max_freq = max(h22.frequency.max(), h21.frequency.max(), h33.frequency.max(), h44.frequency.max())
        deltaT = np.pi / max_freq
        NRModetime = np.arange(h22.time[0], h22.time[-1], deltaT)
        NRModes = NR.mode_resample(NRModetime)

        m1 = NR.mQ1 * args.mtotal_base
        m2 = NR.mQ2 * args.mtotal_base
        s1z = NR.s1z
        s2z = NR.s2z
        sys.stderr.write(f'q = {m1/m2}, chiA = {(s1z-s2z)/2}\n')
        srate = dim_t(m1 + m2) / deltaT
        ge = Generator(approx = approx, executable = exe, verbose = args.verbose)
        CIRC = args.circ
        f0, min_e, max_e = get_ecc_range(SXSnum, args.min_ecc, args.max_ecc)
        if f0 is None:
            f0 = NR.Sf_ini
            # f0 = 0.002
            CIRC = True
        fini = f0 * dim_t(m1 + m2)
        max_ecc = max_e
        min_ecc = min_e
        ecc_range = (min_ecc, max_ecc)
        num_ecc = args.num_ecc

        def estimate_ecc(ecc, Mtotal = None):
            ret = ge(m1 = m1, m2 = m2, s1z = s1z, s2z = s2z, D = 100, 
                    ecc = ecc, srate = srate, f_ini = fini, L = 2, M = 2,
                    timeout = 3600, jobtag = jobtag, mode = 22)
            if isinstance(ret, CEV):
                return -np.inf
            t, h22r, h22i = ret[:,0], ret[:,1], ret[:,2]
            h22EOB = ModeBase(t, h22r, h22i)
            h22NR = NRModes.get_mode(2, 2)
            MtotalList_ecc = (20, 40, 70, 100, 130, 160, 190)
            if Mtotal is not None:
                MtotalList_ecc = Mtotal
            FFL, _, tcL = calculate_ModeFF(h22EOB, h22NR, Mtotal = MtotalList_ecc, psd = psd)
            lnp = -np.power((1-FFL)/0.01, 2) - np.power(tcL/5, 2)
            if args.verbose:
                sys.stderr.write(f'ecc = {ecc}: lnp = {lnp}\n')
            best = np.min(lnp)
            if np.isnan(best):
                return -np.inf
            return best
        
        def estimate_fini(fini_input, Mtotal = None):
            ret = ge(m1 = m1, m2 = m2, s1z = s1z, s2z = s2z, D = 100, 
                    ecc = 0.0, srate = srate, f_ini = fini_input, L = 2, M = 2,
                    timeout = 3600, jobtag = jobtag, mode = 22)
            if isinstance(ret, CEV):
                return -np.inf
            t, h22r, h22i = ret[:,0], ret[:,1], ret[:,2]
            h22EOB = ModeBase(t, h22r, h22i)
            h22NR = NRModes.get_mode(2, 2)
            MtotalList_ini = (20, 40, 70, 100, 130, 160, 190)
            if Mtotal is not None:
                MtotalList_ini = Mtotal
            FFL, _, tcL = calculate_ModeFF(h22EOB, h22NR, Mtotal = MtotalList_ini, psd = psd)
            lnp = -np.power((1-FFL)/0.01, 2) - np.power(tcL/5, 2)
            if args.verbose:
                sys.stderr.write(f'fini = {fini_input}: lnp = {lnp}\n')
            best = np.min(lnp)
            if np.isnan(best):
                return -np.inf
            return best

        if CIRC:
            ecc_fit = 0.0
            if NR.is_prec:
                fini_range = (fini*0.75, fini*1.25)
                MG = MultiGrid1D(estimate_fini, fini_range, 20)
                data = MG.run(None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
                fini_grid, lnp_grid = data[:,0], data[:,1]
                fini = fini_grid[np.argmax(lnp_grid)]
        elif args.ecc is None:
            MG = MultiGrid1D(estimate_ecc, ecc_range, num_ecc)
            data = MG.run(fsave = None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
            ecc_grid, lnp_grid = data[:,0], data[:,1]
            ecc_fit = ecc_grid[np.argmax(lnp_grid)]
            lnp = estimate_ecc(ecc_fit)
            print(f'best lnp = {lnp}')
        else:
            ecc_fit = args.ecc
        
        sys.stderr.write(f'{LOG}: SXS_{SXSnum}, Estimate ecc_fit = {ecc_fit}\n')
        max_mtotal = args.max_mtotal
        min_mtotal = args.min_mtotal
        num_mtotal = args.num_mtotal
        if args.mtotal is None:
            MtotalList = np.linspace(min_mtotal, max_mtotal, num_mtotal)
        else:
            MtotalList = np.array([args.mtotal])

        NRModeList = []
        if 1:
            for l in range(2, 5):
                for m in range(-l, l+1):
                    if m!=0:
                        NRModeList.append((l,m))
        else:
            # NRModeList = [(2,2), (2,-2), (2,1), (2,-1), (3,3), (3,-3), (4,4), (4,-4)]
            NRModeList = [(2,2),(2,-2)]
        if args.only22:
            jtag = 'only22'
            EOBModeList = [(2,2), (2,-2)]
            # EOBModeList = [(3,3),(3,-3)]
        else:
            jtag = 'HM'
            EOBModeList = [(2,2), (2,-2), (2,1), (2,-1), (3,3), (3,-3), (4,4), (4,-4)]
        sys.stderr.write(f'NRModeList:\n{NRModeList}\n')
        sys.stderr.write(f'EOBModeList:\n{EOBModeList}\n')
        def calculate_Max_FF_HM(ecc, Mtotal_input, iota_input, phic_input=None):
            ret = ge(m1 = m1, m2 = m2, s1z = s1z, s2z = s2z, D = 100, 
                    ecc = ecc, srate = srate, f_ini = fini, L = 2, M = 2,
                    timeout = 3600, jobtag = jobtag, mode = 0)
            if isinstance(ret, CEV):
                return 0
            EOBModes = waveform_mode_collector(0)
            t, h22r, h22i, h21r, h21i, h33r, h33i, h44r, h44i = \
                ret[:,0], ret[:,1], ret[:,2], ret[:,3], ret[:,4], ret[:,5], ret[:,6], ret[:,7], ret[:,8]   
            EOBModes.append_mode(t, h22r, h22i, 2, 2)
            EOBModes.append_mode(t, h22r, -h22i, 2, -2)
            EOBModes.append_mode(t, h21r, h21i, 2, 1)
            EOBModes.append_mode(t, h21r, -h21i, 2, -1)
            EOBModes.append_mode(t, h33r, h33i, 3, 3)
            EOBModes.append_mode(t, h33r, -h33i, 3, -3)
            EOBModes.append_mode(t, h44r, h44i, 4, 4)
            EOBModes.append_mode(t, h44r, -h44i, 4, -4)
            hpcNR = NRModes.construct_hpc(iota_input, 0, modelist = NRModeList, phaseFrom0 = True)
            def max_FF_over_phic(phic):
                hpcEOB = EOBModes.construct_hpc(iota_input, phic, modelist = EOBModeList, phaseFrom0 = True)
                FF, _1, _2 = calculate_ModeFF(hpcEOB, hpcNR.copy(), Mtotal = Mtotal_input, psd = psd)
                if args.verbose:
                    sys.stderr.write(f'{phic/np.pi} pi: {FF}\n')
                return FF
            if 1:
                nll = lambda x : max_FF_over_phic(x[0])
                result = op.minimize(nll, (0), args = ())
                phic = result["x"][0]
                FF = result["y"][0]
                return FF, phic
            elif phic_input is None:
                dphic_range = (-np.pi*0.1, 2.1*np.pi)
                MG_phic = MultiGrid1D(max_FF_over_phic, dphic_range, 60)
                data = MG_phic.run(fsave = None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
                return np.max(data[:,1])
            elif hasattr(phic_input, '__len__'):
                if type(phic_input) is str:
                    dphic_range = (-np.pi*0.1, 2.1*np.pi)
                    num_dp = 60
                else:
                    dphic_range = phic_input
                    num_dp = 10
                MG_phic = MultiGrid1D(max_FF_over_phic, dphic_range, num_dp)
                data = MG_phic.run(fsave = None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
                imax = np.argmax(data[:,1])
                phic_max = data[imax,0]
                FF_max = data[imax,1]
            else:
                FF_max = max_FF_over_phic(phic_input)
                phic_max = phic_input
            return FF_max, phic_max

        def calculate_Max_FF_HM_Circ(Mtotal_input, iota_input, phic_input = None, XX = 2):
            ret_C = ge(m1 = m1, m2 = m2, s1z = s1z, s2z = s2z, D = 100, 
                    ecc = 0.0, srate = srate, f_ini = fini, L = 2, M = 2,
                    timeout = 3600, jobtag = jobtag, mode = 0)
            if isinstance(ret_C, CEV):
                return 0
            EOBModes_C = waveform_mode_collector(0)
            t, h22r, h22i, h21r, h21i, h33r, h33i, h44r, h44i = \
                ret_C[:,0], ret_C[:,1], ret_C[:,2], ret_C[:,3], ret_C[:,4], ret_C[:,5], ret_C[:,6], ret_C[:,7], ret_C[:,8]   
            EOBModes_C.append_mode(t, h22r, h22i, 2, 2)
            EOBModes_C.append_mode(t, h22r, -h22i, 2, -2)
            EOBModes_C.append_mode(t, h21r, h21i, 2, 1)
            EOBModes_C.append_mode(t, h21r, -h21i, 2, -1)
            EOBModes_C.append_mode(t, h33r, h33i, 3, 3)
            EOBModes_C.append_mode(t, h33r, -h33i, 3, -3)
            EOBModes_C.append_mode(t, h44r, h44i, 4, 4)
            EOBModes_C.append_mode(t, h44r, -h44i, 4, -4)
            hpcNR = NRModes.construct_hpc(iota_input, 0, modelist = NRModeList, phaseFrom0 = True)
            def max_FF_over_phic(phic):
                hpcEOB = EOBModes_C.construct_hpc(iota_input, phic, modelist = EOBModeList, phaseFrom0 = True)
                FF, _1, _2 = calculate_ModeFF(hpcEOB, hpcNR.copy(), Mtotal = Mtotal_input, psd = psd)
                if args.verbose:
                    sys.stderr.write(f'{phic/np.pi} pi {FF}\n')
                return FF
            if phic_input is None:
                dphic_range = (-np.pi*0.1, 2.1*np.pi)
                MG_phic = MultiGrid1D(max_FF_over_phic, dphic_range, 45)
                data = MG_phic.run(fsave = None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
                imax = np.argmax(data[:,1])
                phic_max = data[imax,0]
                FF_max = data[imax,1]
            elif hasattr(phic_input, '__len__'):
                dphic_range = phic_input
                MG_phic = MultiGrid1D(max_FF_over_phic, dphic_range, 10)
                data = MG_phic.run(fsave = None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
                imax = np.argmax(data[:,1])
                phic_max = data[imax,0]
                FF_max = data[imax,1]
            else:
                FF_max = max_FF_over_phic(phic_input)
                phic_max = phic_input
            return FF_max, phic_max

        def calculate_Max_FF_HM_fit(EOBModes, NRModes, Mtotal_input, iota_input, phic_input = None, XX = 0, kappa = 0, phin = 0, mpQueue = None):        
            hpcNR = NRModes.construct_hpc(iota_input, phin, modelist = NRModeList, phaseFrom0 = False)
            hpcNR.apply_phic(kappa)
            def max_FF_over_phic(phic):
                hpcEOB = EOBModes.construct_hpc(iota_input, phic, modelist = EOBModeList, phaseFrom0 = False)
                FF, _1, _2 = calculate_ModeFF(hpcEOB, hpcNR, Mtotal = Mtotal_input, psd = psd)
                if 0:
                    sys.stderr.write(f'{phic/np.pi} pi {FF}\n')
                return FF
            if 0:
                nll = lambda x : max_FF_over_phic(x[0])
                result = op.minimize(nll, (0), args = ())
                phic = result["x"][0]
                return max_FF_over_phic(phic), phic
            elif phic_input is None:
                dphic_range = (np.pi*(XX-0.1), (2.1+XX)*np.pi)
                MG_phic = MultiGrid1D(max_FF_over_phic, dphic_range, 60)
                data = MG_phic.run(fsave = None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
                imax = np.argmax(data[:,1])
                phic_max = data[imax,0]
                FF_max = data[imax,1]
            elif hasattr(phic_input, '__len__'):
                dphic_range = phic_input
                MG_phic = MultiGrid1D(max_FF_over_phic, dphic_range, 20)
                data = MG_phic.run(fsave = None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
                imax = np.argmax(data[:,1])
                phic_max = data[imax,0]
                FF_max = data[imax,1]
            else:
                FF_max = max_FF_over_phic(phic_input)
                phic_max = phic_input
            if mpQueue is None:
                return FF_max, phic_max
            mpQueue.put(FF_max)
            return

        fresults = prefix / f'results_{SXSnum}_{jtag}.csv'
        ci_slist = []
        for ci in cosiList:
            ci_slist.append(f'#{ci}')
        # CheckPoint
        if fresults.exists():
            check_data = np.loadtxt(fresults, delimiter = ',')
            try:
                Mtotal_final = check_data[:,0][-1]
            except:
                sys.stderr.write(f'over write {fresults}\n')
            else:
                if Mtotal_final >= MtotalList[-1]:
                    continue
                else:
                    Msub = np.abs(MtotalList - Mtotal_final)
                    ind_check = np.argmin(Msub)
                    if MtotalList[ind_check] < Mtotal_final:
                        ind_check += 1
                    sys.stderr.write(f'Check point at {MtotalList[ind_check]}\n')
                    MtotalList = MtotalList[ind_check:]
        # Setting Results savimg filename.
        elif CIRC:
            if args.save_all:
                save_namecol(fresults, data = [['#Mtotal'] + ci_slist])
            else:
                save_namecol(fresults, data = [['#Mtotal', '#FF']])
        else:
            if args.save_all:
                save_namecol(fresults, data = [['#Mtotal', f'#e={ecc_fit}'] + ci_slist])
            else:
                save_namecol(fresults, data = [['#Mtotal', f'#e={ecc_fit}', '#FF']])
        if 1:
            ret1 = ge(m1 = m1, m2 = m2, s1z = s1z, s2z = s2z, D = 100, 
                    ecc = ecc_fit, srate = srate, f_ini = fini, L = 2, M = 2,
                    timeout = 3600, jobtag = jobtag, mode = -1)
            if isinstance(ret1, CEV):
                return 0
            t, h22r, h22i, h21r, h21i, h33r, h33i, h44r, h44i = \
                ret1[:,0], ret1[:,1], ret1[:,2], ret1[:,3], ret1[:,4], ret1[:,5], ret1[:,6], ret1[:,7], ret1[:,8]
            EOBModes = waveform_mode_collector(0)
            EOBModes.append_mode(t, h22r, h22i, 2, 2)
            EOBModes.append_mode(t, h22r, -h22i, 2, -2)
            EOBModes.append_mode(t, h21r, h21i, 2, 1)
            EOBModes.append_mode(t, h21r, -h21i, 2, -1)
            EOBModes.append_mode(t, h33r, h33i, 3, 3)
            EOBModes.append_mode(t, -h33r, h33i, 3, -3)
            EOBModes.append_mode(t, h44r, h44i, 4, 4)
            EOBModes.append_mode(t, h44r, -h44i, 4, -4)
            EOBModes_C, NRModes_C = ModeC_alignment(EOBModes, NRModes)

            # for l,m in [(2,2), (2,-2), (2,1), (2,-1), (3,3), (3,-3), (4,4), (4,-4)]:
            #     hlm = EOBModes.get_mode(l,m)
            #     nlm = NRModes.get_mode(l,m)
            #     FF, _1, _2, hlmC, nlmC = calculate_ModeFF(hlm, nlm, Mtotal = MtotalList[0], psd = psd, retall = True)
            #     print(f'(l,m) = ({l},{m}), FF = {FF}, Mtotal = {MtotalList[0]}, amp = {np.max(hlm.amp)}')
            # return 0
            if len(kappaList) == 1 and len(MtotalList) == 1:
                FFret = np.zeros([len(phiXList),len(iotaList)])
                kappa = kappaList[0]
                Mtotal = MtotalList[0]
                prefix2d = prefix / f'{SXSnum}_c2d_{jtag}'
                if not prefix2d.exists():
                    prefix2d.mkdir(parents = True)
                np.savetxt(prefix2d / 'phiXList.dat', phiXList)
                np.savetxt(prefix2d / 'cosiotaList.dat', cosiList)
                for i, phiX in enumerate(phiXList):
                    phic_fit_list = None
                    FFsave = []
                    for j, iota in enumerate(iotaList):
                        FF, phic_ret = calculate_Max_FF_HM_fit(EOBModes_C, NRModes_C, Mtotal_input = Mtotal, iota_input = iota, phic_input = phic_fit_list, kappa = kappa, phin = phiX)
                        FFret[i,j] = FF
                        FFsave.append(FF)
                        sys.stderr.write(f'Mtotal = {Mtotal}, iota = {iota/np.pi} pi, kappa = {kappa/np.pi} pi, phiX = {phiX/np.pi} pi, FF = {FF}\n')
                        if phic_fit_list is None:
                            phic_fit_list = (phic_ret - np.pi*1.1/5, phic_ret + np.pi*1.1/5)
                    add_csv(prefix2d / f'c2d.csv', [FFsave])
                save_namecol(prefix2d / 'info.csv', [['#Mtotal', '#kappa'], [Mtotal, kappa]])
            elif args.num_mc > 0 and not args.save_all:
                # Use Monte Carlo Integration
                if args.np is None:
                    for Mtotal in MtotalList:
                        FFlist = []
                        for i in range(args.num_mc):
                            vec = np.random.randn(3)
                            vecR = np.linalg.norm(vec)
                            Vcosi = min(vec[2] / vecR, 1)
                            Vcosi = max(Vcosi, -1)
                            Viota = np.arccos(Vcosi)
                            Vphi = np.arctan2(vec[1], vec[0])
                            kappa = 0
                            FF, _ = calculate_Max_FF_HM_fit(EOBModes_C, NRModes_C, Mtotal_input = Mtotal, iota_input = Viota, phic_input = None, kappa = kappa, phin = Vphi)
                            sys.stderr.write(f'Mtotal = {Mtotal}, iota = {Viota/np.pi} pi, phiX = {Vphi/np.pi} pi, FF = {FF}\n')
                            FFlist.append(FF)
                        FFlist = np.asarray(FFlist)
                        avg = np.average(FFlist)
                        add_csv(fresults, [[Mtotal, avg]])
                else:
                    # Use Multi-processing
                    mp_np = args.np
                    num_mc_mp = int(args.num_mc / mp_np)
                    for Mtotal in MtotalList:
                        FFlist = []
                        for i in range(num_mc_mp):
                            jobs = []
                            mp_q = mp.Queue()
                            for j in range(mp_np):
                                vec = np.random.randn(3)
                                vecR = np.linalg.norm(vec)
                                Vcosi = min(vec[2] / vecR, 1)
                                Vcosi = max(Vcosi, -1)
                                Viota = np.arccos(Vcosi)
                                Vphi = np.arctan2(vec[1], vec[0])
                                proc = mp.Process(target=calculate_Max_FF_HM_fit, 
                                    args=(EOBModes_C, NRModes_C, Mtotal, Viota, None, 0, 0, Vphi, mp_q))
                                jobs.append(proc)
                                proc.start()
                            for proc in jobs:
                                proc.join()
                            mp_results = [mp_q.get() for job in jobs]
                            FFlist += mp_results
                            sys.stderr.write(f'Mtotal = {Mtotal}, mp_{i}, FF = {mp_results}\n')
                        FFlist = np.asarray(FFlist)
                        avg = np.average(FFlist)
                        FFmin = np.min(FFlist)
                        add_csv(fresults, [[Mtotal, avg, FFmin]])
            else:
                for Mtotal in MtotalList:
                    FFlist = []
                    for phiX, kappa in product(phiXList, kappaList):
                        phic_fit_list = None
                        for iota in iotaList:
                            FF, phic_ret = calculate_Max_FF_HM_fit(EOBModes_C, NRModes_C, Mtotal_input = Mtotal, iota_input = iota, phic_input = phic_fit_list, kappa = kappa, phin = phiX)
                            sys.stderr.write(f'Mtotal = {Mtotal}, iota = {iota/np.pi} pi, kappa = {kappa/np.pi} pi, phiX = {phiX/np.pi} pi, FF = {FF}\n')
                            if phic_fit_list is None:
                                phic_fit_list = (phic_ret - np.pi*1.1/5, phic_ret + np.pi*1.1/5)
                            FFlist.append(FF)
                    FFlist = np.asarray(FFlist)
                    avg = np.average(FFlist)
                    print(f'avg = {avg}')
                    if args.save_all:
                        add_csv(fresults, [[Mtotal] + FFlist.tolist()])
                    else:
                        add_csv(fresults, [[Mtotal, avg]])
        elif 0:
            for Mtotal in MtotalList:
                FF_avg = 0
                phic_fit_list = None
                for iota in iotaList:
                    FF, phic_ret = calculate_Max_FF_HM_Circ(Mtotal_input = Mtotal, iota_input = iota, phic_input = phic_fit_list)
                    sys.stderr.write(f'Mtotal = {Mtotal}, iota = {iota/np.pi} pi, FF = {FF}, phic_ret = {phic_ret/np.pi}\n')
                    if phic_fit_list is None:
                        phic_fit_list = (phic_ret - np.pi*1.1/7, phic_ret + np.pi*1.1/7)
                    FF_avg += FF
                add_csv(fresults, [[Mtotal, FF_avg / len(iotaList)]])
        elif args.search_ecc_mtotal:
            ecc_range_new = (ecc_fit - 0.02, ecc_fit + 0.02)
            for Mtotal in MtotalList:
                MG = MultiGrid1D(estimate_ecc, ecc_range_new, 20, Mtotal = Mtotal)
                data = MG.run(fsave = None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
                ecc_grid, lnp_grid = data[:,0], data[:,1]
                ecc_fit = ecc_grid[np.argmax(lnp_grid)]
                FF_avg = 0
                for iota in iotaList:
                    sys.stderr.write(f'Mtotal = {Mtotal}, iota = {iota/np.pi} pi\n')
                    FF = calculate_Max_FF_HM(ecc_fit, Mtotal_input = Mtotal, iota_input = iota)
                    FF_avg += FF
                add_csv(fresults, [[Mtotal, ecc_fit, FF_avg / len(iotaList)]])
        elif args.search_ecc:
            ecc_range_new = (ecc_fit - 0.015, ecc_fit + 0.015)
            for Mtotal, iota in product(MtotalList, iotaList):
                sys.stderr.write(f'Mtotal = {Mtotal}, iota = {iota/np.pi} pi\n')
                MG = MultiGrid1D(calculate_Max_FF_HM, ecc_range_new, 10, Mtotal_input = Mtotal, iota_input = iota)
                data = MG.run(fsave = None, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
                indmax = np.argmax(data[:,1])
                final_FF = data[indmax,1]
                final_ecc = data[indmax, 0]
                add_csv(fresults, [[Mtotal, iota, final_ecc, final_FF]])
        else:
            for Mtotal in MtotalList:
                FF_avg = 0
                phic_fit_list = 'init'
                for iota in iotaList:
                    FF, phic_ret = calculate_Max_FF_HM(ecc_fit, Mtotal_input = Mtotal, iota_input = iota, phic_input = phic_fit_list)
                    sys.stderr.write(f'Mtotal = {Mtotal}, iota = {iota/np.pi} pi, FF = {FF}, phic_ret = {phic_ret/np.pi}\n')
                    if phic_fit_list == 'init':
                        phic_fit_list = (phic_ret - np.pi*1.1/7, phic_ret + np.pi*1.1/7)
                    FF_avg += FF
                add_csv(fresults, [[Mtotal, ecc_fit, FF_avg / len(iotaList)]])
    return 0

#-----Recover EOB vs SXS-----#
def GridSearch_KK_dtpeak(argv = None):
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL
    from .SXS import save_namecol, add_csv
    from .generator import self_adaptivor

    parser = OptionParser(description='Waveform Comparation With SXS')

    parser.add_option('--executable', type = 'str', default = DEFAULT_EXEV5, help = 'Exe command')
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
    parser.add_option('--ymode', type = 'int', default = 22, help = 'The mode.')

    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')

    parser.add_option('--num-k', type = 'int', default = 50, help = 'numbers for grid search')
    parser.add_option('--max-k', type = 'float', help = 'Upper bound of parameter')
    parser.add_option('--min-k', type = 'float', help = 'Lower bound of parameter')
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
    ymode = args.ymode
    max_dtpeak = args.max_dtpeak if args.max_dtpeak is not None else 100
    min_dtpeak = args.min_dtpeak if args.min_dtpeak is not None else -10
    dtpeak_range = (min_dtpeak, max_dtpeak)
    num_dtpeak = args.num_dtpeak

    max_k = args.max_k if args.max_k is not None else 5
    min_k = args.min_k if args.min_k is not None else 0
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

    def get_lnprob(k, dtpeak):
        ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = ymode,
                    KK = k, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak, ecc = 0.0)
        return ret[0]
    prefix = Path(args.prefix)
    if not prefix.exists():
        prefix.mkdir(parents=True)
    
    fsave = str(prefix / f'grid_{SXSnum}.txt')
    if not prefix.exists():
        prefix.mkdir(parents = True)
    if not Path(fsave).exists():
        MG = MultiGrid(get_lnprob, k_range, dtpeak_range, num_k, num_dtpeak)
        MG.run(fsave, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
    data = np.loadtxt(fsave)
    k_grid, dtpeak_grid, lnp_grid = data[:,0], data[:,1], data[:,2]
    k_fit = k_grid[np.argmax(lnp_grid)]
    dtpeak_fit = dtpeak_grid[np.argmax(lnp_grid)]
    
    lnp, FF = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = ymode,
                    KK = k_fit, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak_fit, ecc = 0)
    h22_wf = ge.get_waveform(jobtag = args.jobtag, timeout = args.timeout, verbose = True,
                    KK = k_fit, dSO = dSO_default, dSS = dSS_default, 
                    dtPeak = dtpeak_fit, ecc = 0, mode = ymode)
    wf_1, wf_2 = alignment(h22_wf, NR)
    plt.figure(figsize = (14, 7))
    plt.subplot(211)
    plt.title(f'lnp={lnp},FF={FF}')
    plt.plot(wf_1.time, wf_1.amp, label = f'EOB_{ymode}')
    plt.plot(wf_2.time, wf_2.amp, label = f'NR_{ymode}')
    plt.legend()
    plt.subplot(212)
    plt.title(f'lnp={lnp},FF={FF}')
    plt.plot(wf_1.time, wf_1.phaseFrom0, label = f'EOB_{ymode}')
    plt.plot(wf_2.time, wf_2.phaseFrom0, label = f'NR_{ymode}')
    plt.legend()
    plt.savefig(prefix / f'AmpPhase.png', dpi = 200)
    plt.close()

    Mtotal_list = np.linspace(10, 200, 500)
    # Setting saveing prefix
    fresults = prefix / f'results_{SXSnum}_{ymode}.csv'
    # Setting Results savimg filename.
    save_namecol(fresults, data = [['#q', '#chi1', '#chi2', '#Mtotal', '#FF', f'#ecc={0}']])
    ret = ge.get_overlap(jobtag = args.jobtag, minecc = 0, maxecc = 0, eccentricity = 0,
                        timeout = args.timeout, verbose = True, Mtotal = Mtotal_list, 
                        KK = k_fit, dSO = dSO_default, dSS = dSS_default, 
                        dtPeak = dtpeak_fit, ecc = 0, mode = ymode)
    length = len(Mtotal_list)
    q_list = NR.q*np.ones(len(Mtotal_list)).reshape(1,length)
    s1z_list = NR.s1z*np.ones(len(Mtotal_list)).reshape(1,length)
    s2z_list = NR.s2z*np.ones(len(Mtotal_list)).reshape(1,length)
    FF_list = ret[2].reshape(1,length)
    Mtotal_list_out = Mtotal_list.reshape(1, length)
    data = np.concatenate((q_list, s1z_list, s2z_list, Mtotal_list_out, FF_list), axis = 0)
    add_csv(fresults, data.T.tolist())

    return 0

def GridSearch_KK_dtpeak_HM(argv = None):
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL
    from .SXS import save_namecol, add_csv, ModeC_alignment
    from .generator import self_adaptivor
    from .h22datatype import calculate_Overlap_tmp

    parser = OptionParser(description='Waveform Comparation With SXS')

    parser.add_option('--executable', type = 'str', default = DEFAULT_EXEV5, help = 'Exe command')
    parser.add_option('--approx', type = 'str', default = 'SEOBNREv5', help = 'Version of the code')
    parser.add_option('--fini', type = 'float', default = 0, help = 'Initial orbital frequency')
    parser.add_option('--SXS', type = 'str', default = '0071', help = 'SXS template for comparision')
    parser.add_option('--mtotal-base', type = 'float', default = 40, help = 'Total mass')
    parser.add_option('--srate', type = 'float', default = 16384, help = 'Sample rate')
 
    parser.add_option('--prefix', type = 'str', default = '.', help = 'dir for results saving.')
    parser.add_option('--jobtag', type = 'str', default = '_lnprob', help = 'jobtag.')

    parser.add_option('--psd', type = 'str', help = 'Detector psd.')
    parser.add_option('--flow', type = 'float', default = 0, help = 'Lower frequency cut off for psd.')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')
    parser.add_option('--ymode', type = 'int', default = 22, help = 'The mode.')

    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')

    parser.add_option('--num-k', type = 'int', default = 50, help = 'numbers for grid search')
    parser.add_option('--max-k', type = 'float', help = 'Upper bound of parameter')
    parser.add_option('--min-k', type = 'float', help = 'Lower bound of parameter')
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
    # mtotal = args.mtotal
    fini = args.fini
    srate = args.srate
    table = args.table
    # srcloc = args.srcloc
    jobtag = args.jobtag
    srcloc_all = args.srcloc_all
    psd = DetectorPSD(args.psd, flow = args.flow)
    ymode = args.ymode
    max_dtpeak = args.max_dtpeak if args.max_dtpeak is not None else 100
    min_dtpeak = args.min_dtpeak if args.min_dtpeak is not None else -10
    dtpeak_range = (min_dtpeak, max_dtpeak)
    num_dtpeak = args.num_dtpeak

    max_k = args.max_k if args.max_k is not None else 5
    min_k = args.min_k if args.min_k is not None else 0
    k_range = (min_k, max_k)
    num_k = args.num_k

    eps = args.eps
    mag = args.mag
    filter_thresh = args.filter_thresh
    max_step = args.max_step
    NR = SXSAllMode(SXSnum, table = table, srcloc = srcloc_all, cutpct = 1.5)
    h22 = NR.get_mode(2,2)
    h21 = NR.get_mode(2,1)
    h33 = NR.get_mode(3,3)
    h44 = NR.get_mode(4,4)
    amp22 = np.power(np.max(h22.amp), 2)
    amp21 = np.power(np.max(h21.amp), 2)
    amp33 = np.power(np.max(h33.amp), 2)
    amp44 = np.power(np.max(h44.amp), 2)
    ampTOT = amp22 + amp21 + amp33 + amp44
    max_freq = max(h22.frequency.max(), h21.frequency.max(), h33.frequency.max(), h44.frequency.max())
    deltaT = np.pi / max_freq
    NRModetime = np.arange(h22.time[0], h22.time[-1], deltaT)
    NRModes = NR.mode_resample(NRModetime)

    m1 = NR.mQ1 * args.mtotal_base
    m2 = NR.mQ2 * args.mtotal_base
    s1x = NR.s1x
    s1y = NR.s1y
    s1z = NR.s1z
    s2x = NR.s2x
    s2y = NR.s2y
    s2z = NR.s2z
    f0 = NR.Sf_ini
    fini = fini if fini > 0 else f0 * dim_t(m1 + m2)
    sys.stderr.write(f'q = {m1/m2}, chiA = {(s1z-s2z)/2}\n')
    srate = dim_t(m1 + m2) / deltaT
    ge = Generator(approx = approx, executable = exe, verbose = True)
    pms0 = NR.CalculateAdjParamsV4()
    dSO_default = pms0[1]
    dSS_default = pms0[2]

    def get_lnprob(k, dtpeak):
        # ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = ymode,
        #             KK = k, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak, ecc = 0.0)
        ret = ge(m1 = m1, m2 = m2, s1x = s1x, s1y = s1y, s1z = s1z, 
                s2x = s2x, s2y = s2y, s2z = s2z, D = 100, 
                KK = k, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak,
                ecc = 0.0, srate = srate, f_ini = fini, L = 2, M = 2,
                timeout = 3600, jobtag = jobtag, mode = -1)
        if isinstance(ret, CEV):
            return -np.inf
        t, h22r, h22i, h21r, h21i, h33r, h33i, h44r, h44i = \
            ret[:,0], ret[:,1], ret[:,2], ret[:,3], ret[:,4], ret[:,5], ret[:,6], ret[:,7], ret[:,8]   
        # hEOB22 = ModeBase(t, h22r, h22i)
        # hEOB21 = ModeBase(t, h21r, h21i)
        # hEOB33 = ModeBase(t, h33r, h33i)
        # hEOB44 = ModeBase(t, h44r, h44i)
        EOBModes = waveform_mode_collector(0)
        EOBModes.append_mode(t, h22r, h22i, 2, 2)
        EOBModes.append_mode(t, h22r, -h22i, 2, -2)
        EOBModes.append_mode(t, h21r, h21i, 2, 1)
        EOBModes.append_mode(t, h21r, -h21i, 2, -1)
        EOBModes.append_mode(t, h33r, h33i, 3, 3)
        EOBModes.append_mode(t, -h33r, h33i, 3, -3)
        EOBModes.append_mode(t, h44r, h44i, 4, 4)
        EOBModes.append_mode(t, h44r, -h44i, 4, -4)
        EOBModes_C, NRModes_C = ModeC_alignment(EOBModes, NRModes)
        NFFT = len(EOBModes_C.time)
        MtotalList_ini = [40, 120]
        lnplist = []
        dtM = EOBModes_C.time[1] - EOBModes_C.time[0]
        for mt in MtotalList_ini:
            dt = dtM / dim_t(mt)
            df = 1./NFFT/dt
            freqs = np.abs(np.fft.fftfreq(NFFT, dt))
            power_vec = psd(freqs)
            Oxt22 = calculate_Overlap_tmp(EOBModes_C.get_mode(2,2), NRModes_C.get_mode(2,2), power_vec, df, NFFT)
            Oxt21 = calculate_Overlap_tmp(EOBModes_C.get_mode(2,1), NRModes_C.get_mode(2,1), power_vec, df, NFFT)
            Oxt33 = calculate_Overlap_tmp(EOBModes_C.get_mode(3,3), NRModes_C.get_mode(3,3), power_vec, df, NFFT)
            Oxt44 = calculate_Overlap_tmp(EOBModes_C.get_mode(4,4), NRModes_C.get_mode(4,4), power_vec, df, NFFT)
            ln22 = -np.power((1 - Oxt22)/0.01, 2) * amp22 / ampTOT
            ln21 = -np.power((1 - Oxt21)/0.01, 2) * amp21 / ampTOT
            ln33 = -np.power((1 - Oxt33)/0.01, 2) * amp33 / ampTOT
            ln44 = -np.power((1 - Oxt44)/0.01, 2) * amp44 / ampTOT
            lnp = ln22 + ln21 + ln33 + ln44
            idx = np.argmax(lnp)
            lth = len(lnp)
            if idx > lth / 2:
                tc = (idx - lth) * dtM
            else:
                tc = idx * dtM
            lnp = lnp - np.power(tc/5, 2)
            lnplist.append(lnp)
        ret = np.min(lnplist)
        if np.isnan(ret):
            sys.stderr.write(f'lnp is nan!')
            return -np.inf
        return ret
    prefix = Path(args.prefix)
    if not prefix.exists():
        prefix.mkdir(parents=True)
    
    fsave = str(prefix / f'grid_{SXSnum}.txt')
    if not prefix.exists():
        prefix.mkdir(parents = True)
    if not Path(fsave).exists():
        MG = MultiGrid(get_lnprob, k_range, dtpeak_range, num_k, num_dtpeak)
        MG.run(fsave, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
    # data = np.loadtxt(fsave)
    # k_grid, dtpeak_grid, lnp_grid = data[:,0], data[:,1], data[:,2]
    # k_fit = k_grid[np.argmax(lnp_grid)]
    # dtpeak_fit = dtpeak_grid[np.argmax(lnp_grid)]
    
    ret = ge(m1 = m1, m2 = m2, s1x = s1x, s1y = s1y, s1z = s1z, 
            s2x = s2x, s2y = s2y, s2z = s2z, D = 100, 
            KK = k_fit, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak_fit,
            ecc = 0.0, srate = srate, f_ini = fini, L = 2, M = 2,
            timeout = 3600, jobtag = jobtag, mode = -1)
    if isinstance(ret, CEV):
        return -np.inf
    t, h22r, h22i, h21r, h21i, h33r, h33i, h44r, h44i = \
        ret[:,0], ret[:,1], ret[:,2], ret[:,3], ret[:,4], ret[:,5], ret[:,6], ret[:,7], ret[:,8]   
    EOBModes = waveform_mode_collector(0)
    EOBModes.append_mode(t, h22r, h22i, 2, 2)
    EOBModes.append_mode(t, h22r, -h22i, 2, -2)
    EOBModes.append_mode(t, h21r, h21i, 2, 1)
    EOBModes.append_mode(t, h21r, -h21i, 2, -1)
    EOBModes.append_mode(t, h33r, h33i, 3, 3)
    EOBModes.append_mode(t, -h33r, h33i, 3, -3)
    EOBModes.append_mode(t, h44r, h44i, 4, 4)
    EOBModes.append_mode(t, h44r, -h44i, 4, -4)
    EOBModes_C, NRModes_C = ModeC_alignment(EOBModes, NRModes)
    wf_1 = EOBModes_C.get_mode(2,2)
    wf_2 = NRModes_C.get_mode(2,2)
    # wf_1, wf_2 = alignment(h22_wf, NR)
    plt.figure(figsize = (14, 7))
    plt.subplot(211)
    plt.plot(wf_1.time, wf_1.amp, label = f'EOB_{ymode}')
    plt.plot(wf_2.time, wf_2.amp, label = f'NR_{ymode}')
    plt.legend()
    plt.subplot(212)
    plt.plot(wf_1.time, wf_1.phaseFrom0, label = f'EOB_{ymode}')
    plt.plot(wf_2.time, wf_2.phaseFrom0, label = f'NR_{ymode}')
    plt.legend()
    plt.savefig(prefix / f'AmpPhase.png', dpi = 200)
    plt.close()

    # Mtotal_list = np.linspace(10, 200, 500)
    # # Setting saveing prefix
    # fresults = prefix / f'results_{SXSnum}_{ymode}.csv'
    # # Setting Results savimg filename.
    # save_namecol(fresults, data = [['#q', '#chi1', '#chi2', '#Mtotal', '#FF', f'#ecc={0}']])
    # ret = ge.get_overlap(jobtag = args.jobtag, minecc = 0, maxecc = 0, eccentricity = 0,
    #                     timeout = args.timeout, verbose = True, Mtotal = Mtotal_list, 
    #                     KK = k_fit, dSO = dSO_default, dSS = dSS_default, 
    #                     dtPeak = dtpeak_fit, ecc = 0, mode = ymode)
    # length = len(Mtotal_list)
    # q_list = NR.q*np.ones(len(Mtotal_list)).reshape(1,length)
    # s1z_list = NR.s1z*np.ones(len(Mtotal_list)).reshape(1,length)
    # s2z_list = NR.s2z*np.ones(len(Mtotal_list)).reshape(1,length)
    # FF_list = ret[2].reshape(1,length)
    # Mtotal_list_out = Mtotal_list.reshape(1, length)
    # data = np.concatenate((q_list, s1z_list, s2z_list, Mtotal_list_out, FF_list), axis = 0)
    # add_csv(fresults, data.T.tolist())

    return 0


class V5Calibrator(object):
    def __init__(self, 
                SXSnum, 
                fini, 
                mtotal, 
                srate, 
                table, 
                srcloc22, 
                srcloc21, 
                srcloc33, 
                srcloc44,
                srcloc_all):
        self._NR22 = SXSh22(SXSnum = SXSnum,
                    f_ini = fini,
                    Mtotal = mtotal,
                    srate = srate,
                    srcloc = srcloc22,
                    table = table,
                    srcloc_all = srcloc_all)

        self._NR21 = SXSh22(SXSnum = SXSnum,
                    f_ini = fini,
                    Mtotal = mtotal,
                    srate = srate,
                    srcloc = srcloc21,
                    table = table,
                    srcloc_all = srcloc_all)

        self._NR33 = SXSh22(SXSnum = SXSnum,
                    f_ini = fini,
                    Mtotal = mtotal,
                    srate = srate,
                    srcloc = srcloc33,
                    table = table,
                    srcloc_all = srcloc_all)

        self._NR44 = SXSh22(SXSnum = SXSnum,
                    f_ini = fini,
                    Mtotal = mtotal,
                    srate = srate,
                    srcloc = srcloc44,
                    table = table,
                    srcloc_all = srcloc_all)

    @property
    def NR22(self):
        return self._NR22
    @property
    def NR21(self):
        return self._NR21
    @property
    def NR33(self):
        return self._NR33
    @property
    def NR44(self):
        return self._NR44
    @property
    def NRDict(self):
        return {22:self._NR22, 21:self._NR21, 33:self._NR33, 44:self._NR44}
    
    def construct_generator(self, exe, approx, psd):
        ge22 = self._NR22.construct_generator(approx, exe, psd = psd)
        ge21 = self._NR21.construct_generator(approx, exe, psd = psd)
        ge33 = self._NR33.construct_generator(approx, exe, psd = psd)
        ge44 = self._NR44.construct_generator(approx, exe, psd = psd)
        return {22:ge22, 21:ge21, 33:ge33, 44:ge44}
    
    def CalculateAdjParamsV4(self):
        return self._NR22.CalculateAdjParamsV4()    

def GridSearch_calibrator_allmode(argv = None):
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL
    from .SXS import save_namecol, add_csv
    from .generator import self_adaptivor

    parser = OptionParser(description='Waveform Comparation With SXS')

    parser.add_option('--executable', type = 'str', default = DEFAULT_EXEV5, help = 'Exe command')
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

    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc22', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc21', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc33', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc44', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')

    parser.add_option('--num-dtpeak', type = 'int', default = 50, help = 'numbers for grid search')
    parser.add_option('--max-dtpeak', type = 'float', help = 'Upper bound of parameter')
    parser.add_option('--min-dtpeak', type = 'float', help = 'Lower bound of parameter')
    parser.add_option('--delta-dtpeak', type = 'float', help = 'Range of dt around dtV4')

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
    srcloc22 = args.srcloc22
    srcloc21 = args.srcloc21
    srcloc33 = args.srcloc33
    srcloc44 = args.srcloc44
    srcloc_all = args.srcloc_all
    psd = DetectorPSD(args.psd, flow = args.flow)
    ymodelist = (22, 21, 33, 44)
    eps = args.eps
    mag = args.mag
    filter_thresh = args.filter_thresh
    max_step = args.max_step

    V5NR = V5Calibrator(SXSnum = SXSnum,
                        fini = fini,
                        mtotal = mtotal,
                        srate = srate,
                        table = table,
                        srcloc22 = srcloc22,
                        srcloc21 = srcloc21,
                        srcloc33 = srcloc33,
                        srcloc44 = srcloc44,
                        srcloc_all = srcloc_all)
    V5ge = V5NR.construct_generator(exe, approx, psd = psd)
    # Amp22 = np.power(np.max(V5NR.NR22.amp), 2)
    # Amp21 = np.power(np.max(V5NR.NR21.amp), 2)
    # Amp33 = np.power(np.max(V5NR.NR33.amp), 2)
    # Amp44 = np.power(np.max(V5NR.NR44.amp), 2)
    # AmpTotal = Amp22 + Amp21 + Amp33 + Amp44
    pms0 = V5NR.CalculateAdjParamsV4()
    KK_default = pms0[0]
    dSO_default = pms0[1]
    dSS_default = pms0[2]
    if args.delta_dtpeak:
        max_dtpeak = pms0[3] + args.delta_dtpeak
        min_dtpeak = pms0[3] - args.delta_dtpeak
    else:
        max_dtpeak = args.max_dtpeak if args.max_dtpeak is not None else 100
        min_dtpeak = args.min_dtpeak if args.min_dtpeak is not None else -10
    dtpeak_range = (min_dtpeak, max_dtpeak)
    num_dtpeak = args.num_dtpeak
    ge22 = V5ge[22]
    # ge21 = V5ge[21]
    # ge33 = V5ge[33]
    # ge44 = V5ge[44]
    def get_lnprob(dtpeak):
        ret22, _ = ge22.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = 22,
                    KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak, ecc = 0.0)
        # ret21, _ = ge21.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = 21,
        #             KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak, ecc = 0.0)
        # ret33, _ = ge33.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = 33,
        #             KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak, ecc = 0.0)
        # ret44, _ = ge44.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = 44,
        #             KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak, ecc = 0.0)
        # return (ret22 * Amp22 + ret21 * Amp21 + ret33 * Amp33 + ret44 * Amp44) / AmpTotal
        return ret22
    prefix = Path(args.prefix)
    if not prefix.exists():
        prefix.mkdir(parents=True)
    fsave = str(prefix / f'grid_{SXSnum}.txt')
    if not prefix.exists():
        prefix.mkdir(parents = True)
    if not Path(fsave).exists():
        MG = MultiGrid1D(get_lnprob, dtpeak_range, num_dtpeak)
        MG.run(fsave, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
    data = np.loadtxt(fsave)
    dtpeak_grid, lnp_grid = data[:,0], data[:,1]
    dtpeak_fit = dtpeak_grid[np.argmax(lnp_grid)]
    
    if 0:
        for ymode in ymodelist:
            ge = V5ge[ymode]
            NRXX = V5NR.NRDict[ymode]
            lnp, FF = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = ymode,
                            KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak_fit, ecc = 0)
            h22_wf = ge.get_waveform(jobtag = args.jobtag, timeout = args.timeout, verbose = True,
                            KK = KK_default, dSO = dSO_default, dSS = dSS_default, 
                            dtPeak = dtpeak_fit, ecc = 0, mode = ymode)
            wf_1, wf_2 = alignment(h22_wf, NRXX)
            plt.figure(figsize = (14, 7))
            plt.subplot(211)
            plt.title(f'lnp={lnp},FF={FF}')
            plt.plot(wf_1.time, wf_1.amp, label = f'EOB_{ymode}')
            plt.plot(wf_2.time, wf_2.amp, label = f'NR_{ymode}')
            plt.legend()
            plt.subplot(212)
            plt.title(f'lnp={lnp},FF={FF}')
            plt.plot(wf_1.time, wf_1.phaseFrom0, label = f'EOB_{ymode}')
            plt.plot(wf_2.time, wf_2.phaseFrom0, label = f'NR_{ymode}')
            plt.legend()
            plt.savefig(prefix / f'AmpPhase.png', dpi = 200)
            plt.close()

            Mtotal_list = np.linspace(10, 200, 500)
            # Setting saveing prefix
            fresults = prefix / f'results_{SXSnum}_{ymode}.csv'
            # Setting Results savimg filename.
            save_namecol(fresults, data = [['#q', '#chi1', '#chi2', '#Mtotal', '#FF', f'#ecc={0}']])
            ret = ge22.get_overlap(jobtag = args.jobtag, minecc = 0, maxecc = 0, eccentricity = 0,
                                timeout = args.timeout, verbose = True, Mtotal = Mtotal_list, 
                                KK = KK_default, dSO = dSO_default, dSS = dSS_default, 
                                dtPeak = dtpeak_fit, ecc = 0, mode = ymode)
            length = len(Mtotal_list)
            q_list = NRXX.q*np.ones(len(Mtotal_list)).reshape(1,length)
            s1z_list = NRXX.s1z*np.ones(len(Mtotal_list)).reshape(1,length)
            s2z_list = NRXX.s2z*np.ones(len(Mtotal_list)).reshape(1,length)
            FF_list = ret[2].reshape(1,length)
            Mtotal_list_out = Mtotal_list.reshape(1, length)
            data = np.concatenate((q_list, s1z_list, s2z_list, Mtotal_list_out, FF_list), axis = 0)
            add_csv(fresults, data.T.tolist())
    return 0

def GridSearch_Wdt_calibrator_allmode(argv = None):
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL
    from .SXS import save_namecol, add_csv
    from .generator import self_adaptivor

    parser = OptionParser(description='Waveform Comparation With SXS')

    parser.add_option('--executable', type = 'str', default = DEFAULT_EXEV5, help = 'Exe command')
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

    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc22', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc21', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc33', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc44', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')

    parser.add_option('--num-wdt', type = 'int', default = 50, help = 'numbers for grid search')
    parser.add_option('--max-wdt', type = 'float', default = 50, help = 'Upper bound of parameter')
    parser.add_option('--min-wdt', type = 'float', default = 0.1, help = 'Lower bound of parameter')

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
    srcloc22 = args.srcloc22
    srcloc21 = args.srcloc21
    srcloc33 = args.srcloc33
    srcloc44 = args.srcloc44
    srcloc_all = args.srcloc_all
    psd = DetectorPSD(args.psd, flow = args.flow)
    ymodelist = (22, 21, 33, 44)
    eps = args.eps
    mag = args.mag
    filter_thresh = args.filter_thresh
    max_step = args.max_step

    V5NR = V5Calibrator(SXSnum = SXSnum,
                        fini = fini,
                        mtotal = mtotal,
                        srate = srate,
                        table = table,
                        srcloc22 = srcloc22,
                        srcloc21 = srcloc21,
                        srcloc33 = srcloc33,
                        srcloc44 = srcloc44,
                        srcloc_all = srcloc_all)
    V5ge = V5NR.construct_generator(exe, approx, psd = psd)
    # Amp22 = np.power(np.max(V5NR.NR22.amp), 2)
    # Amp21 = np.power(np.max(V5NR.NR21.amp), 2)
    # Amp33 = np.power(np.max(V5NR.NR33.amp), 2)
    # Amp44 = np.power(np.max(V5NR.NR44.amp), 2)
    # AmpTotal = Amp22 + Amp21 + Amp33 + Amp44
    pms0 = V5NR.CalculateAdjParamsV4()
    KK_default = pms0[0]
    dSO_default = pms0[1]
    dSS_default = pms0[2]
    if args.delta_dtpeak:
        max_dtpeak = pms0[3] + args.delta_dtpeak
        min_dtpeak = pms0[3] - args.delta_dtpeak
    else:
        max_dtpeak = args.max_dtpeak if args.max_dtpeak is not None else 100
        min_dtpeak = args.min_dtpeak if args.min_dtpeak is not None else -10
    dtpeak_range = (min_dtpeak, max_dtpeak)
    num_dtpeak = args.num_dtpeak
    ge22 = V5ge[22]
    # ge21 = V5ge[21]
    # ge33 = V5ge[33]
    # ge44 = V5ge[44]
    def get_lnprob(dtpeak):
        ret22, _ = ge22.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = 22,
                    KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak, ecc = 0.0)
        # ret21, _ = ge21.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = 21,
        #             KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak, ecc = 0.0)
        # ret33, _ = ge33.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = 33,
        #             KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak, ecc = 0.0)
        # ret44, _ = ge44.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = 44,
        #             KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak, ecc = 0.0)
        # return (ret22 * Amp22 + ret21 * Amp21 + ret33 * Amp33 + ret44 * Amp44) / AmpTotal
        return ret22
    prefix = Path(args.prefix)
    if not prefix.exists():
        prefix.mkdir(parents=True)
    fsave = str(prefix / f'grid_{SXSnum}.txt')
    if not prefix.exists():
        prefix.mkdir(parents = True)
    if not Path(fsave).exists():
        MG = MultiGrid1D(get_lnprob, dtpeak_range, num_dtpeak)
        MG.run(fsave, eps = eps, magnification = mag, filter_thresh = filter_thresh, maxiter = max_step)
    data = np.loadtxt(fsave)
    dtpeak_grid, lnp_grid = data[:,0], data[:,1]
    dtpeak_fit = dtpeak_grid[np.argmax(lnp_grid)]
    
    if 0:
        for ymode in ymodelist:
            ge = V5ge[ymode]
            NRXX = V5NR.NRDict[ymode]
            lnp, FF = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, mode = ymode,
                            KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtpeak_fit, ecc = 0)
            h22_wf = ge.get_waveform(jobtag = args.jobtag, timeout = args.timeout, verbose = True,
                            KK = KK_default, dSO = dSO_default, dSS = dSS_default, 
                            dtPeak = dtpeak_fit, ecc = 0, mode = ymode)
            wf_1, wf_2 = alignment(h22_wf, NRXX)
            plt.figure(figsize = (14, 7))
            plt.subplot(211)
            plt.title(f'lnp={lnp},FF={FF}')
            plt.plot(wf_1.time, wf_1.amp, label = f'EOB_{ymode}')
            plt.plot(wf_2.time, wf_2.amp, label = f'NR_{ymode}')
            plt.legend()
            plt.subplot(212)
            plt.title(f'lnp={lnp},FF={FF}')
            plt.plot(wf_1.time, wf_1.phaseFrom0, label = f'EOB_{ymode}')
            plt.plot(wf_2.time, wf_2.phaseFrom0, label = f'NR_{ymode}')
            plt.legend()
            plt.savefig(prefix / f'AmpPhase.png', dpi = 200)
            plt.close()

            Mtotal_list = np.linspace(10, 200, 500)
            # Setting saveing prefix
            fresults = prefix / f'results_{SXSnum}_{ymode}.csv'
            # Setting Results savimg filename.
            save_namecol(fresults, data = [['#q', '#chi1', '#chi2', '#Mtotal', '#FF', f'#ecc={0}']])
            ret = ge22.get_overlap(jobtag = args.jobtag, minecc = 0, maxecc = 0, eccentricity = 0,
                                timeout = args.timeout, verbose = True, Mtotal = Mtotal_list, 
                                KK = KK_default, dSO = dSO_default, dSS = dSS_default, 
                                dtPeak = dtpeak_fit, ecc = 0, mode = ymode)
            length = len(Mtotal_list)
            q_list = NRXX.q*np.ones(len(Mtotal_list)).reshape(1,length)
            s1z_list = NRXX.s1z*np.ones(len(Mtotal_list)).reshape(1,length)
            s2z_list = NRXX.s2z*np.ones(len(Mtotal_list)).reshape(1,length)
            FF_list = ret[2].reshape(1,length)
            Mtotal_list_out = Mtotal_list.reshape(1, length)
            data = np.concatenate((q_list, s1z_list, s2z_list, Mtotal_list_out, FF_list), axis = 0)
            add_csv(fresults, data.T.tolist())
    return 0


from .generator import CompGenerator
def mode_compare(argv = None):
    from .SXS import save_namecol, add_csv
    import time as pyt
    parser = OptionParser(description='General model compare.')
    parser.add_option('--approx1', type = 'str', default = 'SEOBNRv1', help = 'approx1 for compare')
    parser.add_option('--approx2', type = 'str', default = 'SEOBNRv4', help = 'approx2 for compare')
    parser.add_option('--executable1', type = 'str', default = 'lalsim-inspiral', help = 'command for waveform generation')
    parser.add_option('--executable2', type = 'str', default = 'lalsim-inspiral', help = 'command for waveform generation')
    parser.add_option('--prefix', type = 'str', default = '.', help = 'prefix for data save')
    parser.add_option('--verbose', action = 'store_true', help = 'If added, will print verbose message.')
    parser.add_option('--mtotal', type = 'float', default = 16, help = 'Total mass of the binary system')

    parser.add_option('--ecc', type = 'float', action = 'append', default = [], help = 'eccentricity')
    parser.add_option('--fini', type = 'float', default = 0.00123, help = 'Initial orbit frequency')
    parser.add_option('--natural', action = 'store_true', help = 'If added, will use natural dimension for fini.')
    parser.add_option('--distance', type = 'float', default = 100, help = 'BBH distance in Mpc')
    parser.add_option('--srate', type = 'float', default = 16384, help = 'Sample rate')
    parser.add_option('--jobtag', type = 'str', default = '_wfcomp', help = 'Tag for this run')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')
    parser.add_option('--psd', type = 'str', help = 'Detector psd.')
    parser.add_option('--flow', type = 'float', default = 0, help = 'Lower frequency cut off for psd.')
    parser.add_option('--f-cut', type = 'float', help = 'Lower frequency cut off for psd.')
    parser.add_option('--ymode', type = 'int', default = 22, help = 'Spherical mode, in (22, 21, 33, 44)')
    # Random mode
    parser.add_option('--random', action = 'store_true', help = 'If added, will use random parameters.')
    parser.add_option('--min-mratio', type = 'float', default = 1, help = 'Used in random mode [1]')
    parser.add_option('--max-mratio', type = 'float', default = 9, help = 'Used in random mode [9]')
    parser.add_option('--prec', action = 'store_true', help = 'use prec')
    parser.add_option('--min-spin1z', type = 'float',  help = 'Used in random mode')
    parser.add_option('--max-spin1z', type = 'float',  help = 'Used in random mode')
    parser.add_option('--min-spin2z', type = 'float',  help = 'Used in random mode')
    parser.add_option('--max-spin2z', type = 'float',  help = 'Used in random mode')
    parser.add_option('--min-ecc', type = 'float',  help = 'Used in random mode')
    parser.add_option('--max-ecc', type = 'float',  help = 'Used in random mode')
    parser.add_option('--min-mtotal', type = 'float', help = 'Min total mass of the binary system')
    parser.add_option('--max-mtotal', type = 'float', help = 'Max total mass of the binary system')
    parser.add_option('--seed', type = 'int', help = 'random seed')
    parser.add_option('--ncompare', type = 'int', default = 10, help = 'Used in random mode [10]')

    parser.add_option('--mratio', type = 'float', default = 1.0, help = 'specific pms')
    parser.add_option('--spin1z', type = 'float', default = 0.0, help = 'specific pms')
    parser.add_option('--spin2z', type = 'float', default = 0.0, help = 'specific pms')

    args, _empty = parser.parse_args(argv)
    approx1 = args.approx1
    approx2 = args.approx2
    exe1 = args.executable1
    exe2 = args.executable2
    verbose = args.verbose
        
    Mtotal = args.mtotal
    D = args.distance
    f_ini = args.fini
    srate = args.srate
    timeout = args.timeout
    jobtag = args.jobtag
    psd = DetectorPSD(args.psd, flow = args.flow)
    prefix = Path(args.prefix)
    if len(prefix.parts) <= 1:
        fname = 'all.csv'
        savedir = prefix
    else:
        savedir = prefix.parent
        fname = f'{prefix.parts[-1]}.csv'

    if not savedir.exists():
        savedir.mkdir(parents = True)
    
    # 1. save all
    namecol = [['#Mtotal',
               '#mass_ratio',
               '#spin1x',
               '#spin1y',
               '#spin1z',
               '#spin2x',
               '#spin2y',
               '#spin2z',
               '#ecc',
               '#FF']]
    fsave = savedir / fname
    # save_namecol(fsave, data = namecol)
    if args.seed is not None:
        seed = int(pyt.time()%10000 / args.seed) + int(pyt.time()%args.seed)
    else:
        seed = int(pyt.time()%10000)
    np.random.seed(seed)
    Comp = CompGenerator(approx1, exe1, approx2, exe2, psd = psd, verbose = verbose)
    if args.ncompare > 1:
        Comp.compare_random(args.min_mratio, args.max_mratio, 
                            args.min_spin1z, args.max_spin1z, 
                            args.min_spin2z, args.max_spin2z, 
                            args.min_ecc, args.max_ecc, fsave,
                            Num = args.ncompare, 
                            Mtotal = Mtotal, 
                            min_Mtotal = args.min_mtotal, max_Mtotal = args.max_mtotal,
                            D = D, f_ini = f_ini, 
                            srate = srate, jobtag = jobtag, timeout = timeout,
                            mode = args.ymode, 
                            use_prec = args.prec, use_fcut = args.f_cut)     
    else:
        eccList = args.ecc
        if len(eccList) == 0:
            eccList = [0.0]
        mtotal_list = np.linspace(10, 200, 30)
        for ecc in eccList:
            ret = Comp.core_calcFF(args.mratio, mtotal_list, 
                        args.spin1z, args.spin2z, ecc,
                        D, f_ini, srate, timeout, jobtag, use_fcut = None)
            sys.stderr.write(f'PMS: q = {args.mratio}, s1z = {args.spin1z}, s2z = {args.spin2z} ecc = {ecc}\n\t FF = {ret}\n\n')
    return 0

def mode_compare_ecc(argv = None):
    from .SXS import save_namecol, add_csv
    import time as pyt
    parser = OptionParser(description='General model compare.')
    parser.add_option('--approx1', type = 'str', default = 'SEOBNRv1', help = 'approx1 for compare')
    parser.add_option('--approx2', type = 'str', default = 'SEOBNRv4', help = 'approx2 for compare')
    parser.add_option('--executable1', type = 'str', default = 'lalsim-inspiral', help = 'command for waveform generation')
    parser.add_option('--executable2', type = 'str', default = 'lalsim-inspiral', help = 'command for waveform generation')
    parser.add_option('--prefix', type = 'str', default = '.', help = 'prefix for data save')
    parser.add_option('--verbose', action = 'store_true', help = 'If added, will print verbose message.')

    parser.add_option('--fini', type = 'float', default = 0.002, help = 'Initial orbit frequency')
    parser.add_option('--natural', action = 'store_true', help = 'If added, will use natural dimension for fini.')
    parser.add_option('--distance', type = 'float', default = 100, help = 'BBH distance in Mpc')
    parser.add_option('--srate', type = 'float', default = 16384, help = 'Sample rate')
    parser.add_option('--jobtag', type = 'str', default = '_wfcomp', help = 'Tag for this run')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')
    parser.add_option('--psd', type = 'str', help = 'Detector psd.')
    parser.add_option('--flow', type = 'float', default = 0, help = 'Lower frequency cut off for psd.')
    parser.add_option('--ymode', type = 'int', default = 22, help = 'Spherical mode, in (22, 21, 33, 44)')
    # Random mode
    parser.add_option('--min-mratio', type = 'float', default = 1, help = 'Used in random mode [1]')
    parser.add_option('--max-mratio', type = 'float', default = 9, help = 'Used in random mode [9]')
    parser.add_option('--min-spin1z', type = 'float',  help = 'Used in random mode')
    parser.add_option('--max-spin1z', type = 'float',  help = 'Used in random mode')
    parser.add_option('--min-spin2z', type = 'float',  help = 'Used in random mode')
    parser.add_option('--max-spin2z', type = 'float',  help = 'Used in random mode')
    parser.add_option('--min-ecc', type = 'float', default = 0.0, help = 'Used in random mode')
    parser.add_option('--max-ecc', type = 'float', default = 0.6, help = 'Used in random mode')
    parser.add_option('--necc', type = 'int', default = 10, help = 'Used in random mode [10]')
    parser.add_option('--min-mtotal', type = 'float', help = 'Min total mass of the binary system')
    parser.add_option('--max-mtotal', type = 'float', help = 'Max total mass of the binary system')
    parser.add_option('--seed', type = 'int', help = 'random seed')
    parser.add_option('--ncompare', type = 'int', default = 10, help = 'Used in random mode [10]')
    
    args, _empty = parser.parse_args(argv)
    approx1 = args.approx1
    approx2 = args.approx2
    exe1 = args.executable1
    exe2 = args.executable2
    verbose = args.verbose
        
    D = args.distance
    f_ini = args.fini
    srate = args.srate
    timeout = args.timeout
    jobtag = args.jobtag
    psd = DetectorPSD(args.psd, flow = args.flow)
    prefix = Path(args.prefix)
    if len(prefix.parts) <= 1:
        fname = 'all.csv'
        savedir = prefix
    else:
        savedir = prefix.parent
        fname = f'{prefix.parts[-1]}.csv'

    if not savedir.exists():
        savedir.mkdir(parents = True)
    eccList = np.linspace(args.min_ecc, args.max_ecc, args.necc)
    fsave = savedir / fname
    np.savetxt(savedir / 'eccList.dat', eccList)

    if args.seed is not None:
        seed = int(pyt.time()%10000 / args.seed) + int(pyt.time()%args.seed) + args.seed
    else:
        seed = int(pyt.time()%10000)
    np.random.seed(seed)
    Comp = CompGenerator(approx1, exe1, approx2, exe2, psd = psd, verbose = verbose)
    Comp.comp_random_ecc(args.min_mratio, args.max_mratio, 
                        args.min_spin1z, args.max_spin1z, 
                        args.min_spin2z, args.max_spin2z, 
                        fsave, args.min_ecc, args.max_ecc,
                        Num = args.ncompare, NumEcc = args.necc,
                        min_Mtotal = args.min_mtotal, max_Mtotal = args.max_mtotal,
                        D = D, f_ini = f_ini, 
                        srate = srate, jobtag = jobtag, timeout = timeout,
                        mode = args.ymode)     
    return

def calculate_energyflux_HMparts(argv = None):
    from .SXS import save_namecol, add_csv, ModeC_alignment
    from .generator import self_adaptivor
    import h5py

    parser = OptionParser(description='Waveform Comparation With SXS')

    parser.add_option('--executable', type = 'str', default = DEFAULT_EXEV5, help = 'Exe command')
    parser.add_option('--approx', type = 'str', default = 'SEOBNREv5', help = 'Version of the code')
    parser.add_option('--verbose', action = 'store_true', help = 'If added, will print verbose message.')

    parser.add_option('--prefix', type = 'str', default = '.', help = 'dir for results saving.')
    parser.add_option('--jobtag', type = 'str', default = '_lnprob', help = 'jobtag.')
    parser.add_option('--fini', type = 'float', default = 0.002, help = 'Initial orbit frequency')

    parser.add_option('--timeout', type = 'int', default = 3600, help = 'Time limit for waveform generation')

    parser.add_option('--num-ecc', type = 'int', default = 30, help = 'numbers for grid search')
    parser.add_option('--max-ecc', type = 'float', default = 0.7, help = 'Upper bound of parameter')
    parser.add_option('--min-ecc', type = 'float', default = 0.0, help = 'Lower bound of parameter')

    parser.add_option('--min-mratio', type = 'float', default = 1, help = 'Used in random mode [1]')
    parser.add_option('--max-mratio', type = 'float', default = 9, help = 'Used in random mode [9]')
    parser.add_option('--min-spin1z', type = 'float', default = -0.99,  help = 'Used in random mode')
    parser.add_option('--max-spin1z', type = 'float', default = 0.99, help = 'Used in random mode')
    parser.add_option('--min-spin2z', type = 'float', default = -0.99, help = 'Used in random mode')
    parser.add_option('--max-spin2z', type = 'float', default = 0.99, help = 'Used in random mode')

    parser.add_option('--seed', type = 'int', help = 'random seed')
    parser.add_option('--ncompare', type = 'int', default = 10, help = 'Used in random mode [10]')

    parser.add_option('--mratio', type = 'float', default = 1, help = 'use grid search')

    args, _ = parser.parse_args(argv)

    exe = args.executable
    approx = args.approx
    jobtag = args.jobtag

    prefix = Path(args.prefix)
    if not prefix.exists():
        prefix.mkdir(parents = True)
    mtotal_base = 40
    # s1zList = np.random.uniform(args.min_spin1z, args.max_spin1z, args.ncompare)
    # s2zList = np.random.uniform(args.min_spin2z, args.max_spin2z, args.ncompare)
    # qList = np.random.uniform(args.min_mratio, args.max_mratio, args.ncompare)
    s1zList = np.array([-0.9, -0.4, 0.0, 0.4, 0.9])
    s2zList = s1zList.copy()
    eccList = np.linspace(args.min_ecc, args.max_ecc, args.num_ecc)
    ge = Generator(approx = approx, executable = exe, verbose = args.verbose)
    ind = 0
    for s1z, s2z in product(s1zList, s2zList):
        ind += 1
        q = max(1, args.mratio)
        # s1z = s1zList[i]
        # s2z = s2zList[i]
        m1 = mtotal_base * q / (1+q)
        m2 = mtotal_base / (1+q)
        srate = 16384

        fini = args.fini * dim_t(m1 + m2)
        fsave = prefix / f'results_{jobtag}_{ind}.h5'

        #===========================================================================
        # Create a HDF5 file.
        f = h5py.File(str(fsave), "w")    # mode = {'w', 'r', 'a'}
        h5g_params = f.create_group("params")
        h5g_params['m1'] = m1
        h5g_params['m2'] = m2
        h5g_params['spin1z'] = s1z
        h5g_params['spin2z'] = s2z
        h5g_params['fini'] = fini
        h5g_params['srate'] = srate
        dEccList = h5g_params.create_dataset("eccentricities", (len(eccList), ), 'float')
        dEccList[...] = eccList
        f.close()
        for j, ecc in enumerate(eccList):
            ret1 = ge(m1 = m1, m2 = m2, s1z = s1z, s2z = s2z, D = 100, 
                    ecc = ecc, srate = srate, f_ini = fini, L = 2, M = 2,
                    timeout = 3600, jobtag = jobtag, mode = -1)
            if isinstance(ret1, CEV):
                continue
            t, h22r, h22i, h21r, h21i, h33r, h33i, h44r, h44i = \
                ret1[:,0], ret1[:,1], ret1[:,2], ret1[:,3], ret1[:,4], ret1[:,5], ret1[:,6], ret1[:,7], ret1[:,8]
            EOBModes = waveform_mode_collector(0)
            EOBModes.append_mode(t, h22r, h22i, 2, 2)
            EOBModes.append_mode(t, h22r, -h22i, 2, -2)
            EOBModes.append_mode(t, h21r, h21i, 2, 1)
            EOBModes.append_mode(t, h21r, -h21i, 2, -1)
            EOBModes.append_mode(t, h33r, h33i, 3, 3)
            EOBModes.append_mode(t, -h33r, h33i, 3, -3)
            EOBModes.append_mode(t, h44r, h44i, 4, 4)
            lenData = len(EOBModes.time)
            f = h5py.File(str(fsave), "a")    # mode = {'w', 'r', 'a'}
            h5g_ej = f.create_group(f"ecc_{j}")
            h5g_ej['ecc'] = ecc
            dTime = h5g_ej.create_dataset("time", (lenData,), 'float')
            dTime[...] = EOBModes.time

            gradt = np.gradient(EOBModes.time)
            h22 = EOBModes.get_mode(2,2)
            h21 = EOBModes.get_mode(2,1)
            h33 = EOBModes.get_mode(3,3)
            h44 = EOBModes.get_mode(4,4)

            h22_absCum = np.cumsum(np.power(np.abs(h22.dot), 2) * gradt) / 8 / np.pi
            h21_absCum = np.cumsum(np.power(np.abs(h21.dot), 2) * gradt) / 8 / np.pi
            h33_absCum = np.cumsum(np.power(np.abs(h33.dot), 2) * gradt) / 8 / np.pi
            h44_absCum = np.cumsum(np.power(np.abs(h44.dot), 2) * gradt) / 8 / np.pi
            EnergyFluxCum = h22_absCum + h21_absCum + h33_absCum + h44_absCum

            dh21FluxCum = h5g_ej.create_dataset("h21FluxCum", (lenData,), 'float')
            dh21FluxCum[...] = h21_absCum
            dh22FluxCum = h5g_ej.create_dataset("h22FluxCum", (lenData,), 'float')
            dh22FluxCum[...] = h22_absCum
            dh33FluxCum = h5g_ej.create_dataset("h33FluxCum", (lenData,), 'float')
            dh33FluxCum[...] = h33_absCum
            dh44FluxCum = h5g_ej.create_dataset("h44FluxCum", (lenData,), 'float')
            dh44FluxCum[...] = h44_absCum
            dEnergyFluxCum = h5g_ej.create_dataset("EnergyFluxCum", (lenData,), 'float')
            dEnergyFluxCum[...] = EnergyFluxCum
            f.close()
