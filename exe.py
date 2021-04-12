#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:06:37 2019

@author: drizl
"""
import matplotlib as mlb
# mlb.use('Agg')

import numpy as np
from .SXS import SXSh22, save_namecol, DEFAULT_NOSPIN_SXS_LIST, DEFAULT_LOWSPIN_SXS_LIST, DEFAULT_HIGHSPIN_SXS_LIST
from .Utils import switch
from pathlib import Path
from optparse import OptionParser
from .psd import DetectorPSD
from .h22datatype import get_fmin, get_fini_dimless
import sys
import h5py
from .SXSlist import DEFAULT_ECC_ORBIT_DICT, DEFAULT_ECC_ORBIT_DICT_V5
from .MultiGrid import MultiGrid



#-----Used For MCMC Calibration----#
def getMCFlikelihood(argv):
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL

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
    parser.add_option('--max-step', type = 'int', default = 1000, help = 'Max iteration step')
    parser.add_option('--mode', type = 'str', default = 'all', help = 'Search mode.')

    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')

    parser.add_option('--max-k', type = 'float', help = 'Upper bound of parameters 1')
    parser.add_option('--min-k', type = 'float', help = 'Lower bound of parameters 1')

    parser.add_option('--max-dso', type = 'float', help = 'Upper bound of parameters 2')
    parser.add_option('--min-dso', type = 'float', help = 'Lower bound of parameters 2')

    parser.add_option('--max-dss', type = 'float', help = 'Upper bound of parameters 3')
    parser.add_option('--min-dss', type = 'float', help = 'Lower bound of parameters 3')

    parser.add_option('--max-dtpeak', type = 'float', help = 'Upper bound of parameters 4')
    parser.add_option('--min-dtpeak', type = 'float', help = 'Lower bound of parameters 4')

    parser.add_option('--max-eccentricity', type = 'float', help = 'Upper bound of parameters 5')
    parser.add_option('--min-eccentricity', type = 'float', help = 'Lower bound of parameters 5')
    parser.add_option('--delta-ecc', type = 'float',  help = 'Eccentricity range')

    parser.add_option('--gridsearch', action = 'store_true', help = 'Grid search, only for dt & ecc')
    parser.add_option('--grid-num-dtpeak', type = 'int', default = 100, help = 'numbers for grid search')
    parser.add_option('--grid-num-ecc', type = 'int', default = 100, help = 'numbers for grid search')
    parser.add_option('--seed', type = 'int', help = 'random seed')

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

    Smode = args.mode.lower()
    if args.gridsearch:
        Smode = 'gridsearch'

    psd = DetectorPSD(args.psd, flow = args.flow)
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
    dtPeak_default = pms0[3]
    ecc_default = NR.ecc
    if type(ecc_default) is str:
        try:
            ecc_default = float(ecc[1:])
        except:
            ecc_default = 0.5
    max_k = args.max_k if args.max_k is not None else 10
    min_k = args.min_k if args.min_k is not None else -10
    max_dss = args.max_dss if args.max_dss is not None else 1e3
    min_dss = args.min_dss if args.min_dss is not None else -1e3
    max_dso = args.max_dso if args.max_dso is not None else 1e4
    min_dso = args.min_dso if args.min_dso is not None else -1e4
    max_dtpeak = args.max_dtpeak if args.max_dtpeak is not None else 100
    min_dtpeak = args.min_dtpeak if args.min_dtpeak is not None else -10
    max_ecc = args.max_eccentricity if args.max_eccentricity is not None else 0.7
    min_ecc = args.min_eccentricity if args.min_eccentricity is not None else 0
    for case in switch(Smode):
        if case('nospin_wind'):
            NR = SXSh22(SXSnum = SXSnum,
                        f_ini = fini,
                        Mtotal = mtotal,
                        srate = srate,
                        srcloc = srcloc,
                        table = table,
                        srcloc_all = srcloc_all)
            ge = NR.construct_generator(approx, exe, psd = psd)
            pms_init = (100, 25)
            def get_lnprob(pms):
                if pms[0] < 0 or pms[0] > 1000 or pms[1] < 1 or pms[1] > 100:
                    return -np.inf
                ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, windt = pms[0], windw = pms[1],
                            KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtPeak_default, ecc = 0.0)
                return ret[0]
            break
        if case('nospin_ecc_wind'):
            # windt (0, 1000), windw (1, 100)
            if SXSnum in DEFAULT_ECC_ORBIT_DICT_V5:
                f0, e0 = DEFAULT_ECC_ORBIT_DICT_V5[SXSnum]
                NR = SXSh22(SXSnum = SXSnum,
                            f_ini = f0,
                            Mtotal = mtotal,
                            srate = srate,
                            srcloc = srcloc,
                            table = table,
                            srcloc_all = srcloc_all)
                ge = NR.construct_generator(approx, exe, psd = psd)
                if args.delta_ecc is None:
                    eB = 40 * np.log(2)
                    chiE = 0.1 + 0.4 * np.exp(-eB * np.abs(e0))
                    e_minA = np.abs(e0) * (1-chiE)
                    e_maxA = np.abs(e0) * (1+chiE)
                    if e0<0:
                        min_ecc_x = -e_maxA
                        max_ecc_x = -e_minA
                    else:
                        min_ecc_x = e_minA
                        max_ecc_x = e_maxA
                else:
                    min_ecc_x = e0 - args.delta_ecc
                    max_ecc_x = e0 + args.delta_ecc
                max_ecc = args.max_eccentricity if args.max_eccentricity is not None else max_ecc_x
                min_ecc = args.min_eccentricity if args.min_eccentricity is not None else min_ecc_x
                if e0 > 0:
                    min_ecc = max(min_ecc, 0)
                else:
                    max_ecc = min(max_ecc, 0)
                if e0 < min_ecc or e0 > max_ecc:
                    e0 = (max_ecc + min_ecc) / 2
                pms_init = (e0, 100, 25)

                def get_lnprob(pms):
                    if pms[0] < min_ecc or pms[0] > max_ecc or pms[1] < 0 or pms[1] > 1000 or pms[2] < 1 or pms[2] > 100:
                        return -np.inf
                    ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout, windt = pms[1], windw = pms[2],
                                KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtPeak_default, ecc = pms[0])
                    return ret[0]
            break
        if case('nospin'):
            pms_init = (KK_default, dtPeak_default)
            # K, dtPeak
            def get_lnprob(pms):
                if pms[0] < min_k or pms[0] > max_k or pms[1] < min_dtpeak or pms[1] > max_dtpeak:
                    return -np.inf
                ret = ge.get_overlap(jobtag = args.jobtag, timeout = args.timeout,
                            KK = pms[0], dSO = dSO_default, dSS = dSS_default, dtPeak = pms[1])
                eps = 1 - ret.max_FF
                if eps > 1:
                    return -np.inf
                dephase = ret.dephase_fit
                return -( pow(eps/0.01, 2) + pow(dephase/5, 2) )/2
            break
        if case('deltaphase'):
            pms_init = pms0
            def get_lnprob(pms):
                if pms[0] < min_k or pms[0] > max_k or \
                    pms[1] < min_dso or pms[1] > max_dso or \
                    pms[2] < min_dss or pms[2] > max_dss or \
                    pms[3] < min_dtpeak or pms[3] > max_dtpeak:
                    return -np.inf
                ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout,
                            KK = pms[0], dSO = pms[1], dSS = pms[2], dtPeak = pms[3])
                return ret[0]
            break
        if case('deltaphase_nodt'):
            pms_init = (KK_default, dSO_default, dSS_default)
            def get_lnprob(pms):
                if pms[0] < min_k or pms[0] > max_k or \
                    pms[1] < min_dso or pms[1] > max_dso or \
                    pms[2] < min_dss or pms[2] > max_dss:
                    return -np.inf
                ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout,
                            KK = pms[0], dSO = pms[1], dSS = pms[2], dtPeak = 0)
                return ret[0]
            break

        if case('deltaphase_nospin'):
            pms_init = (KK_default, dtPeak_default)
            def get_lnprob(pms):
                if pms[0] < min_k or pms[0] > max_k or pms[1] < min_dtpeak or pms[1] > max_dtpeak:
                    return -np.inf
                ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout,
                            KK = pms[0], dSO = dSO_default, dSS = dSS_default, dtPeak = pms[1])
                return ret[0]
            break

        if case('deltaphase_nospin_orbecc'):
            if SXSnum in DEFAULT_ECC_ORBIT_DICT:
                f0, e0 = DEFAULT_ECC_ORBIT_DICT[SXSnum]
                NR = SXSh22(SXSnum = SXSnum,
                            f_ini = f0,
                            Mtotal = mtotal,
                            srate = srate,
                            srcloc = srcloc,
                            table = table,
                            srcloc_all = srcloc_all)
                ge = NR.construct_generator(approx, exe, psd = psd)
                
                if args.delta_ecc is None:
                    eB = 40 * np.log(2)
                    chiE = 0.1 + 0.4 * np.exp(-eB * np.abs(e0))
                    e_minA = np.abs(e0) * (1-chiE)
                    e_maxA = np.abs(e0) * (1+chiE)
                    if e0<0:
                        min_ecc_x = -e_maxA
                        max_ecc_x = -e_minA
                    else:
                        min_ecc_x = e_minA
                        max_ecc_x = e_maxA
                else:
                    min_ecc_x = e0 - args.delta_ecc
                    max_ecc_x = e0 + args.delta_ecc
                max_ecc = args.max_eccentricity if args.max_eccentricity is not None else max_ecc_x
                min_ecc = args.min_eccentricity if args.min_eccentricity is not None else min_ecc_x
                if e0 > 0:
                    min_ecc = max(min_ecc, 0)
                else:
                    max_ecc = min(max_ecc, 0)
                if e0 < min_ecc or e0 > max_ecc:
                    e0 = (max_ecc + min_ecc) / 2
                if dtPeak_default < min_dtpeak or dtPeak_default > max_dtpeak:
                    dtPeak_default = (min_dtpeak + max_dtpeak) / 2
                if KK_default < min_k or KK_default > max_k:
                    KK_default = (min_k + max_k) / 2
                pms_init = (KK_default, dtPeak_default, e0)
                def get_lnprob(pms):
                    if pms[0] < min_k or pms[0] > max_k or pms[1] < min_dtpeak or pms[1] > max_dtpeak or pms[2] < min_ecc or pms[2] > max_ecc:
                        return -np.inf
                    ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout,
                                KK = pms[0], dSO = dSO_default, dSS = dSS_default, dtPeak = pms[1], ecc = pms[2])
                    return ret[0]


            else:
                raise Exception(f'No such case {SXSnum}')
            break

        if case('deltaphase_nospin_withecc'):
            pms_init = (KK_default, dtPeak_default, ecc_default)

            def get_lnprob(pms):
                if pms[0] < min_k or pms[0] > max_k or pms[1] < min_dtpeak or pms[1] > max_dtpeak or pms[2] < min_ecc or pms[2] > max_ecc:
                    return -np.inf
                ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout,
                            KK = pms[0], dSO = dSO_default, dSS = dSS_default, dtPeak = pms[1], ecc = pms[2])
                return ret[0]
            break
        if case('deltaphase_nospin_adjdt'):
            pms_dt_init = [ 4.93803970e+02, -1.11676765e+04,  1.01656992e+05, -4.03487263e+05, 5.83501606e+05]
            dt_init = 0
            for i in range(len(pms_dt_init)):
                dt_init += pms_dt_init[i]*np.power(NR.eta, i)
            pms_init = [dtPeak_default]
            def get_lnprob(pms):
                if pms[0] < min_dtpeak or pms[0] > max_dtpeak:
                    return -np.inf
                ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout,
                            KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = pms[0])
                return ret[0]
            break

        if case('deltaphase_nospin_adjdt_withecc'):
            pms_dt_init = [ 4.93803970e+02, -1.11676765e+04,  1.01656992e+05, -4.03487263e+05, 5.83501606e+05]
            dt_init = 0
            for i in range(len(pms_dt_init)):
                dt_init += pms_dt_init[i]*np.power(NR.eta, i)
            if SXSnum in DEFAULT_ECC_ORBIT_DICT:
                f0, e0 = DEFAULT_ECC_ORBIT_DICT[SXSnum]
                NR = SXSh22(SXSnum = SXSnum,
                            f_ini = f0,
                            Mtotal = mtotal,
                            srate = srate,
                            srcloc = srcloc,
                            table = table,
                            srcloc_all = srcloc_all)
                ge = NR.construct_generator(approx, exe, psd = psd)
                pms_init = (dt_init, e0)
                if args.delta_ecc is None:
                    eB = 40 * np.log(2)
                    chiE = 0.1 + 0.4 * np.exp(-eB * np.abs(e0))
                    e_minA = np.abs(e0) * (1-chiE)
                    e_maxA = np.abs(e0) * (1+chiE)
                    if e0<0:
                        min_ecc_x = -e_maxA
                        max_ecc_x = -e_minA
                    else:
                        min_ecc_x = e_minA
                        max_ecc_x = e_maxA
                else:
                    min_ecc_x = e0 - args.delta_ecc
                    max_ecc_x = e0 + args.delta_ecc
                max_ecc = args.max_eccentricity if args.max_eccentricity is not None else max_ecc_x
                min_ecc = args.min_eccentricity if args.min_eccentricity is not None else min_ecc_x
                if e0 > 0:
                    min_ecc = max(min_ecc, 0)
                else:
                    max_ecc = min(max_ecc, 0)
                if e0 < min_ecc or e0 > max_ecc:
                    e0 = (max_ecc + min_ecc) / 2
                if dtPeak_default < min_dtpeak or dtPeak_default > max_dtpeak:
                    dtPeak_default = (min_dtpeak + max_dtpeak) / 2
                pms_init = (dtPeak_default, e0)

                def get_lnprob(pms):
                    if pms[0] < min_dtpeak or pms[0] > max_dtpeak or pms[1] < min_ecc or pms[1] > max_ecc:
                        return -np.inf
                    ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout,
                                KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = pms[0], ecc = pms[1])
                    return ret[0]
            else:
                raise Exception(f'No such case {SXSnum}')
            break


        else:
            pms_init = pms0
            # K, dSO, dSS, dtPeak
            def get_lnprob(pms):
                if pms[0] < min_k or pms[0] > max_k or \
                    pms[1] < min_dso or pms[1] > max_dso or \
                    pms[2] < min_dss or pms[2] > max_dss or \
                    pms[3] < min_dtpeak or pms[3] > max_dtpeak:
                    return -np.inf
                ret = ge.get_overlap(jobtag = args.jobtag, timeout = args.timeout,
                            KK = pms[0], dSO = pms[1], dSS = pms[2], dtPeak = pms[3])
                eps = 1 - ret.max_FF
                if eps > 1:
                    return -np.inf
                dephase = ret.dephase_fit
                return -( pow(eps/0.01, 2) + pow(dephase/5, 2) )/2
            break
    def get_waveform(KK = KK_default, dSO = dSO_default, dSS = dSS_default, dtPeak = dtPeak_default, ecc = ecc_default, **kwargs):
        wf = ge.get_waveform(jobtag = args.jobtag, timeout = args.timeout,
                        KK = KK, dSO = dSO, dSS = dSS, dtPeak = dtPeak, ecc = ecc, **kwargs)
        ret = ge.get_lnprob(jobtag = args.jobtag, timeout = args.timeout,
                    KK = KK, dSO = dSO, dSS = dSS, dtPeak = dtPeak, ecc = ecc)
        return wf, ret
    return get_lnprob, args, pms_init, get_waveform

    





#-----Parse args-----#
def parseargs(argv):
    # Input Parameters:
    # --executable: exe file path
    # --jobtag: job tag
    # --fini: Initial orbital frequency
    # --approx: Code version
    # --SXS: Template for comparision
    # --prefix: dir for saving
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL
    parser = OptionParser(description='Waveform Comparation With SXS')
    parser.add_option('--executable', type = 'str', default = 'lalsim-inspiral', help = 'Exe command')
    parser.add_option('--jobtag', type = 'str', default = '_test', help = 'Jobtag for the code run')
    parser.add_option('--approx', type = 'str', default = 'SEOBNRv1', help = 'Version of the code')
    parser.add_option('--fini', type = 'float', default = 0, help = 'Initial orbital frequency')
    parser.add_option('--fini-scan', action = 'store_true', help = 'If added, will scan init freq.')
    parser.add_option('--SXS', type = 'str', action = 'append', default = [], help = 'SXS template for comparision')
    parser.add_option('--SXS-nospin', action = 'store_true', help = 'will use no spin SXS wfs')
    parser.add_option('--SXS-lowspin', action = 'store_true', help = 'will use low spin SXS wfs')
    parser.add_option('--SXS-highspin', action = 'store_true', help = 'will use low spin SXS wfs')
    parser.add_option('--min-mtotal', type = 'float', default = 10, help = 'Min Total mass')
    parser.add_option('--max-mtotal', type = 'float', default = 200, help = 'Max Total mass')
    parser.add_option('--num-mtotal', type = 'int', default = 100, help = 'Number of cases')

    parser.add_option('--prefix', type = 'str', default = '.', help = 'dir for results saving.')
    parser.add_option('--plot', action = 'store_true', help = 'If added, will plot sa results and waveform.')
    parser.add_option('--verbose', action = 'store_true', help = 'If added, will print verbose message.')
    parser.add_option('--hertz', action = 'store_true', help = 'If added, will use dimension Hz for fini.')
    parser.add_option('--preset-ecc', action = 'store_true', help = 'If added, will use preset Hz eccentricities.')
    parser.add_option('--maxecc', type = 'float', default = 0, help = 'If zero, will automatically set ecc search range.')
    parser.add_option('--minecc', type = 'float', default = 0, help = 'If zero, will automatically set ecc search range.')
    parser.add_option('--estep', type = 'float', default = 0.02, help = 'If zero, will automatically set ecc search range.')

    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')
    parser.add_option('--modeL', type = 'int', help = 'mode L for HM')
    parser.add_option('--modeM', type = 'int', help = 'mode M for HM')
    parser.add_option('--psd', type = 'str', help = 'Detector psd.')
    parser.add_option('--scan-mtotal', action = 'store_true', help = 'Scan Mtotal')
    parser.add_option('--flow', type = 'float', default = 0, help = 'Lower frequency cut off for psd.')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')

    args = parser.parse_args(argv)
    return args


def main(argv = None):
    args, _ = parseargs(argv)
    SXSnum_list = args.SXS
    if len(SXSnum_list) == 0:
        SXSnum_list.append('0001')
    if args.SXS_nospin:
        SXSnum_list = DEFAULT_NOSPIN_SXS_LIST
    if args.SXS_lowspin:
        SXSnum_list = DEFAULT_LOWSPIN_SXS_LIST
    if args.SXS_highspin:
        SXSnum_list = DEFAULT_HIGHSPIN_SXS_LIST
    Mtotal_min = args.min_mtotal
    Mtotal_max = args.max_mtotal
    Mtotal_num = args.num_mtotal
    mtotal_list = np.linspace(Mtotal_min, Mtotal_max, Mtotal_num)
    if len(mtotal_list) == 0:
        mtotal_list.append(40)
    approx = args.approx
    jobtag = args.jobtag
    exe = args.executable
    prefix = Path(args.prefix)
    isplot = args.plot
    fini = args.fini
    ishertz = args.hertz
    maxecc = args.maxecc
    minecc = args.minecc
    table = args.table
    srcloc = args.srcloc
    srcloc_all = args.srcloc_all
    psd = DetectorPSD(args.psd, flow = args.flow)
    timeout = args.timeout
    Preset = args.preset_ecc
    estep = args.estep

    modeL = args.modeL
    modeM = args.modeM
        
    savedir = prefix / approx
    verbose = args.verbose
    if verbose is None:
        verbose = False
    # Mkdir for data saving
    if not savedir.exists():
        savedir.mkdir(parents=True)
    
    
    if args.psd is None:
        # Setting ErrorMsg filename.
        ferrmsg = savedir / f'errMessageLog_{jobtag}.txt'
        errmsg = []
        # Setting Results savimg filename.
        fresults = savedir / f'results_{jobtag}.csv'

        save_namecol(fresults)
        
        for SXSnum in SXSnum_list:
            # Setting saveing prefix
            Sprefix = savedir / SXSnum
            if not Sprefix.exists():
                Sprefix.mkdir()
            
            s = SXSh22(SXSnum, f_ini = fini, 
                    modeL = modeL,
                    modeM = modeM, 
                    table = table,
                    srcloc = srcloc,
                    srcloc_all = srcloc_all,
                    verbose = verbose, 
                    ishertz = ishertz)
            ge = s.construct_generator(approx, exe, psd = psd)
            ret = ge.get_overlap(jobtag = jobtag, minecc = minecc, maxecc = maxecc, 
                                timeout = timeout, verbose = verbose, Preset = Preset, estep = estep)
            

            if isplot:
                # Setting saving file prefix
                fig_SA_scan = Sprefix / 'fig_SA_scan.png'
                fig_waveform = Sprefix / 'fig_waveform_fit.png'
                ret.plot_results(fig_SA_scan)
                ret.plot_waveform_fit(fig_waveform)
                
            # Setting saving files
            file_SA_scan = Sprefix / 'SA_scan.txt'
            file_waveform = Sprefix / 'waveform_fit.txt'
            
            ret.save_fit(fresults)
            ret.save_results(file_SA_scan)
            ret.save_waveform_fit(file_waveform)
            errmsg.append(ret.ErrorMsg)
        
        np.savetxt(ferrmsg, np.array([errmsg]), fmt = '%s', delimiter = '\n')
    elif args.scan_mtotal:
        # Setting ErrorMsg filename.
        ferrmsg = savedir / f'errMessageLog_{jobtag}.txt'
        errmsg = []
        # Setting Results savimg filename.
        fresults = savedir / f'results_{jobtag}.csv'

        save_namecol(fresults)
        
        for SXSnum in SXSnum_list:
            # Setting saveing prefix
            Sprefix = savedir / SXSnum
            if not Sprefix.exists():
                Sprefix.mkdir()
            
            s = SXSh22(SXSnum, f_ini = fini, 
                    modeL = modeL,
                    modeM = modeM, 
                    table = table,
                    srcloc = srcloc,
                    srcloc_all = srcloc_all,
                    verbose = verbose, 
                    ishertz = ishertz)
            ge = s.construct_generator(approx, exe, psd = psd)
            ret = ge.get_overlap(jobtag = jobtag, minecc = minecc, maxecc = maxecc, scan_mtotal = True,
                                timeout = timeout, verbose = verbose, Preset = Preset, estep = estep)
            

            if isplot:
                # Setting saving file prefix
                fig_SA_scan = Sprefix / 'fig_SA_scan.png'
                fig_waveform = Sprefix / 'fig_waveform_fit.png'
                ret.plot_results(fig_SA_scan)
                ret.plot_waveform_fit(fig_waveform)
                
            # Setting saving files
            file_SA_scan = Sprefix / 'SA_scan.txt'
            file_waveform = Sprefix / 'waveform_fit.txt'
            
            ret.save_fit(fresults)
            ret.save_results(file_SA_scan)
            ret.save_waveform_fit(file_waveform)
            errmsg.append(ret.ErrorMsg)
        
        np.savetxt(ferrmsg, np.array([errmsg]), fmt = '%s', delimiter = '\n')

    else:
        for SXSnum in SXSnum_list:
            Sprefix = savedir / SXSnum
            if not Sprefix.exists():
                Sprefix.mkdir()
            fresults = Sprefix / f'results_{jobtag}.csv'

            s = SXSh22(SXSnum, f_ini = fini, 
                    modeL = modeL,
                    modeM = modeM, 
                    table = table,
                    srcloc = srcloc,
                    srcloc_all = srcloc_all,
                    verbose = verbose, 
                    ishertz = ishertz)
            ge = s.construct_generator(approx, exe, psd = psd)
            dprefix = [['#q', '#chi1', '#chi2', '#Mtotal', '#FF', f'#ecc={s.ecc}']]
            save_namecol(fresults, dprefix)
            for mtotal in mtotal_list:
                ret = ge.get_overlap(jobtag = jobtag, minecc = minecc, maxecc = maxecc, Mtotal = mtotal,
                                    timeout = timeout, verbose = verbose, Preset = Preset, estep = estep)
                data = [[s.q, s.s1z, s.s2z, mtotal, ret.max_FF, ret.ecc_fit]]
                add_csv(fresults, data)

    return 0


def parseargs_compWithFreqCut(argv):
    # Input Parameters:
    # --executable: exe file path
    # --jobtag: job tag
    # --fini: Initial orbital frequency
    # --approx: Code version
    # --SXS: Template for comparision
    # --prefix: dir for saving
    from .SXS import DEFAULT_TABLE
    from .SXS import DEFAULT_SRCLOC
    from .SXS import DEFAULT_SRCLOC_ALL
    parser = OptionParser(description='Waveform Comparation With SXS')
    parser.add_option('--executable', type = 'str', default = 'lalsim-inspiral', help = 'Exe command')
    parser.add_option('--jobtag', type = 'str', default = '_test', help = 'Jobtag for the code run')
    parser.add_option('--approx', type = 'str', default = 'SEOBNRv1', help = 'Version of the code')
    parser.add_option('--fini', type = 'float', default = 0, help = 'Initial orbital frequency')
    parser.add_option('--fini-si', type = 'float', help = 'SI initial orbital frequency')
    parser.add_option('--SXS', type = 'str', action = 'append', default = [], help = 'SXS template for comparision')
    parser.add_option('--allow-ecc', action = 'store_true', help = 'Would use default NR ecc, if nan, will use 0.')
    parser.add_option('--allow-ecc-pass0', action = 'store_true', help = 'Would use default NR ecc, if nan, will skip')
    parser.add_option('--allow-ecc-fit', action = 'store_true', help = 'Would find best fit ecc.')
    parser.add_option('--scan-mtotal', action = 'store_true', help = 'Scan Mtotal')
    parser.add_option('--allow-ecc-pn', action = 'store_true', help = 'Would solve correspond ecc by PN')
    parser.add_option('--allow-ecc-resp', action = 'store_true', help = 'Would fit respectively.')
    parser.add_option('--min-mtotal', type = 'float', default = 10, help = 'Min Total mass')
    parser.add_option('--max-mtotal', type = 'float', default = 200, help = 'Max Total mass')
    parser.add_option('--logscale-mtotal', action = 'store_true', help = 'Use log scale')
    parser.add_option('--estep', type = 'float', default = 0.001, help = 'e step')
    parser.add_option('--eccentricity', type = 'float', help = 'Will use this eccentricity')
    parser.add_option('--num-mtotal', type = 'int', default = 100, help = 'Number of cases')
    parser.add_option('--prefix', type = 'str', default = '.', help = 'dir for results saving.')
    parser.add_option('--verbose', action = 'store_true', help = 'If added, will print verbose message.')
    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')
    parser.add_option('--psd', type = 'str', help = 'Detector psd.')
    parser.add_option('--flow', type = 'float', default = 0, help = 'Lower frequency cut off for psd.')
    parser.add_option('--fhigh', type = 'float', help = 'Higher frequency cut off for psd.')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')

    parser.add_option('--maxecc', type = 'float', default = 0, help = 'When fit')
    args = parser.parse_args(argv)
    return args

from scipy.optimize import root
def aPN(e):
    return np.power(e, 12/19) * np.power(1 + 121 * e * e / 304, 870/2299) / (1-e*e)

def ProduceEccSolver(f, eNR, fNR):
    def solver(e):
        return aPN(e) - aPN(eNR) * np.power(fNR/f, 3/2)
    return solver

def solveEcc(eNR, fNR, f):
    fsr = ProduceEccSolver(f, eNR, fNR)
    ans = root(fsr, 0).x
    return ans[0]

from WTestLib.SXS import preset_ecc
def compWithFreqCut(argv = None):
    args, _ = parseargs_compWithFreqCut(argv)

    approx = args.approx
    exe = args.executable
    prefix = Path(args.prefix)
    fini = args.fini
    fini_si = args.fini_si
    SXSnum_list = args.SXS

    Mtotal_min = args.min_mtotal
    Mtotal_max = args.max_mtotal
    Mtotal_num = args.num_mtotal
    if args.logscale_mtotal:
        Mtotal_list = np.logspace(np.log10(Mtotal_min), np.log10(Mtotal_max), Mtotal_num)
    else:
        Mtotal_list = np.linspace(Mtotal_min, Mtotal_max, Mtotal_num)
    if len(SXSnum_list) == 0:
        SXSnum_list.append('0001')

    psd = DetectorPSD(args.psd, flow = args.flow, fhigh = args.fhigh)
    allow_ecc = args.allow_ecc
    ecc_skipNAN = False
    allow_ecc_resp = args.allow_ecc_resp
    allow_ecc_fit = args.allow_ecc_fit
    allow_ecc_pn = args.allow_ecc_pn
    allow_ecc_pass0 = args.allow_ecc_pass0
    ecc_pre = args.eccentricity
    if allow_ecc_pass0:
        allow_ecc = True
        ecc_skipNAN = True
    timeout = args.timeout
    jobtag = args.jobtag
    
    table = args.table
    srcloc = args.srcloc
    srcloc_all = args.srcloc_all
    verbose = args.verbose

    savedir = prefix / approx
    verbose = args.verbose
    onePSD = DetectorPSD(None)
    if args.maxecc > 0:
        epret = False
    else:
        epret = True

    # Mkdir for data saving
    if not savedir.exists():
        savedir.mkdir(parents=True)
    # Setting ErrorMsg filename.
    if ecc_pre is not None:
        for SXSnum in SXSnum_list:        
            s = SXSh22(SXSnum, f_ini = fini, 
                    modeL = None,
                    modeM = None, 
                    table = table,
                    srcloc = srcloc,
                    srcloc_all = srcloc_all,
                    verbose = verbose, 
                    ishertz = False)
            ge = s.construct_generator(approx, exe, psd = psd)

            # Setting saveing prefix
            fresults = savedir / f'results_{SXSnum}_{jobtag}.csv'
            # Setting Results savimg filename.
            save_namecol(fresults, data = [['#q', '#chi1', '#chi2', '#Mtotal', '#FF', f'#ecc={ecc_pre}']])
            ret = ge.get_overlap(jobtag = jobtag, minecc = 0, maxecc = 0, ecc = ecc_pre,
                                timeout = timeout, verbose = verbose, Mtotal = Mtotal_list)
            length = len(Mtotal_list)
            q_list = s.q*np.ones(len(Mtotal_list)).reshape(1,length)
            s1z_list = s.s1z*np.ones(len(Mtotal_list)).reshape(1,length)
            s2z_list = s.s2z*np.ones(len(Mtotal_list)).reshape(1,length)
            FF_list = ret[2].reshape(1,length)
            Mtotal_list_out = Mtotal_list.reshape(1, length)
            data = np.concatenate((q_list, s1z_list, s2z_list, Mtotal_list_out, FF_list), axis = 0)
            add_csv(fresults, data.T.tolist())

    elif allow_ecc_pn:
        for SXSnum in SXSnum_list:        
            s = SXSh22(SXSnum, f_ini = fini, 
                    modeL = None,
                    modeM = None, 
                    table = table,
                    srcloc = srcloc,
                    srcloc_all = srcloc_all,
                    verbose = verbose, 
                    ishertz = False)
            ge = s.construct_generator(approx, exe, psd = psd)
            ecc = s.ecc
            if type(ecc) is str:
                continue

            # Setting saveing prefix
            fresults = savedir / f'results_{SXSnum}_{jobtag}.csv'
            # Setting Results savimg filename.
            save_namecol(fresults, data = [['#q', '#chi1', '#chi2', '#Mtotal', '#FF', f'#ecc={ecc}', f'#fini = {s.f_ini_dimless}(f0 = {fini_si}Hz)']])

            ecc_list = []
            FF_list = []
            fini_list = []
            for Mtotal in Mtotal_list:
                fini_need = get_fini_dimless(fini_si, Mtotal)
                e0 = solveEcc(ecc, s.f_ini_dimless, fini_need)
                ret = ge.get_overlap(jobtag = jobtag, minecc = 0, maxecc = 0, eccentricity = e0,
                                    timeout = timeout, verbose = verbose, Mtotal = Mtotal, fini = fini_si)
                ecc_list.append(e0)
                FF_list.append(ret.max_FF)
                fini_list.append(fini_need)
            length = len(Mtotal_list)
            q_list = s.q*np.ones(len(Mtotal_list)).reshape(1,length)
            s1z_list = s.s1z*np.ones(len(Mtotal_list)).reshape(1,length)
            s2z_list = s.s2z*np.ones(len(Mtotal_list)).reshape(1,length)
            FF_list = np.array(FF_list).reshape(1,length)
            ecc_list = np.array(ecc_list).reshape(1,length)
            Mtotal_list_out = Mtotal_list.reshape(1, length)
            fini_list = np.array(fini_list).reshape(1, length)
            data = np.concatenate((q_list, s1z_list, s2z_list, Mtotal_list_out, FF_list, ecc_list, fini_list), axis = 0)
            add_csv(fresults, data.T.tolist())
    elif allow_ecc_resp:
        for SXSnum in SXSnum_list:
            s = SXSh22(SXSnum, f_ini = fini, 
                    modeL = None,
                    modeM = None, 
                    table = table,
                    srcloc = srcloc,
                    srcloc_all = srcloc_all,
                    verbose = verbose, 
                    ishertz = False)
            ge = s.construct_generator(approx, exe, psd = psd)
            # Setting saveing prefix
            fresults = savedir / f'results_{SXSnum}_{jobtag}.csv'
            # Setting Results savimg filename.
            save_namecol(fresults, data = [['#q', '#chi1', '#chi2', '#Mtotal', '#FF', f'#ecc={s.ecc}', f'#fini']])

            for Mtotal in Mtotal_list:
                ret_fit = ge.get_overlap(jobtag = jobtag, minecc = 0, maxecc = 0, Mtotal = Mtotal,
                                    timeout = timeout, verbose = verbose, Preset = True)
                e0 = ret_fit.ecc_fit
                FF = ret_fit.max_FF
                data = [[s.q, s.s1z, s.s2z, Mtotal, FF, e0, fini]]
                add_csv(fresults, data)
    else:
        for SXSnum in SXSnum_list:        
            s = SXSh22(SXSnum, f_ini = fini, 
                    modeL = None,
                    modeM = None, 
                    table = table,
                    srcloc = srcloc,
                    srcloc_all = srcloc_all,
                    verbose = verbose, 
                    ishertz = False)
            ge = s.construct_generator(approx, exe, psd = psd)

            if allow_ecc_fit and fini == 0.002:
                if args.scan_mtotal:
                    ge_fit = s.construct_generator(approx, exe, psd = psd)
                else:
                    ge_fit = s.construct_generator(approx, exe, psd = onePSD)
                ret_fit = ge_fit.get_overlap(jobtag = jobtag, minecc = 0, maxecc = args.maxecc, scan_mtotal = args.scan_mtotal,
                                    timeout = timeout, verbose = verbose, Preset = epret, estep = args.estep)
                e0 = ret_fit.ecc_fit
            else:
                if fini == 0:
                    ecc = s.ecc
                    if allow_ecc:
                        if type(ecc) is str:
                            if ecc_skipNAN:
                                continue
                            else:
                                ecc = 0
                        e0 = ecc
                    else:
                        e0 = 0
                    print(f'ecc = {e0}')
                elif fini == 0.002:
                    if allow_ecc:
                        ecc = preset_ecc(fini, retMid = True)
                        if ecc is None and ecc_skipNAN:
                            continue
                        else:
                            ecc = 0
                    else:
                        ecc = 0
                    e0 = ecc
                else:
                    if allow_ecc:
                        continue
                    else:
                        e0 = 0
                
            # Setting saveing prefix
            fresults = savedir / f'results_{SXSnum}_{jobtag}.csv'
            # Setting Results savimg filename.
            save_namecol(fresults, data = [['#q', '#chi1', '#chi2', '#Mtotal', '#FF', f'#ecc={e0}']])
            ret = ge.get_overlap(jobtag = jobtag, minecc = 0, maxecc = 0, ecc = e0,
                                timeout = timeout, verbose = verbose, Mtotal = Mtotal_list)
            length = len(Mtotal_list)
            q_list = s.q*np.ones(len(Mtotal_list)).reshape(1,length)
            s1z_list = s.s1z*np.ones(len(Mtotal_list)).reshape(1,length)
            s2z_list = s.s2z*np.ones(len(Mtotal_list)).reshape(1,length)
            FF_list = ret[2].reshape(1,length)
            Mtotal_list_out = Mtotal_list.reshape(1, length)
            data = np.concatenate((q_list, s1z_list, s2z_list, Mtotal_list_out, FF_list), axis = 0)
            add_csv(fresults, data.T.tolist())
        
    return 0
        

#-------------Resave------------#
from .SXS import resave_results, add_csv, save_namecol
def resave_main(argv=None):
    parser = OptionParser(description='Resave results.')
    parser.add_option('--source', type = 'str', default = '.', help = 'Path for row data.')
    parser.add_option('--prefix', type = 'str', default = 'results', help = 'Prefix for data saveing.')
    parser.add_option('--dataprefix', type = 'str', default = 'results', help = 'csv src file prefix.')
    args,empty = parser.parse_args(argv)
    source = Path(args.source)
    prefix = args.prefix
    datapref = args.dataprefix
    for ddir in source.iterdir():
        fsave = Path(f'{prefix}_{ddir.name}.csv')
        if ddir.is_dir():
            resave_results(str(ddir / datapref), fsave)

#------------Model Comp----------#
from .generator import CompGenerator
def modcomp(argv = None):
    parser = OptionParser(description='General model compare.')
    parser.add_option('--approx1', type = 'str', default = 'SEOBNRv1', help = 'approx1 for compare')
    parser.add_option('--approx2', type = 'str', default = 'SEOBNRv4', help = 'approx2 for compare')
    parser.add_option('--executable1', type = 'str', default = 'lalsim-inspiral', help = 'command for waveform generation')
    parser.add_option('--executable2', type = 'str', default = 'lalsim-inspiral', help = 'command for waveform generation')
    parser.add_option('--prefix', type = 'str', default = '.', help = 'prefix for data save')
    parser.add_option('--verbose', action = 'store_true', help = 'If added, will print verbose message.')
    parser.add_option('--mratio', type = 'float', action = 'append', default = [], help = 'Mass ratio')
    parser.add_option('--spin1z', type = 'float', action = 'append', default = [], help = 'Spin1z')
    parser.add_option('--spin2z', type = 'float', action = 'append', default = [], help = 'Spin2z')
    parser.add_option('--mtotal', type = 'float', default = 16, help = 'Total mass of the binary system')
    parser.add_option('--ecc', type = 'float', action = 'append', default = [], help = 'eccentricity')
    parser.add_option('--fini', type = 'float', default = 40, help = 'Initial orbit frequency')
    parser.add_option('--natural', action = 'store_true', help = 'If added, will use natural dimension for fini.')
    parser.add_option('--distance', type = 'float', default = 100, help = 'BBH distance in Mpc')
    parser.add_option('--srate', type = 'float', default = 16384, help = 'Sample rate')
    parser.add_option('--jobtag', type = 'str', default = '_wfcomp', help = 'Tag for this run')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')
    parser.add_option('--psd', type = 'str', help = 'Detector psd.')
    parser.add_option('--flow', type = 'float', default = 0, help = 'Lower frequency cut off for psd.')

    # Random mode
    parser.add_option('--random', action = 'store_true', help = 'If added, will use random parameters.')
    parser.add_option('--min-mratio', type = 'float', default = 1, help = 'Used in random mode [1]')
    parser.add_option('--max-mratio', type = 'float', default = 9, help = 'Used in random mode [9]')
    parser.add_option('--min-spin1z', type = 'float',  help = 'Used in random mode')
    parser.add_option('--max-spin1z', type = 'float',  help = 'Used in random mode')
    parser.add_option('--min-spin2z', type = 'float',  help = 'Used in random mode')
    parser.add_option('--max-spin2z', type = 'float',  help = 'Used in random mode')
    parser.add_option('--min-ecc', type = 'float',  help = 'Used in random mode')
    parser.add_option('--max-ecc', type = 'float',  help = 'Used in random mode')
    parser.add_option('--ncompare', type = 'int', default = 10, help = 'Used in random mode [10]')


    args, _empty = parser.parse_args(argv)
    approx1 = args.approx1
    approx2 = args.approx2
    exe1 = args.executable1
    exe2 = args.executable2
    verbose = args.verbose
    
    q = args.mratio
    if len(q) == 0:
        q = [1.]
    s1z = args.spin1z
    if len(s1z) == 0:
        s1z = [0.]
    s2z = args.spin2z
    if len(s2z) == 0:
        s2z = [0.]
    ecc = args.ecc
    if len(ecc) == 0:
        ecc = [0.]
    
    Mtotal = args.mtotal
    D = args.distance
    f_ini = args.fini
    if args.natural:
        f_ini = get_fmin(f_ini, Mtotal)
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
    namecol = [['#mass ratio',
               '#spin1z',
               '#spin2z',
               '#ecc',
               '#FF']]
    fsave = savedir / fname
    save_namecol(fsave, data = namecol)
    Comp = CompGenerator(approx1, exe1, approx2, exe2, psd = psd, verbose = verbose)

    if args.random:
        Comp.compare_random(args.min_mratio, args.max_mratio, 
                                args.min_spin1z, args.max_spin1z, 
                                args.min_spin2z, args.max_spin2z, 
                                args.min_ecc, args.max_ecc, fsave = fsave,
                                Num = args.ncompare, 
                                Mtotal = Mtotal, 
                                D = D, f_ini = f_ini, 
                                srate = srate, jobtag = jobtag)
        return 0
    else:
        ret = Comp.compare(q, s1z, s2z, ecc, Mtotal, D, f_ini, srate, timeout, jobtag)
        # shape of ret:[nq, ns1z, ns2z, necc]
        nq, ns1z, ns2z, necc = ret.shape
        data = []
        for i in range(nq):
            for j in range(ns1z):
                for k in range(ns2z):
                    for l in range(necc):
                        data.append([q[i], s1z[j], s2z[k], ecc[l], ret[i,j,k,l]])
    add_csv(fsave, data)
    
    if not args.random:
        # np.savetxt(savedir / 'all.txt', ret)
        from .Utils import plot_marker
        # 2. plot
        if len(q) > 0:
            x = q
            y = 1 - ret[:,0,0,0]
            if min(y) <= 1e-8:
                LOGY = False
            else:
                LOGY = True
            plot_marker(x, y, fname = savedir / 'CompMratio.png', 
                        title = 'q vs 1 - FF', 
                        xlabel = 'q', 
                        ylabel = 'log 1 - FF', 
                        ylim = [0, 1],
                        ylog = LOGY)
        
        if len(s1z) > 0:
            x = s1z
            y = 1 - ret[0,:,0,0]
            if min(y) <= 1e-8:
                LOGY = False
            else:
                LOGY = True
            plot_marker(x, y, fname = savedir / 'CompSpin1z.png', 
                        title = 's1z vs 1 - FF', 
                        xlabel = 's1z', 
                        ylabel = 'log 1 - FF', 
                        ylim = [0, 1],
                        ylog = LOGY)
            
        if len(s2z) > 0:
            x = s2z
            y = 1 - ret[0,0,:,0]
            if min(y) <= 1e-8:
                LOGY = False
            else:
                LOGY = True
            plot_marker(x, y, fname = savedir / 'CompSpin2z.png', 
                        title = 's2z vs 1 - FF', 
                        xlabel = 's2z', 
                        ylabel = 'log 1 - FF', 
                        ylim = [0, 1],
                        ylog = LOGY)
            
        if len(ecc) > 0:
            x = ecc
            y = 1 - ret[0,0,0,:]
            if min(y) <= 1e-8:
                LOGY = False
            else:
                LOGY = True
            plot_marker(x, y, fname = savedir / 'CompEcc.png', 
                        title = 'ecc vs 1 - FF', 
                        xlabel = 'ecc', 
                        ylabel = 'log 1 - FF', 
                        ylim = [0, 1],
                        ylog = LOGY)
        
    return 0

    
    