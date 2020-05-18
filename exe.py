#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:06:37 2019

@author: drizl
"""
import matplotlib as mlb
mlb.use('Agg')

import numpy as np
from .SXS import SXSh22, save_namecol
from pathlib import Path
from optparse import OptionParser
from .psd import DetectorPSD


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
    parser.add_option('--SXS', type = 'str', action = 'append', default = [], help = 'SXS template for comparision')
    parser.add_option('--prefix', type = 'str', default = '.', help = 'dir for results saving.')
    parser.add_option('--plot', action = 'store_true', help = 'If added, will plot sa results and waveform.')
    parser.add_option('--verbose', action = 'store_true', help = 'If added, will print verbose message.')
    parser.add_option('--hertz', action = 'store_true', help = 'If added, will use dimension Hz for fini.')
    parser.add_option('--maxecc', type = 'float', default = 0, help = 'If zero, will automatically set ecc search range.')
    parser.add_option('--table', type = 'str', default = str(DEFAULT_TABLE), help = 'Path of SXS table.')
    parser.add_option('--srcloc', type = 'str', default = str(DEFAULT_SRCLOC), help = 'Path of SXS waveform data.')
    parser.add_option('--srcloc-all', type = 'str', default = str(DEFAULT_SRCLOC_ALL), help = 'Path of SXS waveform data all modes')
    parser.add_option('--modeL', type = 'int', help = 'mode L for HM')
    parser.add_option('--modeM', type = 'int', help = 'mode M for HM')
    parser.add_option('--psd', type = 'str', help = 'Detector psd.')
    parser.add_option('--flow', type = 'float', default = 0, help = 'Lower frequency cut off for psd.')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')
    args = parser.parse_args(argv)
    return args


def main(argv = None):
    args, empty = parseargs(argv)
    SXSnum_list = args.SXS
    if len(SXSnum_list) == 0:
        SXSnum_list.append('0001')
    approx = args.approx
    jobtag = args.jobtag
    exe = args.executable
    prefix = Path(args.prefix)
    isplot = args.plot
    fini = args.fini
    ishertz = args.hertz
    maxecc = args.maxecc
    table = args.table
    srcloc = args.srcloc
    srcloc_all = args.srcloc_all
    psd = DetectorPSD(args.psd, flow = args.flow)
    timeout = args.timeout
    
    modeL = args.modeL
    modeM = args.modeM
        
    savedir = prefix / approx
    verbose = args.verbose
    if verbose is None:
        verbose = False
    # Mkdir for data saving
    if not savedir.exists():
        savedir.mkdir(parents=True)
    
    
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
        ret = ge.get_overlap(jobtag = jobtag, maxecc = maxecc, timeout = timeout)
        

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
    parser.add_option('--distance', type = 'float', default = 100, help = 'BBH distance in Mpc')
    parser.add_option('--srate', type = 'float', default = 16384, help = 'Sample rate')
    parser.add_option('--jobtag', type = 'str', default = '_test', help = 'Tag for this run')
    parser.add_option('--timeout', type = 'int', default = 60, help = 'Time limit for waveform generation')
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

    args, empty = parser.parse_args(argv)
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
    srate = args.srate
    timeout = args.timeout
    jobtag = args.jobtag
    
    savedir = Path(args.prefix)
    if not savedir.exists():
        savedir.mkdir(parents = True)
    
    # 1. save all
    namecol = [['#mass ratio',
               '#spin1z',
               '#spin2z',
               '#ecc',
               '#FF']]
    fsave = savedir / 'all.csv'
    save_namecol(fsave, data = namecol)
    Comp = CompGenerator(approx1, exe1, approx2, exe2, verbose = verbose)

    if args.random:
        ret = Comp.compare_random(args.min_mratio, args.max_mratio, 
                                  args.min_spin1z, args.max_spin1z, 
                                  args.min_spin2z, args.max_spin2z, 
                                  args.min_ecc, args.max_ecc, 
                                  Num = args.ncompare, 
                                  Mtotal = 16, 
                                  D = 100, f_ini = 40, 
                                  srate = 16384, jobtag = jobtag)
        data = ret
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

    
    