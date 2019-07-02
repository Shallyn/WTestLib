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
                   table = table,
                   srcloc = srcloc,
                   verbose = verbose, 
                   ishertz = ishertz)
        ge = s.construct_generator(approx, exe)
        ret = ge.get_overlap(jobtag = jobtag, maxecc = maxecc)
        

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

