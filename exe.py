#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:06:37 2019

@author: drizl
"""

import numpy as np
from .SXS import SXSh22, save_namecol
from .Utils import parseargs
from pathlib import Path


def main(argv = None):
    args, empty = parseargs(argv)
    SXSnum_list = args.SXS
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