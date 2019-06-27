#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:36:40 2019

@author: drizl
"""
import matplotlib as mlb
mlb.use('Agg')

import numpy as np
from optparse import OptionParser
import time, os, sys
import subprocess
from enum import Enum, auto
from scipy.interpolate import InterpolatedUnivariateSpline
from collections import Iterable
import matplotlib.pyplot as plt
from matplotlib import gridspec


#------Color Str------#
class COLOR(object):
    LOG = '\033[1;34;47m'
    WARNING = '\033[4;31m'
    DEBUG = '\033[4;31;46m'
    END = '\033[0m'

def logWrapper(log):
    return f'{COLOR.LOG}{log}{COLOR.END}'
LOG = logWrapper('LOG')

def warningWrapper(warning):
    return f'{COLOR.WARNING}{warning}{COLOR.END}'
WARNING = warningWrapper('Warning')

def debugWrapper(debug):
    return f'{COLOR.DEBUG}{debug}{COLOR.END}'
MESSAGE = debugWrapper('Message')

#------Exception Type------#
class CEV(Enum):
    SUCCESS = auto()
    GEN_FAIL = auto()
    PMS_ERROR = auto()
    NORMAL = auto()
    GEN_TIMEOUT = auto()
    UNKNOWN = auto()
    
def CEV_parse_value(val):
    if isinstance(val, int):
        try:
            return CEV(val).name
        except:
            return CEV.UNKNOWN
    if isinstance(val, Iterable):
        ret = []
        for value in val:
            try:
                rst = CEV(value)
            except:
                rst = CEV.UNKNOWN
            ret.append(rst)
        return CEV_Array(ret)
    
class CEV_Array(object):
    def __init__(self, array):
        self._array = array
    
    def __iter__(self):
        for x in self._array:
            yield x
            
    @property
    def name(self):
        return np.array([itr.name for itr in self])
    
    @property
    def value(self):
        return np.array([itr.value for itr in self])
    
    def __str__(self):
        return '{}'.format(self._array)
    
    def __len__(self):
        return len(self._array)
    
    def __repr__(self):
        return self.__str__()
    
    def __format__(self):
        return self.__str__()

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, np.integer):
            return self._array[key]
        return self._getslice(key)

    def __setitem__(self, key, value):
        if isinstance(value, int):
            value = CEV_parse_value(value)
        if not isinstance(value, CEV):
            raise ValueError('The value to be set must be CEV or int.')
        self._array[key] = value

    def _getslice(self, index):
        if index.start is not None and index.start < 0:
            raise ValueError(('Negative start index ({}) is not supported').format(index.start))        
        return CEV_Array(self._array[index])


#------shell tools-----#
def mkdir(path):
    isPathExist = os.path.exists(path)
    if not isPathExist:
        os.makedirs(path)

#-----Progress-----#
def Progress_with_bar(itr,N):
    arrow = '|'
    pcg_str = '%.2f'%min( 100, float(101*itr/N)) 
    pcg = float(pcg_str)
    for i in range(50):
        if 2 * i < pcg:
            arrow += '>'
        else:
            arrow += ' '
    arrow += '|'
    sys.stderr.write('\r')
    sys.stderr.write(pcg_str+ '%|'+arrow)
    sys.stderr.flush()
    time.sleep(0.02)
    
def Progress(itr,N, remarks = ''):
    pcg_str = '%.2f'%min( 100, float(101*itr/N)) 
    sys.stderr.write('\r')
    sys.stderr.write(pcg_str+ '%|'+remarks)
    sys.stderr.flush()
    time.sleep(0.02)
    
def Progress_time(dt, itr, N, remarks = None):
    tr_str = '%.1f'%( (dt+0.02) * (N-itr) / 60)
    pcg_str = '%.2f'%min(100, float(101*itr/N))
    sys.stderr.write('\r')
    if remarks is None:
        printout = pcg_str+ '%|time remain: '+tr_str+' min'
    else:
        printout = pcg_str+ '%|time remain: '+tr_str+' min-' + remarks
    sys.stderr.write(printout)
    sys.stderr.flush()
    time.sleep(0.02)

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
    parser.add_option('--jobtag', type = 'str', default = 'test', help = 'Jobtag for the code run')
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

#------cmd-------#
def cmd(CMD, timeout = 60):
    obj = subprocess.Popen(CMD,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
    t_bgn = time.time()
    while(True):
        if obj.poll() is not None:
            return True
        time_cost = time.time() - t_bgn
        if time_cost > timeout:
            obj.terminate()
            return False
        time.sleep(0.5)
        
def cmd_stdout_cev(CMD, name_out, timeout = 60):
    obj = subprocess.Popen(CMD, stdout = open(name_out, "w"), shell = True)
    t_start = time.time()
    while(True):
        if obj.poll() is not None:
            data = np.loadtxt(name_out)
            os.remove(name_out)
            return CEV.SUCCESS, data
        time_cost = time.time() - t_start
        if time_cost > timeout:
            obj.terminate()
            os.remove(name_out)
            return CEV.GEN_TIMEOUT, None
        time.sleep(0.5)

#-------PLOT------#
def plot_compare_attach_any(LIST, 
                            tstart = None,
                            figsize=(18, 6),
                            savefig = 'savefig.jpg',
                            Nlth_end = 8,
                            savedpi = 200):
    # LIST [wfdict_1, wfdict_2, ...]
    # wfdict: 'name'
    #         'x'
    #         'y'
    #         'linestyle'
    #         'color'
    #         'linewidth'
    #         'alpha'
    namelist = []
    tend_ele_list = []
    rgh_ele_list = []
    t_start_list = []
    type_alpha = dict()
    type_linewidth = dict()
    type_color = dict()
    type_line = dict()
    Xdict = dict()
    Ydict = dict()
    for i,wfdict in enumerate(LIST):
        key = wfdict.pop('name', f'Waveform_{i}')
        namelist.append(key)
        x = wfdict.pop('x', None)
        if x is None:
            sys.stderr.write(f'{WARNING}:Invalid waveform dict.')
            continue
        Xdict[key] = x
        y = wfdict.pop('y', None)
        if y is None:
            sys.stderr.write(f'{WARNING}:Invalid waveform dict.')
            continue
        Ydict[key] = y
        t_start_list.append(x[0])
        t_peak = x[y.real.argmax()]
        t_bott = x[y.real.argmin()]
        tend_ele_list.append(t_peak)
        tend_ele_list.append(t_bott)
        rgh_ele_list.append(np.min(y))
        rgh_ele_list.append(np.max(y))
        plt_line = wfdict.pop('linestyle', None)
        plt_color = wfdict.pop('color', None)
        plt_linewidth = wfdict.pop('linewidth', None)
        plt_alpha = wfdict.pop('alpha', None)
        type_line[key] = plt_line
        type_color[key] = plt_color
        type_linewidth[key] = plt_linewidth
        type_alpha[key] = plt_alpha


    t_end_peak = max(tend_ele_list)
    t_start_peak = min(tend_ele_list)
    if tstart is None:
        t_start = min(t_start_list)
    else:
        t_start = tstart

    # END part:
    lth_end = (t_end_peak - t_start_peak) * Nlth_end
    t_divg = (t_start_peak + t_end_peak - lth_end) / 2
    rg_end = [t_divg, t_end_peak + lth_end/2]
    rg_start = [t_start, t_divg]
    rg_h = [min(rgh_ele_list)*1.2, max(rgh_ele_list)*1.2]
    
    plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(1, 5)
    gs.update(left=0.05, right=0.95, wspace=0.02)
    ax1 = plt.subplot(gs[0,:4])
    ax2 = plt.subplot(gs[0,4])
    for key in Xdict:
        x = Xdict[key]
        y = Ydict[key]
        ax1.plot(x, y, 
                 linestyle = type_line[key], 
                 color = type_color[key], 
                 linewidth = type_linewidth[key], 
                 alpha = type_alpha[key])
    ax1.set_xlim(rg_start)
    ax1.set_ylim(rg_h)
    ax1.legend(namelist)
    for key in Xdict:
        x = Xdict[key]
        y = Ydict[key]
        ax2.plot(x, y, 
                 linestyle = type_line[key], 
                 color = type_color[key], 
                 linewidth = type_linewidth[key], 
                 alpha = type_alpha[key])
    ax2.set_xlim(rg_end)
    ax2.set_ylim(rg_h)
    ax2.set_yticks([])
    if savefig != False:
        plt.savefig(savefig, dpi = savedpi)

def plot_marker(x, y, 
                figsize = None,
                marker = '.',
                color = 'black',
                fname = 'marker.png',
                title = 'marker plot',
                xlabel = 'x',
                ylabel = 'y'):
        if figsize is None:
            figsize = (8,5)
            
        plt.figure(figsize = figsize)
        plt.plot(x, y, marker = marker, color = color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(fname, dpi = 100)


#-----Other Methods-----#
class interp1d_complex(object):
    def __init__(self, x, y, w=None, bbox = [None]*2, k=3,ext=0, check_finite = False):
        yreal = y.real
        yimag = y.imag
        self._func_real = InterpolatedUnivariateSpline(x,yreal,w=w,bbox=bbox,k=k,ext=ext,check_finite=check_finite)
        self._func_imag = InterpolatedUnivariateSpline(x,yimag,w=w,bbox=bbox,k=k,ext=ext,check_finite=check_finite)

    def __call__(self, x):
        return self._func_real(x) + 1.j*self._func_imag(x)
        

    
#-----switch method-----#
class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False
