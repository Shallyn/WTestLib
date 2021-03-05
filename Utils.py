#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:36:40 2019

@author: drizl
"""

import numpy as np
import time, os, sys
import subprocess
from enum import Enum, auto
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import splev, splrep
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
DEBUG = debugWrapper('DEBUG')

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
    sys.stderr.write(remarks + '|' + pcg_str+ '%|')
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


#------cmd-------#
def construct_cmd(exe, **kwargs):
    ops = [str(exe)]
    for kw in kwargs:
        kwR = kw.replace('_','-')
        ops.append(f'--{kwR}={kwargs[kw]}')
    return ' '.join(ops)

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
            if len(data) == 0:
                return CEV.GEN_FAIL, None
            return CEV.SUCCESS, data
        time_cost = time.time() - t_start
        if time_cost > timeout:
            obj.terminate()
            os.remove(name_out)
            return CEV.GEN_TIMEOUT, None
        time.sleep(0.5)

def cmd_hang(CMD, name_err):
    obj = subprocess.Popen(CMD, stdout = subprocess.PIPE, stderr = open(name_err, "w"), shell = True)
    return obj, name_err

#-------PLOT------#
def plot_compare_attach_any(LIST, 
                            tstart = None,
                            figsize=(18, 6),
                            savefig = 'savefig.jpg',
                            Nlth_end = 8,
                            savedpi = 200,
                            transparent = False):
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
        plt.savefig(savefig, transparent = transparent, dpi = savedpi)
        plt.close()
    else:
        plt.show()

def plot_marker(x, y, 
                figsize = None,
                marker = '.',
                color = 'black',
                fname = 'marker.png',
                title = 'marker plot',
                xlabel = 'x',
                ylabel = 'y',
                xlim = None,
                ylim = None,
                ylog = False):
        if figsize is None:
            figsize = (8,5)
            
        plt.figure(figsize = figsize)
        plt.scatter(x, y, marker = marker, color = color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if ylog:
            plt.yscale('log')
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(ylim)
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
        
def polyfit(x, y, order):
    if len(x) != len(y) or order >= len(x):
        return None
    n = len(x)
    m = int(order)+1
    X = np.zeros([n, m])
    Y = y.copy()
    for i in range(m):
        X[:,i] = np.power(x, i)
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)

def polyfit2d(x, y, z, xorder, yorder):
    indtot = int((xorder + 1) * (yorder + 1)) - 1
    def parse_index(ind):
        indi = ind % (xorder + 1)
        indj = int(ind / (xorder + 1))
        return indi, indj
    X = np.zeros([indtot, indtot])
    Y = np.zeros(indtot)
    for J in range(indtot):
        indl, indm = parse_index(J)
        for I in range(indtot):
            indi, indj = parse_index(I)
            X[J, I] = np.sum(np.power(x, indi+indl) * np.power(y, indj + indm))
        Y[J] = np.sum(np.power(x, indi) * np.power(y, indm) * z)
    Sol = np.dot(Y, np.linalg.inv(X))
    ret = np.zeros([yorder + 1, xorder + 1])
    for J in range(indtot):
        indi, indj = parse_index(J)
        ret[indj, indi] = Sol[J]
    return ret

def splfind(x, y, val, eps):
    ay = np.abs(y - val)
    sply = splrep(x, ay)
    idx = np.where( ay == np.min(ay))[0][0]
    if idx == 0:
        return x[0]
    if ay[idx] == 0:
        return x[idx]
    dayi = splev(x[idx], sply, der = 1)
    if dayi < 0:
        x1 = x[idx]
        x2 = x[idx+1]
    else:
        x1 = x[idx-1]
        x2 = x[idx]
    while(np.abs(x1 - x2) > eps):
        xi = (x1 + x2) / 2.
        dayi = splev(xi, sply, der = 1)
        if dayi < 0:
            x2 = xi
        else:
            x1 = xi
    return (x1 + x2) / 2

    
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


# Spin weighted -2 Spherical Harmonics
def SpinWeightedM2SphericalHarmonic(theta, phi, l, m):
    modetag = l*100 + m
    GET_SQRT = np.sqrt
    CST_PI = np.pi
    GET_COS = np.cos
    GET_SIN = np.sin
    GET_POW = np.power
    for case in switch(modetag):
        if case(198):
            # mode 2, -2
            fac = GET_SQRT( 5.0 / ( 64.0 * CST_PI ) ) * ( 1.0 - GET_COS( theta ))*( 1.0 - GET_COS( theta ))
            break
        if case(199):
            # mode 2, -1
            fac = GET_SQRT( 5.0 / ( 16.0 * CST_PI ) ) * GET_SIN( theta )*( 1.0 - GET_COS( theta ))
            break
        if case(200):
            # mode 2, 0
            fac = GET_SQRT( 15.0 / ( 32.0 * CST_PI ) ) * GET_SIN( theta )*GET_SIN( theta )
            break
        if case(201):
            # mode 2, 1
            fac = GET_SQRT( 5.0 / ( 16.0 * CST_PI ) ) * GET_SIN( theta )*( 1.0 + GET_COS( theta ))
            break
        if case(202):
            # mode 2, 2
            fac = GET_SQRT( 5.0 / ( 64.0 * CST_PI ) ) * ( 1.0 + GET_COS( theta ))*( 1.0 + GET_COS( theta ))
            break
        if case(297):
            # mode 3, -3
            fac = GET_SQRT(21.0/(2.0*CST_PI))*GET_COS(theta/2.0)*GET_POW(GET_SIN(theta/2.0),5.0)
            break
        if case(298):
            # mode 3, -2
            fac = GET_SQRT(7.0/(4.0*CST_PI))*(2.0 + 3.0*GET_COS(theta))*GET_POW(GET_SIN(theta/2.0),4.0)
            break
        if case(299):
            # mode 3, -1
            fac = GET_SQRT(35.0/(2.0*CST_PI))*(GET_SIN(theta) + 4.0*GET_SIN(2.0*theta) - 3.0*GET_SIN(3.0*theta))/32.0
            break
        if case(300):
            # mode 3, 0
            fac = (GET_SQRT(105.0/(2.0*CST_PI))*GET_COS(theta)*GET_POW(GET_SIN(theta),2.0))/4.0
            break
        if case(301):
            # mode 3, 1
            fac = -GET_SQRT(35.0/(2.0*CST_PI))*(GET_SIN(theta) - 4.0*GET_SIN(2.0*theta) - 3.0*GET_SIN(3.0*theta))/32.0
            break
        if case(302):
            # mode 3, 2
            fac = GET_SQRT(7.0/CST_PI)*GET_POW(GET_COS(theta/2.0),4.0)*(-2.0 + 3.0*GET_COS(theta))/2.0
            break
        if case(303):
            # mode 3, 3
            fac = -GET_SQRT(21.0/(2.0*CST_PI))*GET_POW(GET_COS(theta/2.0),5.0)*GET_SIN(theta/2.0)
            break
        if case(396):
            # mode 4, -4
            fac = 3.0*GET_SQRT(7.0/CST_PI)*GET_POW(GET_COS(theta/2.0),2.0)*GET_POW(GET_SIN(theta/2.0),6.0)
            break
        if case(397):
            # mode 4, -3
            fac = 3.0*GET_SQRT(7.0/(2.0*CST_PI))*GET_COS(theta/2.0)*(1.0 + 2.0*GET_COS(theta))*GET_POW(GET_SIN(theta/2.0),5.0)
            break
        if case(398):
            # mode 4, -2
            fac = (3.0*(9.0 + 14.0*GET_COS(theta) + 7.0*GET_COS(2.0*theta))*GET_POW(GET_SIN(theta/2.0),4.0))/(4.0*GET_SQRT(CST_PI))
            break
        if case(399):
            # mode 4, -1
            fac = (3.0*(3.0*GET_SIN(theta) + 2.0*GET_SIN(2.0*theta) + 7.0*GET_SIN(3.0*theta) - 7.0*GET_SIN(4.0*theta)))/(32.0*GET_SQRT(2.0*CST_PI))
            break
        if case(400):
            # mode 4, 0
            fac = (3.0*GET_SQRT(5.0/(2.0*CST_PI))*(5.0 + 7.0*GET_COS(2.0*theta))*GET_POW(GET_SIN(theta),2.0))/16.0
            break
        if case(401):
            # mode 4, 1
            fac = (3.0*(3.0*GET_SIN(theta) - 2.0*GET_SIN(2.0*theta) + 7.0*GET_SIN(3.0*theta) + 7.0*GET_SIN(4.0*theta)))/(32.0*GET_SQRT(2.0*CST_PI))
            break
        if case(402):
            # mode 4, 2
            fac = (3.0*GET_POW(GET_COS(theta/2.0),4.0)*(9.0 - 14.0*GET_COS(theta) + 7.0*GET_COS(2.0*theta)))/(4.0*GET_SQRT(CST_PI))
            break
        if case(403):
            # mode 4, 3
            fac = -3.0*GET_SQRT(7.0/(2.0*CST_PI))*GET_POW(GET_COS(theta/2.0),5.0)*(-1.0 + 2.0*GET_COS(theta))*GET_SIN(theta/2.0)
            break
        if case(404):
            # mode 4, 4
            fac = 3.0*GET_SQRT(7.0/CST_PI)*GET_POW(GET_COS(theta/2.0),6.0)*GET_POW(GET_SIN(theta/2.0),2.0)
            break
        if case(495):
            fac = GET_SQRT(330.0/CST_PI)*GET_POW(GET_COS(theta/2.0),3.0)*GET_POW(GET_SIN(theta/2.0),7.0)
            break
        if case(496):
            fac = GET_SQRT(33.0/CST_PI)*GET_POW(GET_COS(theta/2.0),2.0)*(2.0 + 5.0*GET_COS(theta))*GET_POW(GET_SIN(theta/2.0),6.0)
            break
        if case(497):
            fac = (GET_SQRT(33.0/(2.0*CST_PI))*GET_COS(theta/2.0)*(17.0 + 24.0*GET_COS(theta) + 15.0*GET_COS(2.0*theta))*GET_POW(GET_SIN(theta/2.0),5.0))/4.0
            break
        if case(498):
            fac = (GET_SQRT(11.0/CST_PI)*(32.0 + 57.0*GET_COS(theta) + 36.0*GET_COS(2.0*theta) + 15.0*GET_COS(3.0*theta))*GET_POW(GET_SIN(theta/2.0),4.0))/8.0
            break
        if case(499):
            fac = (GET_SQRT(77.0/CST_PI)*(2.0*GET_SIN(theta) + 8.0*GET_SIN(2.0*theta) + 3.0*GET_SIN(3.0*theta) + 12.0*GET_SIN(4.0*theta) - 15.0*GET_SIN(5.0*theta)))/256.0
            break
        if case(500):
            fac = (GET_SQRT(1155.0/(2.0*CST_PI))*(5.0*GET_COS(theta) + 3.0*GET_COS(3.0*theta))*GET_POW(GET_SIN(theta),2.0))/32.0
            break
        if case(501):
            fac = GET_SQRT(77.0/CST_PI)*(-2.0*GET_SIN(theta) + 8.0*GET_SIN(2.0*theta) - 3.0*GET_SIN(3.0*theta) + 12.0*GET_SIN(4.0*theta) + 15.0*GET_SIN(5.0*theta))/256.0
            break
        if case(502):
            fac = GET_SQRT(11.0/CST_PI)*GET_POW(GET_COS(theta/2.0),4.0)*(-32.0 + 57.0*GET_COS(theta) - 36.0*GET_COS(2.0*theta) + 15.0*GET_COS(3.0*theta))/8.0
            break
        if case(503):
            fac = -GET_SQRT(33.0/(2.0*CST_PI))*GET_POW(GET_COS(theta/2.0),5.0)*(17.0 - 24.0*GET_COS(theta) + 15.0*GET_COS(2.0*theta))*GET_SIN(theta/2.0)/4.0
            break
        if case(504):
            fac = GET_SQRT(33.0/CST_PI)*GET_POW(GET_COS(theta/2.0),6.0)*(-2.0 + 5.0*GET_COS(theta))*GET_POW(GET_SIN(theta/2.0),2.0)
            break
        if case(505):
            fac = -GET_SQRT(330.0/CST_PI)*GET_POW(GET_COS(theta/2.0),7.0)*GET_POW(GET_SIN(theta/2.0),3.0)
            break
        else:
            raise Exception(f'Unsupported mode {(l, m)}')
    return fac * np.exp(1.j* m * phi)


def estimate_initcond(q, nrPr, nrPphi, d):
    p1Vec = np.array([nrPr, nrPphi])
    p2Vec = np.array([-nrPr, -nrPphi])
    m1 = q/(q + 1)
    m2 = 1/(q + 1)
    v1Vec = p1Vec/m1
    v2Vec = p2Vec/m2
    vVec = v1Vec/m2
    mu = m1*m2
    L = (vVec[1]/d)*d*d
    pr = vVec[0]
    return pr, L
