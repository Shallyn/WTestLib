#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:41:46 2019

@author: drizl
"""

import numpy as np
import sys, os, json
from .Utils import switch, CEV, LOG, WARNING, CEV_parse_value, MESSAGE, plot_compare_attach_any, plot_marker
from .h22datatype import dim_h, dim_t, h22base
from pathlib import Path
from .generator import Generator, self_adaptivor
import csv,codecs


DEFAULT_SRCLOC = Path('/Users/drizl/Documents/2018/SEOBNRE/Program_Test/SXS_Data_txt')
DEFAULT_TABLE = Path('/Users/drizl/Documents/2018/SEOBNRE/Program_Test/table_data.json')
DEFAULT_SXS_PLOT_LINESTYLE = '-'
DEFAULT_SXS_PLOT_COLOR = 'red'
DEFAULT_SXS_ALPHA = 0.5
DEFAULT_SXS_LINEWIDTH = 2.0

DEFAULT_FIT_PLOT_LINESTYLE = '--'
DEFAULT_FIT_PLOT_COLOR = 'seagreen'
DEFAULT_FIT_ALPHA = 1.0 
DEFAULT_FIT_LINEWIDTH = 1.0

#-------------Load File--------------#
def loadSXStxtdata(SXSnum, srcloc):
    srcpath = Path(srcloc)
    filename = srcpath / f'BBH_{SXSnum}.txt'
    t, hr, hi = np.loadtxt(filename)
    return t, hr, hi
    
#-----------------GET WAVEFORM PARAMETERS----------------#    
def get_SXS_parameters(SXSnum, pmsName, 
                             srcloc = DEFAULT_SRCLOC,
                             table = DEFAULT_TABLE,
                             **kwargs):
    wfp = SXSh22(SXSnum, srcloc, table, **kwargs)
    return wfp.__dict__[pmsName]

def GET_PARAMS_FROM_TABLE(SXSnum, table):
    with open(table,"r") as load_f:
        load_dict = json.load(load_f)
    SXSnum = 'SXS:BBH:'+SXSnum
    #return load_dict
    for x in load_dict:
        if x['alternative_names'] is SXSnum or SXSnum in x['alternative_names']:
            return x
    sys.stderr.write('Error: No such SXS in table: {}\n'.format(SXSnum))
    return CEV.PMS_ERROR


#------------------CLASS----------------#
class SXSObject(object):
    def __init__(self, SXSnum, table, verbose = False):
        self._SXSnum = SXSnum
        self._table = table
        if verbose:
            sys.stderr.write(f'{LOG}:Loading SXS parameters from table...\n')
        # Parse parameters from table
        pms = GET_PARAMS_FROM_TABLE(SXSnum, table)
        if isinstance(pms, CEV):
            self.CEV_STATE = pms
            if verbose:
                sys.stderr.write(f'{WARNING}Warning: Failed to load parameters of SXS ID: {SXSnum}\n')
        else:
            if verbose:
                sys.stderr.write(f'{LOG}:Loading succeed..Starting parsing\n')
            self.CEV_STATE = CEV.NORMAL
            self._load_pms(pms)
        if verbose:
            sys.stderr.write(f'{LOG}:Loading SXS parameters from table...Done\n')
            
            
    def _load_pms(self, pms):
        # mass ratio - q
        mratio = max(1, pms['initial_mass_ratio'])
        self.q = float('%.6f'%mratio)
        # spin
        spin1 = pms['initial_dimensionless_spin1']
        spin2 = pms['initial_dimensionless_spin2']
        self.s1x = float('%.6f'%spin1[0])
        self.s1y = float('%.6f'%spin1[1])
        self.s1z = float('%.6f'%spin1[2])
        self.s2x = float('%.6f'%spin2[0])
        self.s2y = float('%.6f'%spin2[1])
        self.s2z = float('%.6f'%spin2[2])
        self.spin1 = np.sqrt(self.s1x**2 + self.s1y**2 + self.s1z**2)
        self.spin2 = np.sqrt(self.s2x**2 + self.s2y**2 + self.s2z**2)
        # eccentricity
        self.Sf_ini = pms['initial_orbital_frequency'] / np.pi
        # Relaxed Eccentricity
        self.ecc = pms['relaxed_eccentricity']
        # remnant
        self.final_mass = pms['remnant_mass']
        fspin = pms['remnant_dimensionless_spin']
        self.final_spin = np.sqrt(fspin[0]**2 + fspin[1]**2 + fspin[2]**2)
        self.final_spinx = fspin[0]
        self.final_spiny = fspin[1]
        self.final_spinz = fspin[2]
        fkv = pms['remnant_velocity']
        self.final_kick_v = np.sqrt(fkv[0]**2 + fkv[1]**2 + fkv[2]**2)
        self.final_kick_vx = fkv[0]
        self.final_kick_vy = fkv[1]
        self.final_kick_vz = fkv[2]
        
    @property
    def SXSnum(self):
        return self._SXSnum

class SXSparameters(SXSObject):
    def __init__(self, SXSnum, table, f_ini = 0, Mtotal = 16, D = 100, verbose = False, ishertz = False):
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize SXSObject...\n')
        super(SXSparameters, self).__init__(SXSnum, table, verbose = verbose)
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize SXSObject...Done\n')
        self._D = D
        self._Mtotal = Mtotal
        self._f_ini = f_ini
        self._ishertz = ishertz
    
    @property
    def D(self):
        return self._D
    
    @property
    def Mtotal(self):
        return self._Mtotal
    
    @property
    def m1(self):
        return self._Mtotal * self.q / (1 + self.q)
    
    @property
    def m2(self):
        return self._Mtotal / (1 + self.q)
    
    @property
    def f_ini(self):
        if self._f_ini == 0:
            return self.Sf_ini * dim_t(self._Mtotal)
        else:
            if self._ishertz:
                return self._f_ini
            else:
                return self._f_ini * dim_t(self._Mtotal)
        
    @property
    def f_ini_dimless(self):
        return self.f_ini / dim_t(self.m1 + self.m2)


class SXSh22(SXSparameters, h22base):
    # def __new__(cls, SXSnum, srcloc = DEFAULT_SRCLOC, table = DEFAULT_TABLE , 
    #              f_ini = 0, srate = 16384, Mtotal = 16, D = 100, verbose = False):
    #     SXSparameters.__new__(cls, SXSnum, table, f_ini, Mtotal, D, verbose = verbose)
    
    def __init__(self, SXSnum, srcloc = DEFAULT_SRCLOC, table = DEFAULT_TABLE , 
                 f_ini = 0, srate = 16384, Mtotal = 16, D = 100, verbose = False, ishertz = False):
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize SXSparameters...\n')
        SXSparameters.__init__(self, SXSnum, table, f_ini, Mtotal, D, verbose = verbose, ishertz = ishertz)
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize SXSparameters...Done\n')
            sys.stderr.write(f'{LOG}:Loading SXS template from srcloc...')
        t, hr, hi = loadSXStxtdata(SXSnum, srcloc)
        t /= dim_t(Mtotal)
        self._srcloc = srcloc
        if verbose:
            sys.stderr.write('Done\n')
            sys.stderr.write(f'{LOG}:Initialize h22base...\n')
        h22base.__init__(self, t, hr, hi, srate, verbose = verbose) 
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize h22base...Done\n')
        self._verbose = verbose
        
    def construct_generator(self, approx, executable):
        return SXSCompGenerator(approx, executable, self, verbose = self._verbose)
    
    def copy(self):
        return SXSh22(self._SXSnum, self._srcloc, self._table, self._f_ini, self._srate, self._Mtotal, self._D)
        
        
class SXSCompGenerator(Generator):
    def __init__(self, approx, executable, sxsh22, verbose = False):
        #print(isinstance(sxsh22, SXSh22))
        if not isinstance(sxsh22, SXSh22):
            raise TypeError('Incorrect input type: {}'.format(type(sxsh22)))
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize Generator...\n')
        super(SXSCompGenerator, self).__init__(approx, executable)
        if verbose:
            sys.stderr.write(f'{LOG}:Initialize Generator...Done\n')
        self._core = sxsh22
        self._verbose = verbose
        self._fecc = lambda ecc : self._CMD(m1 = self._core.m1,
                                            m2 = self._core.m2,
                                            s1z = self._core.s1z,
                                            s2z = self._core.s2z,
                                            D = self._core.D,
                                            ecc = ecc,
                                            srate = self._core.srate,
                                            f_ini = self._core.f_ini)
        
    @property
    def SXS(self):
        return self._core
        
        
    def get_waveform(self, ecc = None, 
                     jobtag = 'test',
                     timeout = 60,
                     verbose = None):
        if verbose is None:
            verbose = self._verbose
        if verbose:
            sys.stderr.write(f'{LOG}:Checking SXSh22 status.\n')
        if self._core.CEV_STATE != CEV.NORMAL:
            sys.stderr.write(f'{WARNING}:Abnormal SXSh22...\n')
            return self._core.CEV_STATE
        if verbose:
            sys.stderr.write(f'{LOG}:Checking SXSh22 status...Normal\n')
        if ecc is None:
            ecc = 0
        if ecc is not None and not self.allow_ecc:
            sys.stderr.write(f'{WARNING}: parameter ecc is unused.\n')
        if verbose:
            sys.stderr.write(f'{LOG}:Calling Generator to generate waveform...\n')
        ret = self.__call__(m1 = self._core.m1,
                            m2 = self._core.m2,
                            s1z = self._core.s1z,
                            s2z = self._core.s2z,
                            D = self._core.D,
                            ecc = ecc,
                            srate = self._core.srate,
                            f_ini = self._core.f_ini,
                            jobtag = jobtag,
                            timeout = timeout)
        if verbose:
            sys.stderr.write(f'{LOG}:Generator runs out, check the results...\n')
        if isinstance(ret, CEV):
            if verbose:
                sys.stderr.write(f'{WARNING}:Fail to generate waveform...terminate\n')
            return ret
        if verbose:
            sys.stderr.write(f'{LOG}:Generation succeed, construct h22 data.\n')
        t, hr, hi = ret[:,0], ret[:,1], ret[:,2]
        if verbose:
            sys.stderr.write(f'{LOG}:Pretreatment.\n')
        hr, hi = self._pretreat(hr, hi, self._core._D, self._core._Mtotal)
        h22_wf = h22base(t, hr, hi, self._core._srate)
        return h22_wf
    
    def get_overlap(self, jobtag = 'test', maxecc = 0, **kwargs):
        if self._verbose:
            sys.stderr.write(f'{LOG}:Checking ecc is allowed or not.\n')
        if not self.allow_ecc:
            if self._verbose:
                sys.stderr.write(f'{LOG}:ecc is unused in approx: {self._approx}, now calculate overlap.\n')
            h22_wf = self.get_waveform(jobtag = jobtag)
            ret = self.__core_calculate_overlap(h22_wf)
            wraper = [np.array([0]),np.asarray([ret])]
        else:
            if self._verbose:
                sys.stderr.write(f'{LOG}:ecc is allowed in approx: {self._approx}, now run self-adaptivor.\n')
            wraper = self.__core_scan_ecc_overlap(jobtag = jobtag, maxecc = maxecc, **kwargs)
        if self._verbose:
            sys.stderr.write(f'{LOG}:Calculation complete, construct result processor.\n')
        return CompResults(self, wraper, self._verbose, jobtag = jobtag)
        
    def __core_calculate_overlap(self, h22_wf, verbose = None):
        if verbose is None:
            verbose = self._verbose
        if verbose:
            sys.stderr.write(f'{LOG}:Checking input h22 status...\n')
        if isinstance(h22_wf, CEV):
            if verbose:
                sys.stderr.write(f'{WARNING}:Abnormal h22...\n')
            return 0,0,-1,0,h22_wf.value
        SXS = self._core.copy()
        # Check sample rate
        if verbose:
            sys.stderr.write(f'{LOG}:Align data for comparison...')
        SXS, h22_wf, tmove = h22_alignment(SXS, h22_wf)
        if verbose:
            sys.stderr.write('Done\n')
            sys.stderr.write(f'{LOG}:Calculating overlap...')
        fs = SXS.srate
        NFFT = len(SXS)
        df = fs/NFFT
        Stilde = SXS.h22f
        htilde = h22_wf.h22f
        O11 = np.sum(Stilde * Stilde.conjugate()).real * df
        O22 = np.sum(htilde * htilde.conjugate()).real * df
        Ox = Stilde * htilde.conjugate()
        Oxt = np.fft.ifft(Ox) * fs
        Oxt_abs = np.abs(Oxt) / np.sqrt(O11 * O22)
        idx = np.where(Oxt_abs == max(Oxt_abs))[0][0]
        lth = len(Oxt_abs)
        if idx > lth / 2:
            tc = (idx - lth) / fs
        else:
            tc = idx / fs
        phic = np.angle(Oxt[idx])
        if verbose:
            sys.stderr.write('Done\n')
        return tc, phic, Oxt_abs[idx], tmove, CEV.SUCCESS.value
    
    def __core_scan_ecc_overlap(self, estep = 0.01, maxitr = None, verbose = False,
                                prec_x = 1e-6, prec_y = 1e-6, jobtag = 'test', maxecc = 0):
        # Parse ecc
        if self._verbose:
            sys.stderr.write(f'{LOG}:Parsing ecc...')
        ecc_range = parse_ecc(self._core.ecc, maxecc = maxecc)
        if self._verbose:
            sys.stderr.write('Done: Ecc Range: {}\n'.format(ecc_range))
            sys.stderr.write(f'{LOG}:Construct self-adaptivor...')
        def ecc_wf(ecc):
            h22_wf = self.get_waveform(ecc = ecc, verbose = False, jobtag = jobtag)
            ret = self.__core_calculate_overlap(h22_wf, verbose = False)
            return ret
        SA = self_adaptivor(ecc_wf, ecc_range, estep, outindex = 2)
        if self._verbose:
            sys.stderr.write('Done\n')
            sys.stderr.write(f'{LOG}:Run self-adaptivor...\n')
        return SA.run(maxitr = maxitr, 
                      verbose = self._verbose, 
                      prec_x = prec_x, 
                      prec_y = prec_y)

class CompResults(object):
    def __init__(self, generator, results, verbose = False, jobtag = 'test'):
        self._core = generator
        self._results = results
        self._verbose = verbose
        self._jobtag = jobtag
        if self._verbose:
            sys.stderr.write(f'{LOG}:Parsing results...')
        self._parse_results()
        if self._verbose:
            sys.stderr.write('Done\n')
            sys.stderr.write(f'{LOG}:Parsing waveform...\n')
        self._get_fit_waveform()
        if self._verbose:
            sys.stderr.write(f'{LOG}:Parsing waveform...Done\n')
    
    def _parse_results(self):
        ecc, olpout = self._results
        self._ecc = ecc
        self._tc = olpout[:,0]
        self._phic = olpout[:,1]
        self._FF = olpout[:,2]
        self._tmove = olpout[:,3]
        self._CEV_STATE = CEV_parse_value(olpout[:,4].astype(np.int))
        fitarg = self._FF.argmax()
        self._max_FF = self._FF[fitarg]
        self._fit_tc = self._tc[fitarg]
        self._fit_phic = self._phic[fitarg]
        self._fit_STATE = self._CEV_STATE[fitarg]
        self._fit_ecc = self._ecc[fitarg]
        self._fit_tmove = self._tmove[fitarg]
    
    def plot_waveform_fit(self, fname, 
                          FIT_linestyle = None, FIT_color = None, 
                          FIT_alpha = None, FIT_linewidth = None,
                          SXS_linestyle = None, SXS_color = None, 
                          SXS_alpha = None, SXS_linewidth = None,
                          **kwargs):
        if self.CEV_STATE_fit is not CEV.SUCCESS:
            sys.stderr.write(f'{WARNING}:State is not success, skip waveform plotting.\n')
        else:
            filename = fname
            if self._verbose:
                sys.stderr.write(f'{LOG}:Plotting waveform, name as {filename}\n')

            if FIT_linestyle is None:
                FIT_linestyle = DEFAULT_FIT_PLOT_LINESTYLE
            if FIT_color is None:
                FIT_color = DEFAULT_FIT_PLOT_COLOR
            if FIT_alpha is None:
                FIT_alpha = DEFAULT_FIT_ALPHA
            if FIT_linewidth is None:
                FIT_linewidth = DEFAULT_FIT_LINEWIDTH
    
            if SXS_linestyle is None:
                SXS_linestyle = DEFAULT_SXS_PLOT_LINESTYLE
            if SXS_color is None:
                SXS_color = DEFAULT_SXS_PLOT_COLOR
            if SXS_alpha is None:
                SXS_alpha = DEFAULT_SXS_ALPHA
            if SXS_linewidth is None:
                SXS_linewidth = DEFAULT_SXS_LINEWIDTH
    
    
            SXSplot = dict()
            FITplot = dict()
            SXSplot['x'] = self.generator.SXS.time
            SXSplot['y'] = self.generator.SXS.real
            SXSplot['name'] = self.generator.SXS.SXSnum
            SXSplot['linestyle'] = SXS_linestyle
            SXSplot['color'] = SXS_color
            SXSplot['alpha'] = SXS_alpha
            SXSplot['linewidth'] = SXS_linewidth
            
            FITplot['x'] = self.h22_fit.time
            FITplot['y'] = self.h22_fit.real
            FITplot['name'] = self.generator.approx
            FITplot['linestyle'] = FIT_linestyle
            FITplot['color'] = FIT_color
            FITplot['alpha'] = FIT_alpha
            FITplot['linewidth'] = FIT_linewidth
    
            
            plot_compare_attach_any([SXSplot, FITplot], savefig = filename, tstart=0, **kwargs)
        
    
    def plot_results(self, fname, **kwargs):
        if not self.generator.allow_ecc:
            sys.stderr.write(f'{WARNING}:Parameter ecc is unused, skip sa plot.\n')
        else:
            if not isinstance(fname, str):
                filename = 'scan.png'
            else:
                filename = fname
            if self.CEV_STATE_fit is not CEV.SUCCESS:
                sys.stderr.write(f'{WARNING}:State is not success, skip sa plot.\n')
            else:
                if self._verbose:
                    sys.stderr.write(f'{LOG}:Plotting sa results, name as {filename}\n')
                plot_marker(self.ecc_out, self.FF_out, 
                            title = self.generator.SXS.SXSnum,
                            xlabel = 'eccentricity',
                            ylabel = 'FF',
                            fname = filename, **kwargs)
            
    def save_waveform_fit(self, fname, **kwargs):
        if self.h22_fit is not None:
            if self._verbose:
                sys.stderr.write(f'{LOG}:Saving fitting waveform to {fname}.\n')
            self._h22_fit.saveh22(fname, **kwargs)
    
    def save_results(self, fname, **kwargs):
        # if not isinstance(fname, str):
        #     filename = 'scan_results.txt'
        # else:
        #     filename = fname
        filename = fname
        if self._verbose:
            sys.stderr.write(f'{LOG}:Saving sa results to {filename}\n')
        data = np.stack([self.tc_out.astype(np.str), 
                         self.phic_out.astype(np.str),
                         self.FF_out.astype(np.str),
                         self.CEV_STATE_out.name], axis = 1)
        np.savetxt(filename, data, fmt = '%s', **kwargs)
    
    
    def save_fit(self, fname, add_tag = False, **kwargs):
        # if not isinstance(fname, str):
        #     filename = 'results.csv'
        # else:
        #     filename = fname
        
        filename = fname
        if self._verbose:
            sys.stderr.write(f'{LOG}:Saving SXS fitting results to {filename}\n')
        
        if add_tag:
            fsplit = fname.split('.')
            suffix = fsplit[-1]
            filename = '.'.join(fsplit[:-1])
            filename += f'_{self._jobtag}.{suffix}'
        
        data = [[self.generator.SXS.SXSnum,
                self.generator.approx,
                self.generator.SXS.q,
                self.generator.SXS.s1x,
                self.generator.SXS.s1y,
                self.generator.SXS.s1z,
                self.generator.SXS.s2x,
                self.generator.SXS.s2y,
                self.generator.SXS.s2z,
                self.generator.SXS.ecc,
                self.generator.SXS.f_ini_dimless,
                self.tc_fit,
                self.phic_fit,
                self.max_FF,
                self.ecc_fit,
                self.CEV_STATE_fit.name]]
        file = codecs.open(filename, 'a+', "gbk")
        writer = csv.writer(file)

        writer.writerows(data)
        file.close()
    
    def _get_fit_waveform(self):
        if self.CEV_STATE_fit is not CEV.SUCCESS:
            msg = f'SXSnum: {self.generator.SXS.SXSnum}\napprox: {self.generator.approx}\nerrtype: {self.CEV_STATE_fit.name}\nCMD: {self.generator._fecc(self.ecc_fit)}'
            sys.stderr.write(f'{WARNING}:Failed Job: \n{MESSAGE}: \n{msg}\n')
            self._errmsg = msg
            self._h22_fit = None
        else:
            self._errmsg = f'{self.generator.SXS.SXSnum}:{self.generator.approx}:Success'
            fit = self.generator.get_waveform(ecc = self.ecc_fit,
                                              jobtag = self._jobtag,
                                              verbose = False)
            fit.apply(-self.tc_fit + self.tmove_fit, -self.phic_fit)
            self._h22_fit = fit
    @property
    def h22_fit(self):
        return self._h22_fit
            
    @property
    def generator(self):
        return self._core
    
    @property
    def ecc_out(self):
        return self._ecc
    
    @property
    def tc_out(self):
        return self._tc
    
    @property
    def phic_out(self):
        return self._phic
    
    @property
    def FF_out(self):
        return self._FF
    
    @property
    def CEV_STATE_out(self):
        return self._CEV_STATE
    
    @property
    def ecc_fit(self):
        return self._fit_ecc
    
    @property
    def tc_fit(self):
        return self._fit_tc
    
    @property
    def phic_fit(self):
        return self._fit_phic
    
    @property
    def tmove_fit(self):
        return self._fit_tmove
    
    @property
    def max_FF(self):
        return self._max_FF
    
    @property
    def CEV_STATE_fit(self):
        return self._fit_STATE
    
    @property
    def ErrorMsg(self):
        return self._errmsg
    
    
#--------Utils-------#
def h22_alignment(wfA, wfB):
    fs_A = wfA.srate
    fs_B = wfB.srate
    if fs_A > fs_B:
        wfA.resample(fs_B)
        fs = fs_B
    else:
        wfB.resample(fs_A)
        fs = fs_A
    peak_A = wfA.argpeak
    peak_B = wfB.argpeak           

    if peak_A > peak_B:
        idx_A = peak_A - peak_B
        idx_B = 0
        tmove = -idx_A / fs
    else:
        idx_A = 0
        idx_B = peak_B - peak_A
        tmove = idx_B / fs

    wfA = wfA[idx_A:]
    wfB = wfB[idx_B:]
    len_A = len(wfA)
    len_B = len(wfB)
    peak_A = wfA.argpeak
    peak_B = wfB.argpeak           
    tail_A = len_A - peak_A
    tail_B = len_B - peak_B

    # Check data shape 
    if tail_A > tail_B:
        lpad = tail_A - tail_B
        wfB.pad((0,lpad), 'constant')
    else:
        lpad = tail_B - tail_A
        wfA.pad((0,lpad), 'constant')
        
    return wfA, wfB, tmove

def parse_ecc(ecc, maxecc):
    if maxecc > 0:
        return [0, maxecc]
    if type(ecc) is str:
        try:
            xecc = float(ecc[1:])
        except:
            xecc = 0.5
        ecc = -1
        if xecc > 0.5:
            elip_min = 0
            elip_max = 0.6
        else:
            elip_max = min(0.6, xecc + 0.1)
            elip_min = max(0, xecc - 0.1)
    else:
        elip_max = min(0.6, ecc + 0.1)
        elip_min = max(0, ecc - 0.1)
    return [elip_min, elip_max]
    

DEFAULT_NAMECOL = ['#SXS id',
            '#approx',
            '#mass ratio',
            '#spin1z',
            '#spin1y',
            '#spin1z',
            '#spin2x',
            '#spin2y',
            '#spin2z',
            '#relaxed ecc',
            '#f_ini',
            '#tc_fit',
            '#phic_fit',
            '#FF',
            '#ecc_fit',
            '#Status']
def save_namecol(filename):
    file = codecs.open(filename, 'wb', "gbk")
    writer = csv.writer(file)
    data = [DEFAULT_NAMECOL]
    writer.writerows(data)
    file.close()
    
def load_csv(filename):
    file = codecs.open(filename, 'r', "gbk")
    reader = csv.reader(file,dialect='excel')
    data = []
    for i,line in enumerate(reader):
        if i>0:
            data.append(line)
    file.close()
    return data

def add_csv(filename, data):
    file = codecs.open(filename, 'a+', "gbk")
    writer = csv.writer(file)
    writer.writerows(data)
    file.close()

def resave_results(prefix, target):
    from pathlib import Path
    pre_list = prefix.split('/')
    prefix = pre_list.pop(-1)
    loc = Path('/'.join(pre_list))
    lth = len(prefix)
    save_namecol(target)
    for file in loc.iterdir():
        if file.name[:lth] == prefix:
            sys.stderr.write(f'{LOG}:Saving {file.name} to {target}...\n')
            data = load_csv(file)
            add_csv(target, data)