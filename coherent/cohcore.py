#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:14:28 2019

@author: drizl
"""

import numpy as np
from ..strain.strain import gwStrain
from ..h22datatype import c_SI
from ..strain.template import template

#--------------------Combined Strain-------------------#
class gwStrainCOH(object):
    def __init__(self, sLIST):
        self._sLIST = sLIST
        # Check input
        self._check_sLIST()
        
    def _check_sLIST(self):
        for i, strain in enumerate(self):
            # Check datatype
            if not isinstance(strain , gwStrain):
                raise TypeError('The type of input is not gwStrain.')
            
            # Check datalength
            if i==0:
                _len = len(strain)
            elif len(strain) != _len:
                raise ValueError('The length of input strains are not equal to each other.')
            self._Nt = _len
            
            # Check sample rate
            if i==0:
                _srate = strain.fs
            elif strain.fs != _srate:
                raise ValueError('The sample rate of input strains are not equal to each other.')
            self._srate = _srate
            
            # Check epoch
            if i==0:
                _epoch = strain.epoch
            elif strain.epoch != _epoch:
                raise ValueError('The epoch of input strains are not equal to each other.')
            self._epoch = _epoch
            
    def __len__(self):
        return self._len
    
    @property
    def fs(self):
        return self._srate
    
    def __iter__(self):
        for strain in self._sLIST:
            yield strain

    def get_localize_prob_func(self, tmpl,
                               distance_factor = 2):
        if not isinstance(tmpl, template):
            raise TypeError('The type of tmpl should be strain.template.template.')
            
        if tmpl.fs != self.fs:
            raise ValueError('Sample rate of template and strain data is not equal to each other.')
        
        r_min = 0
        r_max = 0 # TODO
        T_min = 0
        T_max = 0 # TODO
        # Get range of integration
        for strain in self:
            # T_max
            loc = strain.ifo_location
            delay = np.linalg.norm(loc) / c_SI + 5e-3
            if delay > T_max:
                T_max = delay
            
            horizon = tmpl.get_horizon(psd = strain.psdfun_setted, ret_SI = True)
            distance = horizon / 4
            print(f'{strain.ifo} horizon = {horizon}')
            if distance > r_max:
                r_max = distance
            
            
        
    
    