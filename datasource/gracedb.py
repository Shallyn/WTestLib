#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:36:31 2019

@author: drizl
"""

from ligo.gracedb.rest import GraceDb
from glue.ligolw import ligolw, lsctables
from ..Utils import CEV, WARNING
import sys
from .datasource import gwStrainSRC

class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
        pass

client = GraceDb()

class GraceEvent(object):
    def __init__(self, GraceID, verbose = False):
        self._GraceID = GraceID
        response = client.event(GraceID)
        datatable = response.json()
        self._verbose = verbose
        self._load_coinc(datatable)
        self._load_sngl(datatable)
        
    def _load_coinc(self, table):
        data = table['extra_attributes']['CoincInspiral']
        self._snr = data.pop('snr', 0)
        self._end_time = data.pop('end_time', 0) + 1e-9 * data.pop('end_time_ns', 0)
        self._combined_far = data.pop('combined_far', 0)
        ifos = data.pop('ifos', '')
        if len(ifos) == 0:
            self._ifos = []
        else:
            self._ifos = ifos.split(',')
        
    def _load_sngl(self, table):
        data = table['extra_attributes']['SingleInspiral']
        names = self.__dict__
        for ele in data:
            names[ele['ifo']] = DetTable(ele)
    
    def get_sngl(self, name):
        try:
            ret = self.__dict__[name]
            if isinstance(ret, DetTable):
                return ret
            sys.stderr.write(f'{WARNING}: Invalid name {name}\n')
            return None
        except:
            sys.stderr.write(f'{WARNING}: Invalid name {name}\n')
            return None
    @property
    def end_time(self):
        return self._end_time
    
    @property
    def combined_far(self):
        return self._combined_far
    
    @property
    def snr(self):
        return self._snr
    
    def load_data(self, stepback = 15, stepforward = 15, channel = 'GATED', fs = 4096):
        tstart = self.end_time - abs(stepback)
        tend = self.end_time + abs(stepforward)
        ret = dict()
        for ifo in self._ifos:
            print(len(ifo))
            gws = gwStrainSRC(ifo, tstart, tend, channel = f'{ifo}_{channel}')
            ret[f'{ifo}'] = gws.load_data(fs)
        return ret
        

class DetTable(object):
    def __init__(self, table):
        names = self.__dict__
        for key in table:
            names[key] = table[key]
        self.gps = self.end_time + 1e-9 * self.end_time_ns


class GraceSuperEvent(object):
    def __init__(self, SGraceID, verbose = False):
        self._SGraceID = SGraceID
        response = client.superevent(SGraceID)
        datatable = response.json()
        self._verbose = verbose
        self._load_table(datatable)
        
    def _load_table(self, datatable):
        self._GraceID_list = datatable['gw_events']
        self._GraceID_preferred = datatable['preferred_event']
        self._gps_start = datatable['t_start']
        self._gps = datatable['t_0']
        self._gps_end = datatable['t_end']
        self._far = datatable['far']
        self._GraceEvent = GraceEvent(self._GraceID_preferred, self._verbose)
    
    def __iter__(self):
        for gid in self._GraceID_list:
            yield gid
    
    @property
    def Event(self):
        return self._GraceEvent
    
    @property
    def gps_start(self):
        return self._gps_start
    
    @property
    def gps_trigger(self):
        return self._gps
    
    @property
    def gps_end(self):
        return self._gps_end
    
    @property
    def far(self):
        return self._far
        
