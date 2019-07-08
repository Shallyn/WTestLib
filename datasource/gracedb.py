#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:36:31 2019

@author: drizl
"""

from ligo.gracedb.rest import GraceDb
from glue.ligolw import ligolw, lsctables

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
        if ifos == 0:
            self._ifos = []
        else:
            self._ifos = ifos.split(',')
        
    def _load_sngl(self, table):
        data = table['extra_attributes']['SingleInspiral']
        names = self.__dict__
        for ele in data:
            names[ele['ifo']] = DetTable(ele)
            
            

class DetTable(object):
    def __init__(self, table):
        names = self.__dict__
        for key in table:
            names[key] = table[key]