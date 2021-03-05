#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:30:34 2019

@author: drizl
"""

from optparse import OptionParser
from ..Utils import LOG, WARNING, cmd_hang
import sys, time
from pathlib import Path
from .gracedb import get_Sevents_from_time, get_nowtime, GPS2ISO
import os

STEPFWD = 15
STEPBACK = 15
SRATE = 4096
NSIDE = 16
INTERVAL = 5

def GraceDB_Scanner(argv = None):
    parser = OptionParser(description='GraceDB Scanner')
    parser.add_option('--prefix', type = 'str', default = '.', help = 'prefix for data saving')
    parser.add_option('--time-interval', type = float, default = 60, help = 'time interval for each scan')
    parser.add_option('--executable', type = 'str', help = 'executable file')
    parser.add_option('--ref-psd', type = 'str', help = 'reference psd')
    parser.add_option('--log', type = 'str', default = 'log.err', help = 'log file')
    args, empty = parser.parse_args(argv)
    sys.stderr.write(f'{LOG}:Parsing args.\n')
    prefix = Path(args.prefix)
    twait = args.time_interval
    if twait < INTERVAL:
        twait = INTERVAL * 1.5
    exe = args.executable
    flog = Path(args.log)
    
    if exe is None:
        sys.stderr.write(f'{WARNING}:executable must be specified.\n')
        return 0
    exe = Path(exe).absolute()
    if not exe.exists():
        sys.stderr.write(f'{WARNING}:{exe} does not exist.\n')
        return 0
    ref_psd = args.ref_psd
    if ref_psd is None:
        sys.stderr.write(f'{WARNING}:ref-psd must be specified.\n')
        return 0
    
    def fCMD(gid, save):
        savepath = prefix / save
        return f'{exe} --graceid={gid} \
                        --ref-psd={ref_psd} \
                        --prefix={savepath} \
                        --stepback={STEPBACK} \
                        --stepforward={STEPFWD} \
                        --sample-rate={SRATE} \
                        --nside={NSIDE}'
    sys.stderr.write(f'{LOG}:Run scanner...\n')
    # Run
    process = SubprocessHandler(10)
    t_ini = time.time() - twait
    while(1):
        process.checkprocess(flog)
        time.sleep(INTERVAL)
        now = get_nowtime()
        ISO = GPS2ISO(now)
        tscan = time.time() - t_ini
        if tscan < twait:
            continue
        t_ini = time.time()
        start = now - tscan
        Slist = get_Sevents_from_time(start, now)
        if len(Slist) == 0:
            sys.stderr.write(f'{LOG}:{ISO}-No new superevent.\n')
            continue
        for Sevt in Slist:
            Sid = Sevt.SGraceID
            sys.stderr.write(f'{LOG}:{ISO}-New superevent {Sid}\n')
            Gevent = Sevt.Preferred_GraceEvent
            Gid = Gevent.GraceID
            if len(Gevent.ifos) < 2:
                continue
            CMD = fCMD(Gid, f'{Sid}_{Gid}')
            process.createprocess(CMD, f'{Sid}.err', flog)
    return 0


class SubprocessHandler(object):
    def __init__(self, lim = 10):
        self._OBJlist = []
        self._lim = lim
    
    def __iter__(self):
        for obj in self._OBJlist:
            yield obj
    @property
    def num_obj(self):
        return len(self._OBJlist)
            
    def createprocess(self, CMD, ferr, flog = None):
        sys.stderr.write(f'{LOG}:Creating:\n{CMD}\n\n')
        self._OBJlist.append(mysubprocess(CMD, ferr))
        if self.num_obj > self._lim:
            self.remove_oldest_one(flog)
            
    def remove_oldest_one(self, flog = None):
        cost = 0
        for obj in self:
            if obj.cost() > cost:
                cost = obj.cost()
                rem = obj
        rem.shut(flog = flog)
        self._OBJlist.remove(rem)
    
    def checkprocess(self, flog):
        dellist = []
        for obj in self:
            if obj.poll() is not None:
                obj.shut(flog)
                dellist.append(obj)
        for obj in dellist:
            try:
                self._OBJlist.remove(obj)
            except:
                sys.stderr.write(f'{WARNING}:The element need to remove is not in the list.\n')
        sys.stderr.write(f'{LOG}: Now {self.num_obj} subprocess are running.\n')
            
class mysubprocess(object):
    def __init__(self, CMD, ferr):
        obj, ferr = cmd_hang(CMD, ferr)
        self._obj = obj
        self._ferr = Path(ferr)
        self._start_time = time.time()
        self._RUN = True
    
    @property
    def RUN(self):
        return self._RUN
    
    def cost(self):
        return time.time() - self._start_time
    
    def poll(self, **kwargs):
        return self._obj.poll(**kwargs)
    
    def terminate(self, **kwargs):
        return self._obj.terminate(**kwargs)
    
    @property
    def ferr(self):
        return self._ferr
            
    def kill(self, **kwargs):
        return self._obj.kill(**kwargs)
    
    def clean(self, flog = None):
        if flog is not None:
            return os.system(f'cat {self.ferr} >> {flog} && rm {self.ferr}')
        else:
            if self.ferr.exists():
                return os.system(f'rm {self.ferr}')
            return 0
    
    def shut(self, flog = None):
        if self.poll() is None:
            ret = self.terminate()
        else:
            ret = self.poll()
        self.clean(flog = flog)
        self._RUN = False
        return ret
    
    def __del__(self):
        self.terminate()
        self.clean()
        del self._obj
        del self._ferr
    
