#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:44:16 2019

@author: drizl
"""
import matplotlib as mlb
mlb.use('Agg')
from optparse import OptionParser

from .signal import sngl_load_file, whiten, get_psdfun, plotit
from .strain import gwStrain
from .template import template
from .detectors import Detector
import numpy as np
import time, sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from ..coherent.core import utdk_times, cohTF, snr_cohTF
from ..coherent.pix import nside2npix, pix2ang
from ..coherent.skymap import mollview, graticule, MollweideProj
from pathlib import Path
from .snr_qTansform import snr_q_scanf
from .detectors import time_delay
from ..Utils import WARNING, LOG, DEBUG
from ..datasource.gracedb import GraceEvent, GraceSuperEvent
from scipy.interpolate import interp1d
from ..generator import dim_t

DEFAULT_NSIDE = 32
DEFAULT_QRANGE = (7,8)
DEFAULT_FRANGE = (30, 1200)
DEFAULT_PCOLORBINS = 100
DEFAULT_CMAP = 'jet'
DEFAULT_MISMATCH = 0.1
FIGSIZE_QSCAN = (14,6)

def parseargs(argv):
    parser = OptionParser(description='Waveform Comparation With SXS')
    parser.add_option('--ra', type = 'float', help = 'ra of this event, if added, will use this value.')
    parser.add_option('--de', type = 'float', help = 'dec of this event, if added, will use this value.')
    
    parser.add_option('--Sgraceid', type = 'str', help = 'GraceDB Super Event ID, if added, will load the Preferred event parameters.')
    parser.add_option('--graceid', type = 'str', help = 'GraceDB Event ID, if added, will load such Grace event parameters(Prefer).')
    parser.add_option('--stepback', type = 'int', default = 15, help = 'Used for GraceDB data load.')
    parser.add_option('--stepforward', type = 'int', default = 15, help = 'Used for GraceDB data load.')
    
    parser.add_option('--gps', type = 'float', help = 'gps trigger time for this event.')
    parser.add_option('--sample-rate', type = 'int', default = 4096, help = 'sample rate used.')
    
    parser.add_option('--m1', type = 'float', help = 'mass1 of this event, for template generation.')
    parser.add_option('--m2', type = 'float', help = 'mass2 of this event, for template generation.')
    parser.add_option('--s1z', type = 'float', help = 'spin1z of this event, for template generation.')
    parser.add_option('--s2z', type = 'float', help = 'spin2z of this event, for template generation.')
    parser.add_option('--fini', type = 'float', default = 0.003, help = 'Initial frequency for template generation(natual dimension).')
    parser.add_option('--approx', type = 'str', help = 'approx for template generation.')
    
    parser.add_option('--nside', type = 'int', default = DEFAULT_NSIDE, help = 'Nside for skymap pix.')
    parser.add_option('--minq', type = 'int', default = DEFAULT_QRANGE[0], help = 'min Q.')
    parser.add_option('--maxq', type = 'int', default = DEFAULT_QRANGE[1], help = 'max Q.')
    parser.add_option('--minf', type = 'float', default = DEFAULT_FRANGE[0], help = 'min Frequency.')
    parser.add_option('--maxf', type = 'float', default = DEFAULT_FRANGE[1], help = 'max Frequency.')
    parser.add_option('--mismatch', type = 'float', default = DEFAULT_MISMATCH, help = 'mismatch for qscan.')
    parser.add_option('--cmap', type = 'str', default = DEFAULT_CMAP, help = 'Plot color map type.')
    parser.add_option('--pcolorbins', type = 'int', default = DEFAULT_PCOLORBINS, help = 'color bins for pcolor mesh plot.')
    
    parser.add_option('--datadir', type = 'str', help = 'dir for data saving.')
    parser.add_option('--prefix', type = 'str', default = '.', help = 'prefix for results saving.')
    parser.add_option('--ref', type = 'str', help = 'prefix for reference psd.')
    parser.add_option('--ref-psd', type = 'str', help = 'prefix for reference psd, preferred.')
    parser.add_option('--channel', type = 'str', default = 'GATED', help = 'channel type, if local data used.')
    
    parser.add_option('--track', action = 'store_true', help = 'If added, will plot track.')
    args = parser.parse_args(argv)
    return args

def get_proper_approx(m1, m2):
    if m1 + m2 > 6:
        return 'SEOBNRv4'
    else:
        return 'SpinTaylorT4'

def main(argv = None):
    # Step.1 parse args...
    args, empty = parseargs(argv)
    ra = args.ra
    de = args.de
    
    gps = args.gps
    Sgraceid = args.Sgraceid
    graceid = args.graceid
    sback = args.stepback
    sfwd = args.stepforward
    fs = args.sample_rate
    
    m1 = args.m1
    m2 = args.m2
    s1z = args.s1z
    s2z = args.s2z
    fini = args.fini
    approx = args.approx
    
    nside = args.nside
    qrange = (args.minq, args.maxq)
    frange = (args.minf, args.maxf)
    mismatch = args.mismatch
    cmaptype = args.cmap
    pcolorbins = args.pcolorbins
    
    datadir = args.datadir
    prefix = args.prefix
    ref = args.ref
    refpsd = args.ref_psd
    channel = args.channel
    
    if refpsd is not None:
        fdict_refpsd = sngl_load_file(refpsd, channel)

    track = args.track
    # Step.2 load data...
    
    if graceid is None and Sgraceid is None:
        if datadir is None or \
            m1 is None or \
            m2 is None or\
            s1z is None or \
            s2z is None:
            sys.stderr.write(f'{WARNING}:Input parameters is insufficient, exit.\n')
            return -1
        fdict = sngl_load_file(datadir, channel)
        if ref is None:
            fdict_ref = sngl_load_file(datadir, channel)
        else:
            fdict_ref = sngl_load_file(ref, channel)
        
        # load data, make gwStrain, set psd.
        for ifo in ['H1', 'L1', 'V1']:
            if ifo not in fdict:
                locals()[f's{ifo}'] = None
            else:
                src = np.loadtxt(fdict[ifo])
                datatime = src[:,0]
                fs_new = int(1./(datatime[1] - datatime[0]))
                if fs_new != fs:
                    resample = True
                else:
                    resample = False
                if refpsd is not None:
                    datapsd = np.loadtxt(fdict_refpsd[ifo])
                    psd = interp1d(datapsd[0,:], datapsd[1,:])
                else:
                    refdata = np.loadtxt(fdict_ref[ifo])
                    psd = get_psdfun(refdata[:,1], fs = fs)
                if np.max(src[:,1]) > 0:
                    locals()[f's{ifo}'] = gwStrain(src[:,1], epoch = src[:,0][0], ifo = ifo, fs = fs)
                    if resample:
                        locals()[f's{ifo}'] = locals()[f's{ifo}'].resample(fs)
                    locals()[f's{ifo}'].set_psd(psd)
    else:
        if refpsd is None:
            sys.stderr.write(f'{WARNING}:No ref psd, exit.\n')
            return -1;
        sys.stderr.write(f'{LOG}:Parse GraceID...\n')
        try:
            if graceid is not None:
                Gevt = GraceEvent(GraceID=graceid, verbose = True)
            else:
                Gevt = GraceSuperEvent(SGraceID=Sgraceid, verbose = True).Preferred_GraceEvent
            sngl = Gevt.get_sngl('L1')
            m1 = sngl.mass1
            m2 = sngl.mass2
            s1z = sngl.spin1z
            s2z = sngl.spin2z
            gps = Gevt.end_time
            sys.stderr.write(f'{LOG}:Parameters:\n\t\
                             m1 = {m1}\n\t\
                             m2 = {m2}\n\t\
                             s1z = {s1z}\n\t\
                             s2z = {s2z}\n\t\
                             gps end time: {gps}\n')
        except:
            sys.stderr.write(f'{WARNING}:Failed to parse GraceEvent, exit\n')
            return -1
        sys.stderr.write(f'{LOG}:Loading data...\n')
        try:
            datadict = Gevt.load_data(stepback = sback, stepforward = sfwd, channel = channel, fs = fs)
            for key in datadict:
                tmp = datadict[key]
                sys.stderr.write(f'{LOG}:Load {key} data, duration = {tmp.duration}\n')
                locals()[f's{key}'] = tmp
        except:
            sys.stderr.write(f'{WARNING}:Failed to load data, exit\n')
            return -1
        
        for ifo in datadict:
            if refpsd is not None:
                datapsd = np.loadtxt(fdict_refpsd[ifo])
                psd = interp1d(datapsd[0,:], datapsd[1,:])
            else:
                refdata = np.loadtxt(fdict_ref[ifo])
                psd = get_psdfun(refdata[:,1], fs = fs)
            locals()[f's{ifo}'].set_psd(psd)
    
    # reset approx and initial frequency for waveform generation
    if approx is None:
        approx = get_proper_approx(m1, m2)
    fini = fini * dim_t(m1 + m2)
    
    for sifo in ['sH1', 'sL1', 'sV1']:
        if sifo not in locals():
            locals()[sifo] = None
    # Step.3 Call....
    return event_scan(gps = gps,
                      sH1 = locals()['sH1'],
                      sL1 = locals()['sL1'],
                      sV1 = locals()['sV1'],
                      m1 = m1, m2 = m2,
                      s1z = s1z, s2z = s2z,
                      fini = fini, approx = approx, fs = fs,
                      prefix = prefix,
                      ra = ra, de = de,
                      track = track,
                      nside = nside,
                      qrange = qrange,
                      frange = frange,
                      pcolorbins = pcolorbins,
                      cmaptype = cmaptype,
                      mismatch = mismatch)

def event_scan(gps, sH1, sL1, sV1,
               m1, m2, s1z, s2z, fini, approx, fs,
               prefix = '.', ra = None, de = None,
               track = False,
               nside = DEFAULT_NSIDE,
               qrange = DEFAULT_QRANGE,
               frange = DEFAULT_FRANGE,
               pcolorbins = DEFAULT_PCOLORBINS,
               cmaptype = DEFAULT_CMAP,
               mismatch = DEFAULT_MISMATCH):
    # Step.0 fsave prefix setting.
    fsave = Path(prefix)
    if not fsave.exists():
        fsave.mkdir(parents=True)
    
    # Step.1 Making Template    
    tmpl = template(m1 = m1,
                    m2 = m2,
                    s1z = s1z,
                    s2z = s2z,
                    fini = fini,
                    approx = approx,
                    srate = fs,
                    D = 100,
                    duration = sH1.duration/3)
    track_x, track_y = tmpl.get_track(gps)
    tmpl.plot(fsave = fsave / 'template.png', 
              title = 'template',
              figsize = (12,5))
    # Step.2 Matched filtering
    tmpmax = 0
    snrLIST = []
    sLIST = []
    psd_freqs = np.logspace(np.log10(1), np.log10(1500), 500)
    for strain in [sH1, sL1, sV1]:
        if strain is not None:
            strain.plot(fsave = fsave / f'data_{strain.ifo}.png', 
                        title = f'{strain.ifo} data',
                        ylabel = 'strain',
                        figsize = (12, 5))
            # plot psd
            psd_data = strain.psdfun_setted(psd_freqs)
            plt.figure(figsize = (8,6))
            plt.plot(psd_freqs, psd_data, color = 'black')
            plt.xlabel('freq [Hz]')
            plt.ylabel('strain [$Hz^-2$]')
            plt.yscale('log')
            plt.title(f'psd {strain.ifo}')
            plt.savefig(fsave / f'psd_{strain.ifo}.png', dpi = 200)
            plt.close()
            
            sLIST.append(strain)
            SNR = \
            strain.matched_filter(tmpl.template, 
                                  cut = [30,1000], 
                                  window = True, 
                                  psd = 'set', 
                                  ret_complex = True, 
                                  shift = tmpl.dtpeak)
            index_gps = slice( int(((gps-0.5) - SNR.epoch) * SNR.fs) ,\
                               int(((gps+0.5) - SNR.epoch) * SNR.fs))

            snrLIST.append(SNR)
            sys.stderr.write(f'{DEBUG}: tSNR {SNR.ifo} = {SNR.time[0]}, {SNR.time[-1]}\n')
            if max(SNR.value[index_gps]) > tmpmax:
                tmpmax = max(SNR.value[index_gps])
                tmap = SNR.time[np.argmax(np.abs(SNR.value[index_gps]))]
    
    # Step.3 Plot setting.
    # sys.stderr.write(f'{DEBUG}:tmap = {tmap}\n')
    tpeak = gps - tmpl.dtpeak
    h_dur = min(0.5, tmpl.dtpeak)
    tlim = [tmap - h_dur, tmap + h_dur]
    tlim2 = [tmap - h_dur*2, tmap + h_dur*2]
    tlim3 = [tmap - h_dur*3, tmap + h_dur*3]
    cmap = plt.get_cmap(cmaptype)
    tsnr = np.linspace(tlim3[0], tlim3[1], 1500)
    fout = np.logspace(np.log10(30), np.log10(1000), 600)

    # sys.stderr.write(f'{DEBUG}: tlim3 = {tlim3}\n')
    # sys.stderr.write(f'{DEBUG}: tpeak = {tmpl.dtpeak}\n')
    # sys.stderr.write(f'{DEBUG}: tmap = {tmap}\n')
    
    # Step.4 Plot SNR time series.
    for SNR in snrLIST:
        SNR.plot(xrange = tlim, title = f'SNR {SNR.ifo}', fsave = fsave / f'SNR_{SNR.ifo}_zoom.png')
        SNR.plot(xrange = None, title = f'SNR {SNR.ifo}', fsave = fsave / f'SNR_{SNR.ifo}.png')
        #sys.stderr.write(f'{DEBUG}: snr {SNR.ifo} epoch = {SNR.epoch}\n')
    
    # Step.5 Plot snr q scan spectrum.
    for data in sLIST:
        func, xout, yout = snr_q_scanf(data.value, tmpl.template, 
                            data.fs, data.epoch + tmpl.dtpeak, 
                            cut=None, psd = data.psdfun_setted,
                            qrange = qrange, retfunc = True, window = True)
        #sys.stderr.write(f'{LOG}:xout = {xout}\n')
        Eng = np.abs(func(tsnr, fout))
        
        levels = MaxNLocator(nbins=pcolorbins).tick_values(Eng.min(), Eng.max())
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        
        idx_tpeak, idx_fpeak = get_2D_argpeak(Eng.T)
        tpeak = '%.2f'%tsnr[idx_tpeak]
        fpeak = '%.1f'%fout[idx_fpeak]
        snrpeak = '%.3f'%Eng.T[idx_tpeak, idx_fpeak]
        label = f'loudest snr = {snrpeak}, at t = {tpeak}, f = {fpeak}'
        
        fig = plt.figure(figsize = FIGSIZE_QSCAN)
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(tsnr, fout, Eng, cmap = cmap, norm = norm)
        fig.colorbar(im, ax=ax)
        plt.title(f'{data.ifo} snr Qscan')
        if track and ra is not None and de is not None:
            delay = time_delay(data.ifo, ra, de, tsnr[int(tsnr.size / 2)])
            plt.plot(track_x + delay, track_y, '-', color='#ba7b00', zorder=3, lw=1.5)
        plt.xlabel(label)
        plt.ylabel('frequency')
        plt.ylim([30, 1000])
        plt.yscale('log')
        plt.xlim(tlim3)
        plt.savefig(fsave/f'Qscan_{data.ifo}.png', dpi = 200)
        plt.close()
        
        plot_wscan(tsnr, fout, Eng, 
                    cmap = cmap, norm = norm, 
                    figsize = FIGSIZE_QSCAN, 
                    xlabel = label, ylabel = 'frequency', 
                    xlim = tlim2, ylim = [30, 1000], 
                    fsave = fsave/f'Qscan_{data.ifo}_zoom1.png', 
                    title = f'{data.ifo} snr Qscan')

        plot_wscan(tsnr, fout, Eng, 
                    cmap = cmap, norm = norm, 
                    figsize = FIGSIZE_QSCAN, 
                    xlabel = label, ylabel = 'frequency', 
                    xlim = tlim, ylim = [30, 1000], 
                    fsave = fsave/f'Qscan_{data.ifo}_zoom2.png', 
                    title = f'{data.ifo} snr Qscan')

    #
    #
    # Step.6 Plot coherent skymap
    #
    #
    npix=nside2npix(nside) # 12 * npix * npix
    theta,phi = pix2ang(nside,np.arange(npix))
    ra_pix = phi-np.pi
    de_pix = -theta+np.pi/2
    
    ntime = 100
    ndet = len(snrLIST)
    det_time = np.linspace(tlim[0],tlim[1],ntime)
    umx = utdk_times(snrLIST, ra_pix, de_pix, det_time, verbose = True)
    utdka2 = np.multiply(umx,umx.conj()).real
    null = np.sum(utdka2[:,:,2:],axis=2)
    LLR = utdka2[:,:,0] + utdka2[:,:,1] # LOG(likelihood ratio) (4)
    coh_snr2 = LLR.max(axis=0)
    #coh_snr2 = LLR[int(det_time.size / 2),:]
    
    projector  = MollweideProj()
    max_de,max_ra = pix2ang(nside,np.argmax(coh_snr2))
    max_de = max_de[0]
    max_ra = max_ra[0]
    x1,y1 = projector.ang2xy(np.array([max_de, max_ra]))
    
    #plot
    mollview(np.sqrt(coh_snr2),title='Coherent SNR')
    graticule(coord='G',local=True)
    plt.plot(x1,y1,'rx')
    if ra is not None and de is not None:
        x2,y2 = projector.ang2xy(np.array([np.pi/2 - de, ra + np.pi]))
        plt.plot(x2,y2,'r+')
    plt.savefig(fsave/'Coherent_Skymap.png', dpi = 200)
    plt.show()
    
    #
    #
    # Step.7 Plot coherent snr q scan spectrum
    #
    #
    if ra is not None and de is not None:
        max_de = np.pi / 2 - de
        max_ra = ra - np.pi
    else:
        max_de,max_ra = pix2ang(nside,np.argmax(coh_snr2))
        max_ra = max_ra[0] - np.pi
        max_de = np.pi/2 - max_de[0]
        
    tout = np.linspace(tlim3[0], tlim3[1], 1500)
    fout = np.logspace(np.log10(30), np.log10(1000), 500)
    coh_matrix = snr_cohTF(sLIST, max_ra, max_de, 0, 
                           tout, fout, 
                           tmpl = tmpl.template, verbose = True, 
                           qrange = qrange, frange = frange, 
                           mismatch = mismatch, shift = tmpl.dtpeak)
    ucoh2 = np.multiply(coh_matrix, coh_matrix.conjugate()).real
    coh_oscan = np.sqrt(ucoh2[:,:,0] + ucoh2[:,:,1])
    coh_oscan_01 = np.sqrt(ucoh2[:,:,0])
    coh_oscan_02 = np.sqrt(ucoh2[:,:,1])
    null_oscan = np.sqrt(np.sum(ucoh2[:,:,2:], axis = 2))
    
    #plot
    levels = MaxNLocator(nbins=pcolorbins).tick_values(coh_oscan.min(), coh_oscan.max())
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    # coh_SNRscan
    idx_tpeak_0, idx_fpeak_0 = get_2D_argpeak(coh_oscan)
    tpeak = '%.2f'%tout[idx_tpeak_0]
    fpeak = '%.1f'%fout[idx_fpeak_0]
    snrpeak = '%.3f'%coh_oscan[idx_tpeak_0, idx_fpeak_0]
    label = f'loudest snr = {snrpeak}, at t = {tpeak}, f = {fpeak}'
    plot_wscan(tout, fout, coh_oscan.T, 
               cmap = cmap, norm = norm, 
               figsize = FIGSIZE_QSCAN, 
               xlabel = label, ylabel = 'frequency', 
               xlim = tlim3, ylim = [30,750], 
               fsave = fsave/'snrQscan_coh.png', 
               title = 'Coherent SNR wscan')
    
    plot_wscan(tout, fout, coh_oscan.T, 
               cmap = cmap, norm = norm, 
               figsize = FIGSIZE_QSCAN, 
               xlabel = label, ylabel = 'frequency', 
               xlim = tlim2, ylim = [30,750], 
               fsave = fsave/'snrQscan_coh_zoom1.png', 
               title = 'Coherent SNR wscan')

    plot_wscan(tout, fout, coh_oscan.T, 
               cmap = cmap, norm = norm, 
               figsize = FIGSIZE_QSCAN, 
               xlabel = label, ylabel = 'frequency', 
               xlim = tlim, ylim = [30,750], 
               fsave = fsave/'snrQscan_coh_zoom2.png', 
               title = 'Coherent SNR wscan')

    idx_tpeak, idx_fpeak = get_2D_argpeak(coh_oscan_01)
    tpeak = '%.2f'%tout[idx_tpeak]
    fpeak = '%.1f'%fout[idx_fpeak]
    snrpeak = '%.3f'%coh_oscan_01[idx_tpeak, idx_fpeak]
    label = f'loudest snr = {snrpeak}, at t = {tpeak}, f = {fpeak}'

    # coh_SNRscan1
    plot_wscan(tout, fout, coh_oscan_01.T, 
               cmap = cmap, norm = norm, 
               figsize = FIGSIZE_QSCAN, 
               xlabel = label, ylabel = 'frequency', 
               xlim = tlim3, ylim = [30,750], 
               fsave = fsave/'snrQscan_coh_01.png', 
               title = 'Coherent SNR wscan stream 01')
    
    plot_wscan(tout, fout, coh_oscan_01.T, 
               cmap = cmap, norm = norm, 
               figsize = FIGSIZE_QSCAN, 
               xlabel = label, ylabel = 'frequency', 
               xlim = tlim2, ylim = [30,750], 
               fsave = fsave/'snrQscan_coh_01_zoom1.png', 
               title = 'Coherent SNR wscan stream 01')

    plot_wscan(tout, fout, coh_oscan_01.T, 
               cmap = cmap, norm = norm, 
               figsize = FIGSIZE_QSCAN, 
               xlabel = label, ylabel = 'frequency', 
               xlim = tlim, ylim = [30,750], 
               fsave = fsave/'snrQscan_coh_01_zoom2.png', 
               title = 'Coherent SNR wscan stream 01')
    
    idx_tpeak, idx_fpeak = get_2D_argpeak(coh_oscan_02)
    tpeak = '%.2f'%tout[idx_tpeak]
    fpeak = '%.1f'%fout[idx_fpeak]
    snrpeak = '%.3f'%coh_oscan_02[idx_tpeak, idx_fpeak]
    label = f'loudest snr = {snrpeak}, at t = {tpeak}, f = {fpeak}'

    # coh_SNRscan2
    plot_wscan(tout, fout, coh_oscan_02.T, 
               cmap = cmap, norm = norm, 
               figsize = FIGSIZE_QSCAN, 
               xlabel = label, ylabel = 'frequency', 
               xlim = tlim3, ylim = [30,750], 
               fsave = fsave/'snrQscan_coh_02.png', 
               title = 'Coherent SNR wscan stream 02')
    
    plot_wscan(tout, fout, coh_oscan_02.T, 
               cmap = cmap, norm = norm, 
               figsize = FIGSIZE_QSCAN, 
               xlabel = label, ylabel = 'frequency', 
               xlim = tlim2, ylim = [30,750], 
               fsave = fsave/'snrQscan_coh_02_zoom1.png', 
               title = 'Coherent SNR wscan stream 02')

    plot_wscan(tout, fout, coh_oscan_02.T, 
               cmap = cmap, norm = norm, 
               figsize = FIGSIZE_QSCAN, 
               xlabel = label, ylabel = 'frequency', 
               xlim = tlim, ylim = [30,750], 
               fsave = fsave/'snrQscan_coh_02_zoom2.png', 
               title = 'Coherent SNR wscan stream 02')

    nullsnr = null_oscan[idx_tpeak_0, idx_fpeak_0]
    tpeak = '%.2f'%tout[idx_tpeak_0]
    fpeak = '%.1f'%fout[idx_fpeak_0]
    label = f'null snr = {nullsnr}, at t = {tpeak}, f = {fpeak}'

    # null
    plot_wscan(tout, fout, null_oscan.T, 
               cmap = cmap, norm = norm, 
               figsize = FIGSIZE_QSCAN, 
               xlabel = label, ylabel = 'frequency', 
               xlim = tlim3, ylim = [30,750], 
               fsave = fsave/'null_oscan.png', 
               title = 'NULL SNR wscan stream')
    
    plot_wscan(tout, fout, null_oscan.T, 
               cmap = cmap, norm = norm, 
               figsize = FIGSIZE_QSCAN, 
               xlabel = label, ylabel = 'frequency', 
               xlim = tlim2, ylim = [30,750], 
               fsave = fsave/'null_oscan_zoom1.png', 
               title = 'NULL SNR wscan stream')

    plot_wscan(tout, fout, null_oscan.T, 
               cmap = cmap, norm = norm, 
               figsize = FIGSIZE_QSCAN, 
               xlabel = label, ylabel = 'frequency', 
               xlim = tlim, ylim = [30,750], 
               fsave = fsave/'null_oscan_zoom2.png', 
               title = 'NULL SNR wscan stream')

    
    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(111)
    # im = ax.pcolormesh(tout, fout, coh_oscan.T, cmap = cmap, norm = norm)
    # fig.colorbar(im, ax=ax)
    # plt.title('Coherent SNR wscan')
    # plt.xlabel('gps time')
    # plt.ylabel('frequency')
    # plt.ylim([30, 500])
    # plt.xlim(tlim)
    # plt.yscale('log')
    # plt.savefig(fsave/'snrQscan_coh.png',dpi = 200)
    # plt.show()
    
    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(111)
    # im = ax.pcolormesh(tout, fout, coh_oscan_01.T, cmap = cmap, norm = norm)
    # fig.colorbar(im, ax=ax)
    # plt.title('Coherent SNR wscan stream 01')
    # plt.xlabel('gps time')
    # plt.ylabel('frequency')
    # plt.ylim([30, 500])
    # plt.xlim(tlim)
    # plt.yscale('log')
    # plt.savefig(fsave/'snrQscan_coh_01.png',dpi = 200)
    # plt.show()
    
    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(111)
    # im = ax.pcolormesh(tout, fout, coh_oscan_02.T, cmap = cmap, norm = norm)
    # fig.colorbar(im, ax=ax)
    # plt.title('Coherent SNR wscan stream 02')
    # plt.xlabel('gps time')
    # plt.ylabel('frequency')
    # plt.ylim([30, 500])
    # plt.xlim(tlim)
    # plt.yscale('log')
    # plt.savefig(fsave/'snrQscan_coh_02.png',dpi = 200)
    # plt.show()

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(111)
    # im = ax.pcolormesh(tout, fout, null_oscan.T, cmap = cmap, norm = norm)
    # fig.colorbar(im, ax=ax)
    # plt.title('NULL SNR wscan')
    # plt.xlabel('gps time')
    # plt.ylabel('frequency')
    # plt.ylim([30, 500])
    # plt.xlim(tlim)
    # plt.yscale('log')
    # plt.savefig(fsave/'snrQscan_null.png',dpi = 200)
    # plt.show()
    
    return 0
        
def plot_wscan(x, y, z, cmap, norm,
               figsize, xlabel, ylabel,
               xlim, ylim, 
               fsave, title):
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(x, y, z, cmap = cmap, norm = norm)
    fig.colorbar(im, ax=ax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.yscale('log')
    plt.savefig(fsave ,dpi = 200)
    plt.close()

def get_2D_argpeak(matrix):
    arg = np.where(matrix == np.max(matrix))
    return arg[0][0], arg[1][0]
