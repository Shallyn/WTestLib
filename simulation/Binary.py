#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:41:46 2019

@author: drizl
"""

import numpy as np
from ..h22datatype import dim_t

"""
    In binary frame(bf),
        where ex = n0.
"""
class BinaryParams(object):
    def __init__(self, q, Mtotal, 
                 s1x, s1y, s1z,
                 s2x, s2y, s2z):
        self._q = q
        self._Mtotal = Mtotal
        self._s1hat = np.array([s1x, s1y, s1z])
        self._chi1 = np.linalg.norm(self._s1hat)
        if self._chi1 == 0:
            self._s1hat = np.array([0, 0, 1])
        else:
            self._s1hat = self._s1hat / self._chi1
        self._s2hat = np.array([s2x, s2y, s2z])
        self._chi2 = np.linalg.norm(self._s2hat)
        if self._chi2 == 0:
            self._s2hat = np.array([0, 0, 1])
        else:
            self._s2hat = self._s2hat / self._chi2

    @property
    def m1(self):
        return self._q * self._Mtotal / (1+self._q)
    
    @property
    def m2(self):
        return self._Mtotal / (1+self._q)

    @property
    def beta(self):
        return self._chi1 * (113*np.power(self.m1/self._Mtotal, 2) + 75*self.eta) * self.spin1hat[2] + \
            self._chi2 * (113*np.power(self.m2/self._Mtotal, 2) + 75*self.eta) * self.spin2hat[2]

    @property
    def sigma(self):
        s1 = self.spin1hat
        s2 = self.spin2hat
        s1dots2 = np.dot(s1, s2)
        return self.eta*self._chi1*self._chi2*(-247*s1dots2 + 721*s1[2] * s2[2])

    @property
    def spin1hat(self):
        return self._s1hat
    
    @property
    def spin2hat(self):
        return self._s2hat

    @property
    def spin1VecOverM2(self):
        return self._chi1 * self._s1hat

    @property
    def spin2VecOverM2(self):
        return self._chi2 * self._s2hat

    @property
    def spin1Vec(self):
        return self._chi1 * np.power(self.m1, 2) * self._s1hat

    @property
    def spin2Vec(self):
        return self._chi2 * np.power(self.m2, 2) * self._s2hat

    @property
    def spin1VecOverMt2(self):
        return self.spin1Vec / np.power(self._Mtotal, 2)
    
    @property
    def spin2VecOverMt2(self):
        return self.spin2Vec / np.power(self._Mtotal, 2)

    @property
    def eta(self):
        return self._q / (1 + self._q) / (1 + self._q)

    @property
    def MChirp(self):
        return np.power(self.eta, 3./5.) * self._Mtotal

    @property
    def Mtotal_t(self):
        return dim_t(self._Mtotal)
    
    @property
    def MChirp_t(self):
        return dim_t(self.MChirp)

"""
    solar barycenter frame(sf)
"""
class BinaryFrame(object):
    def __init__(self, thN, phN, thL, phL, phiorb):
        self._thN = thN
        self._phN = phN
        self._thL = thL
        self._phL = phL
        self._phiorb = phiorb

    @property
    def xVec_sf(self):
        xsfx = np.sin(self._phL)*np.sin(self._thL)
        xsfy = -np.cos(self._phL)*np.sin(self._thL)
        xsfz = 0
        xsfnorm = np.sqrt(xsfx*xsfx + xsfy*xsfy + xsfz*xsfz)
        return np.array([xsfx, xsfy, xsfz]) / xsfnorm

    @property
    def yVec_sf(self):
        ysfx = np.cos(self._phL) * np.cos(self._thL) * np.sin(self._thL)
        ysfy = np.cos(self._thL) * np.sin(self._phL) * np.sin(self._thL)
        ysfz = -np.sin(self._thL) * np.sin(self._thL)
        ysfnorm = np.sqrt(ysfx*ysfx + ysfy*ysfy + ysfz*ysfz)
        return np.array([ysfx, ysfy, ysfz]) / ysfnorm

    @property
    def zVec_sf(self):
        return self.LVec

    @property
    def rVec(self):
        return self.xVec_sf * np.cos(self._phiorb) + self.yVec_sf * np.sin(self._phiorb)
    
    @property
    def vVec(self):
        vx, vy, vz = np.cross(self.LVec, self.rVec).tolist()
        v = np.sqrt(vx*vx + vy*vy + vz*vz)
        return np.array([vx, vy, vz]) / v

    @property
    def RotMatrix_sf2bf(self):
        Lx, Ly, Lz = self.LVec.tolist()
        rx, ry, rz = self.rVec.tolist()
        vx, vy, vz = self.vVec.tolist()
        alpha = np.math.atan2(Lx, -Ly)
        beta = np.math.atan2(np.sqrt(Lx*Lx + Ly*Ly), Lz)
        gamma = np.math.atan2(rz, vz)
        cosa = np.cos(alpha)
        cosa = 0 if abs(cosa) < 1e-16 else cosa
        sina = np.sin(alpha)
        sina = 0 if abs(sina) < 1e-16 else sina
        cosb = np.cos(beta)
        cosb = 0 if abs(cosb) < 1e-16 else cosb
        sinb = np.sin(beta)
        sinb = 0 if abs(sinb) < 1e-16 else sinb
        cosg = np.cos(gamma)
        cosg = 0 if abs(cosg) < 1e-16 else cosg
        sing = np.sin(gamma)
        sing = 0 if abs(sing) < 1e-16 else sing
        rotMatrix = np.zeros([3,3])
        rotMatrix[0,0] = cosg*cosa - cosb*sina*sing
        rotMatrix[0,1] = cosg*sina + cosb*cosa*sing
        rotMatrix[0,2] = sing*sinb
        rotMatrix[1,0] = -sing*cosa - cosb*sina*cosg
        rotMatrix[1,1] = -sing*sina + cosb*cosa*cosg
        rotMatrix[1,2] = cosg*sinb
        rotMatrix[2,0] = sinb*sina
        rotMatrix[2,1] = -sinb*cosa
        rotMatrix[2,2] = cosb
        return rotMatrix

    @property
    def RotMatrix_bf2sf(self):
        return self.RotMatrix_sf2bf.T

    @property
    def inclination(self):
        return np.arccos(self.LDotN)

    @property
    def NVec(self):
        return np.array([np.sin(self._thN)*np.cos(self._phN), np.sin(self._thN)*np.sin(self._phN), np.cos(self._thN)])
    
    @property
    def LVec(self):
        return np.array([np.sin(self._thL)*np.cos(self._phL), np.sin(self._thL)*np.sin(self._phL), np.cos(self._thL)])

    @property
    def LDotN(self):
        return np.cos(self._thL) * np.cos(self._thN) + np.sin(self._thL) * np.sin(self._thN) * np.cos(self._phL - self._phN)
    
    @property
    def NDotL(self):
        return self.LDotN

    @property
    def pVec(self):
        thN = self._thN
        thL = self._thL
        phN = self._phN
        phL = self._phL
        px = -(np.cos(thN)*np.sin(phL)*np.sin(thL)) + np.cos(thL)*np.sin(phN)*np.sin(thN)
        py = np.cos(phL)*np.cos(thN)*np.sin(thL) - np.cos(phN)*np.cos(thL)*np.sin(thN)
        pz = np.sin(phL - phN)*np.sin(thL)*np.sin(thN)
        pNorm = np.sqrt(px*px + py*py + pz*pz)
        return np.array([px, py, pz]) / pNorm
    
    @property
    def qVec(self):
        thN = self._thN
        # thL = self._thL
        phN = self._phN
        # phL = self._phL
        px, py, pz = self.pVec.tolist()
        qx = py*np.cos(thN) - pz*np.sin(phN) *np.sin(thN)
        qy = -px*np.cos(thN) + pz*np.cos(phN)*np.sin(thN)
        qz = (-py * np.cos(phN) + px*np.sin(phN))*np.sin(thN)
        return np.array([qx, qy, qz])


    
"""
    fig.1 of prd 81 064008
    solar barycenter frame
"""
class SpaceDetFrame(BinaryFrame):
    def __init__(self, inc, phit, thN, phN, thL, phL):
        super(SpaceDetFrame, self).__init__(thN, phN, thL, phL)
        self._det_inc = inc
        self._det_phit = phit

    def set_phit(self, phit):
        self._det_phit = phit
    
    @property
    def det_zVec(self):
        det_z_x = np.sin(self._det_inc)*np.cos(self._det_phit)
        det_z_y = np.sin(self._det_inc)*np.sin(self._det_phit)
        det_z_z = np.cos(self._det_inc)
        return det_z_x, det_z_y, det_z_z
    
    @property
    def zDotN(self):
        return np.cos(self._det_inc) * np.cos(self._thN) + np.sin(self._det_inc) * np.sin(self._thN) * np.cos(self._det_phit - self._phN)
    
    @property
    def zDotL(self):
        return np.cos(self._det_inc) * np.cos(self._thL) + np.sin(self._det_inc) * np.sin(self._thL) * np.cos(self._det_phit - self._phL)