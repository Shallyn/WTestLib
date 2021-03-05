#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:41:46 2019

@author: drizl
"""

import numpy as np

def CombineTPLEQMFits(eta, A1, fEQ, fTPL):
    eta2 = eta*eta
    # Impose that TPL and equal-mass limit are exactly recovered
    A0 = -0.00099601593625498 * A1 - 0.00001600025600409607 * fEQ + 1.000016000256004 * fTPL
    A2 = -3.984063745019967 * A1 + 16.00025600409607 * fEQ - 16.0002560041612 * fTPL
    # Final formula
    return A0 + A1 * eta + A2 * eta2

# chiX = chiS + chiA * (m1 - m2) / (m1 + m2) / (1 - 2*eta)
def calculate_PeakAmp22(eta, chiX):
    chi = chiX
    chi2 = chi*chi
    chi3 = chi2*chi
    # TPL fit
    fTPL = 1.4528573105413543 + 0.16613449160880395 * chi + 0.027355646661735258 * chi2 - 0.020072844926136438 * chi3
    # Equal-mass fit
    fEQ = 1.577457498227 - 0.0076949474494639085 * chi +  0.02188705616693344 * chi2 + 0.023268366492696667 * chi3
    # Global fit coefficients
    e0 = -0.03442402416125921
    e1 = -1.218066264419839
    e2 = -0.5683726304811634
    e3 = 0.4011143761465342
    A1 = e0 + e1 * chi + e2 * chi2 + e3 * chi3
    res = eta * CombineTPLEQMFits(eta, A1, fEQ, fTPL)
    return res

def calculate_PeakAmpDDot22(eta, chiX):
    chi = chiX
    chiMinus1 = -1. + chi
    # TPL fit
    fTPL = 0.002395610769995033 * chiMinus1 -  0.00019273850675004356 * chiMinus1 * chiMinus1 - 0.00029666193167435337 * chiMinus1 * chiMinus1 * chiMinus1
    # Equal-mass fit
    fEQ = -0.004126509071377509 + 0.002223999138735809 * chi
    # Global fit coefficients
    e0 = -0.005776537350356959
    e1 = 0.001030857482885267
    A1 = e0 + e1 * chi
    res = eta * CombineTPLEQMFits(eta, A1, fEQ, fTPL)
    return res

def calculate_PeakOmega22(eta, chiX):
    chi = chiX
    # From TPL fit
    c0 = 0.5626787200433265
    c1 = -0.08706198756945482
    c2 = 25.81979479453255
    c3 = 25.85037751197443
    # From equal-mass fit
    d2 = 7.629921628648589
    d3 = 10.26207326082448
    # Combine TPL and equal-mass
    A4 = d2 + 4 * (d2 - c2) * (eta - 0.25)
    A3 = d3 + 4 * (d3 - c3) * (eta - 0.25)
    c4 = 0.00174345193125868
    # Final formula
    res = c0 + (c1 + c4 * chi) * np.log(A3 - A4 * chi)
    return res

def calculate_PeakOmegaDot22(eta, chiX):
    chi = chiX
    # TPL fit
    fTPL = -0.011209791668428353 +  (0.0040867958978563915 + 0.0006333925136134493 * chi) * np.log(68.47466578100956 - 58.301487557007206 * chi)
    # Equal-mass fit
    fEQ = 0.01128156666995859 + 0.0002869276768158971* chi
    # Global fit coefficients
    e0 = 0.01574321112717377
    e1 = 0.02244178140869133
    A1 = e0 + e1 * chi
    res = CombineTPLEQMFits(eta, A1, fEQ, fTPL)
    return res

def calculate_NRPeakParams(eta, chiX):
    PeakAmp = calculate_PeakAmp22(eta, chiX)
    PeakAmpDot = calculate_PeakAmpDDot22(eta, chiX)
    PeakOmega = calculate_PeakOmega22(eta, chiX)
    PeakOmegaDot = calculate_PeakOmegaDot22(eta, chiX)
    return PeakAmp, PeakAmpDot, PeakOmega, PeakOmegaDot
