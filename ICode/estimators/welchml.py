"""
This module have been implemented to compute an Hurst exponent
Estimator using maximum of likelyhood on Welch periodogram.
It is not a log regression as in Hest_Welp
the welch periodogram is the one given by the scipy library
these functions hab been made according to whittlenew,
wittlefunc and fspecFGN of Matlab Biyu_code
"""

import numpy as np
import scipy.optimize as so
from scipy.signal import welch
from math import gamma


def welch_ml(data):
    """This function compute the Hurst exponent of a signal using
    a maximum of likelihood on welch periodogram
    """
    frq, pwr = welch(data)
    nhalfm = len(frq)
    func = lambda H: whittlefunc(H, pwr[1:nhalfm], 2 * (nhalfm))
    return so.fminbound(func, 0, 10)


def welch_ml_s(data, idx):
    """This function compute the Hurst exponent of a signal data[idx,:]
    using a maximum of likelihood on welch periodogram
    """
    frq, pwr = welch(data[idx, :])
    nhalfm = len(frq)
    func = lambda H: whittlefunc(H, pwr[1:nhalfm], 2 * (nhalfm))
    return so.fminbound(func, 0, 1)


def welch_ml_norm_s(data, idx):
    """This function compute the Hurst exponent of a signal data[idx,:]
    using a maximum of likelihood on welch periodogram
    the data are normalized which does'nt change anything !
    """
    datanorm = (data - np.mean(data)) / np.var(data)
    frq, pwr = welch(datanorm[idx, :])
    nhalfm = len(frq)
    func = lambda H: whittlefunc(H, pwr[1:nhalfm], 2 * (nhalfm))
    return so.fminbound(func, 0, 1)


def whittlefunc(hurst, gammahat, nbpoints):
    """This is the Whittle function
    """
    gammatheo = fspec_fgn(hurst, nbpoints)
    qml = gammahat / gammatheo
    return 2 * (2 * np.pi / nbpoints) * np.sum(qml)


def fspec_fgn(hest, nbpoints):
    """This is the spectral density of a fGN of Hurst exponent hest
    """
    hhest = - ((2 * hest) + 1)
    const = np.sin(np.pi * hest) * gamma(- hhest) / np.pi
    nhalfm = int((nbpoints - 1) / 2)
    dpl = 2 * np.pi * np.arange(1, nhalfm + 1) / nbpoints
    fspec = np.ones(nhalfm)
    for i in np.arange(0, nhalfm):
        dpfi = np.arange(0, 200)
        dpfi = 2 * np.pi * dpfi
        fgi = (np.abs(dpl[i] + dpfi)) ** hhest
        fhi = (np.abs(dpl[i] - dpfi)) ** hhest
        dpfi = fgi + fhi
        dpfi[0] = dpfi[0] / 2
        dpfi = (1 - np.cos(dpl[i])) * const * dpfi
        fspec[i] = np.sum(dpfi)
    fspec = fspec / np.exp(2 * np.sum(np.log(fspec)) / nbpoints)
    return fspec
