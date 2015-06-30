"""
These functions compute the Hurst exponent of a signal using the
Welch periodogram
We use here the scipy estimator of welch periodogram
"""
import numpy as np
from scipy.signal import welch
#import nitime.algorithms.spectral as nas


def welchp(data):
    """
    These functions compute the Hurst exponent of a signal using the
    Welch periodogram
    data : your signal
    """
    frq, pwr = welch(data)
    #f,p = nas.get_spectra(data)
    mask = np.all([(frq > 0), (frq < 0.1)], axis=0)
    if any(mask):
        tmp = np.polyfit(np.log2(frq[mask]), np.log2(np.abs(pwr[mask])), deg=1)
        H = (- 1 - tmp[0]) / 2
        return H
    else:
        return 0


def welchps(data, idx):
    """
    These functions compute the Hurst exponent of a signal using the
    Welch periodogram
    data : your signal Space x Time
    idx : the indice of space you want to use (usefull for script see
    test example and fast)
    """
    frq, pwr = welch(data[idx, :])
    mask = np.all([(frq > 0.001)], axis=0)
    if any(mask):
        tmp = np.polyfit(np.log2(frq[mask]), np.log2(np.abs(pwr[mask])), deg=1)
        H = (- 1 - tmp[0]) / 2
        return H
    else:
        return 0
