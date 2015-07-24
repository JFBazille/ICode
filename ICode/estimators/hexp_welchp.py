"""
These functions compute the Hurst exponent of a signal using the
Welch periodogram
We use here the nitime estimator of welch periodogram
"""
from nitime.analysis.spectral import SpectralAnalyzer
from nitime.timeseries import TimeSeries
import numpy as np


def hurstexp_welchper(data, samp=1, f_max=1):
    """
    These functions compute the Hurst exponent of a signal using the
    Welch periodogram
    data : your signal
    samp : sampling rate in Hz 1 for an fMRI series
    f_max: the higher frequency you want to take into account
    """
    #data could be two dimensional(but no more...) in that cas time should
    #be on second position
    time_series = TimeSeries(data=data, sampling_rate=samp)
    spectral_analysis = SpectralAnalyzer(time_series)
    frq, pwr = spectral_analysis.psd
    #We need to take only the small frequency, but the exact choice is a
    #bit arbitrary we need to have alpha between 0 and 1
    masker = np.all([(frq > 0), (frq < f_max)], axis=0)
    tmp = np.polyfit(np.log2(frq[masker]), np.log2(pwr.T[masker]), deg=1)
    return (1 - tmp[0]) / 2
