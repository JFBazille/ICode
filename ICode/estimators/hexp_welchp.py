"""
These functions compute the Hurst exponent of a signal using the
Welch periodogram
We use here the nitime estimator of welch periodogram
"""
from nitime.analysis.spectral import SpectralAnalyzer
from nitime.timeseries import TimeSeries
import numpy as np


def hurstexp_welchper(data, samp=0.2, f_max=0.1):
    """
    These functions compute the Hurst exponent of a signal using the
    Welch periodogram
    data : your signal
    samp : sampling rate in Hz 0.2 for an fMRI series
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


def hurstexp_welchper_2(d, samp=0.2, f_max=0.1):
    """
    These functions compute the Hurst exponent of a signal using the
    Welch periodogram
    data : your signal
    samp : sampling rate in Hz 0.2 for an fMRI series
    f_max: the higher frequency you want to take into account
    I have change a little bit the computation of H (see below retour)
    """
    #data could be two dimensional(but no more...) in that cas time should
    #be on second position
    time_series = TimeSeries(data=d, sampling_rate=samp)
    spectral_analysis = SpectralAnalyzer(time_series)
    frq, pwr = spectral_analysis.psd
    #We need to take only the small frequency, but the exact choice
    #is a bit arbitrary we need to have alpha between 0 and 1
    masker = np.all([(frq > 0), (frq < f_max)], axis=0)
    tmp = np.polyfit(np.log2(frq[masker]), np.log2(pwr.T[masker]), deg=1)
    beta = - tmp[0]
    masker = beta < 1
    #print np.sum(masker)
    retour = np.array((beta - 1) / 2)
    retour[masker] = (beta[masker] + 1) / 2
    return retour
