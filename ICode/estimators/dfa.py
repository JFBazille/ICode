"""In this module one can find different way to compute the Hurst exponent
of a signal using detrend fluctuation analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def dfa(data, CumSum=1, j1=2, j2=8, fignummer=0):
    """This function compute the DFA exponent of a signal,
    it gives the same result as the matlab function
    HDFAEstim of Biyu_code in schubert project
    """
    #first we should use numpy cumsum to sum up the data array
    if CumSum == 1:
        CSdata = np.cumsum(data)
    else:
        CSdata = data
    lendata = len(CSdata)
    b2 = j2
    if j2 > np.log2(lendata / 2):
        b2 = int(np.log2(lendata / 4))
    scales = np.arange(j1, b2 + 1)
    scales = 2 ** scales
    n = lendata / scales
    F = np.zeros(len(scales))
    for i in np.arange(0, len(scales)):
        for j in np.arange(0, n[i]):
            data0 = signal.detrend(CSdata[j * scales[i]:(j + 1) * scales[i]])
            F[i] = F[i] + np.var(data0) ** (1 / 2.)
        F[i] = F[i] / n[i]
    tmp = np.polyfit(np.log2(scales), np.log2(F), deg=1)
    alpha = tmp[0]
    if fignummer > 0:
        qq = np.polyval(tmp, np.log2(scales))
        plt.figure(fignummer)
        plt.plot(np.log2(scales), qq, color="blue")
        plt.plot(np.log2(scales), np.log2(F), color="red")
        plt.xlabel("log(L)")
        plt.ylabel("log(F(L)**2)")
        plt.title("DFA\nestimated alpha :" + str(alpha))
        plt.legend()
        plt.show()
    return alpha


def dfa_t(data, pointeur, idx, CumSum=1, j1=2, j2=8):
    """This function compute the DFA exponent of a signal,
    it gives the same result as the matlab function
    HDFAEstim of Biyu_code in schubert project
    """
    #first we should use numpy cumsum to sum up the data array
    if CumSum == 1:
        CSdata = np.cumsum(data)
    else:
        CSdata = data
    lendata = len(CSdata)
    b2 = j2
    if j2 > np.log2(lendata / 2):
        b2 = int(np.log2(lendata / 4))
    scales = np.arange(j1, b2 + 1)
    scales = 2 ** scales
    n = lendata / scales
    F = np.zeros(len(scales))
    for i in np.arange(0, len(scales)):
        for j in np.arange(0, n[i]):
            data0 = signal.detrend(CSdata[j * scales[i]:(j + 1) * scales[i]])
            F[i] = F[i] + np.var(data0) ** (1 / 2.)
        F[i] = F[i] / n[i]
    tmp = np.polyfit(np.log2(scales), np.log2(F), deg=1)
    #here we give the reference of the data we want to modify
    pointeur[idx] = tmp[0]


def dfa_norm_t(data, pointeur, idx, CumSum=1, j1=2, j2=8):
    """This function compute the DFA exponent of a signal,
    it gives the same result as the matlab function
    HDFAEstim of Biyu_code in schubert project
    it store the result in pointeur[idx]
    """
    datanorm = (data - np.mean(data)) / np.std(data)
    #first we should use numpy cumsum to sum up the data array
    if CumSum == 1:
        CSdata = np.cumsum(datanorm)
    else:
        CSdata = datanorm
    lendata = len(CSdata)
    b2 = j2
    if j2 > np.log2(lendata / 2):
        b2 = int(np.log2(lendata / 4))
    scales = np.arange(j1, b2 + 1)
    scales = 2 ** scales
    n = lendata / scales
    F = np.zeros(len(scales))
    for i in np.arange(0, len(scales)):
        for j in np.arange(0, n[i]):
            data0 = signal.detrend(CSdata[j * scales[i]:(j + 1) * scales[i]])
            F[i] = F[i] + np.var(data0) ** (1 / 2.)
        F[i] = F[i] / n[i]
    tmp = np.polyfit(np.log2(scales), np.log2(F), deg=1)
    #here we give the reference of the data we want to modify
    pointeur[idx] = tmp[0]


def dfa_s(data, idx, CumSum=1, j1=2, j2=8):
    """This function compute the DFA exponent of a signal,
    it gives the same result as the matlab function
    HDFAEstim of Biyu_code in schubert project
    it consider only the data data[idx,:]
    """
    dataidx = data[idx, :]
    #first we should use numpy cumsum to sum up the data array
    if CumSum == 1:
        CSdata = np.cumsum(dataidx)
    else:
        CSdata = dataidx
    lendata = len(CSdata)
    b2 = j2
    if j2 > np.log2(lendata / 2):
        b2 = int(np.log2(lendata / 4))
    scales = np.arange(j1, b2 + 1)
    scales = 2 ** scales
    n = lendata / scales
    F = np.zeros(len(scales))
    for i in np.arange(0, len(scales)):
        for j in np.arange(0, n[i]):
            data0 = signal.detrend(CSdata[j * scales[i]:(j + 1) * scales[i]])
            F[i] = F[i] + np.var(data0) ** (1 / 2.)
        F[i] = F[i] / n[i]
    tmp = np.polyfit(np.log2(scales), np.log2(F), deg=1)
    return tmp[0]


def dfa_norm_s(data, idx, CumSum=1, j1=2, j2=8):
    """This function compute the DFA exponent of a signal,
    it gives the same result as the matlab function
    HDFAEstim of Biyu_code in schubert project
    """
    datanorm = (data - np.mean(data)) / np.std(data)
    datanormidx = datanorm[idx, :]
    #first we should use numpy cumsum to sum up the data array
    if CumSum == 1:
        CSdata = np.cumsum(datanormidx)
    else:
        CSdata = datanormidx
    lendata = len(CSdata)
    #then we take a M the default M is 3
    b2 = j2
    if j2 > np.log2(lendata / 2):
        b2 = int(np.log2(lendata / 4))
    scales = np.arange(j1, b2 + 1)
    scales = 2 ** scales
    n = lendata / scales
    F = np.zeros(len(scales))
    for i in np.arange(0, len(scales)):
        for j in np.arange(0, n[i]):
            data0 = signal.detrend(CSdata[j * scales[i]:(j + 1) * scales[i]])
            F[i] = F[i] + np.var(data0) ** (1 / 2.)
        F[i] = F[i] / n[i]
    tmp = np.polyfit(np.log2(scales), np.log2(F), deg=1)
    return tmp[0]
