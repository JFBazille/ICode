"""In this module one can find different way to compute the Hurst exponent
of a signal using detrend fluctuation analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
__all__ = ["hust_by_dfa"]


def hurstexp_dfa(data, CumSum=0, j1=2, j2=8, fignummer=0):
    """This function compute the DFA exponent of a signal,
    it gives the same result as the matlab function
    HDFAEstim of Biyu_code in schubert project
    """
    #first we should use numpy cumsum to sum up the data array
    if CumSum == 1:
        CSdata = np.cumsum(data, axis=-1)
    else:
        CSdata = data
    lendata = CSdata.shape[-1]
    b2 = j2
    if j2 > np.log2(lendata / 2):
        b2 = int(np.log2(lendata / 4))
    scales = np.arange(j1, b2 + 1)
    scales = 2 ** scales
    n = lendata / scales
    if data.ndim>1:
        F = np.zeros((len(scales), data.shape[0]))
    else:
        F = np.zeros(len(scales))
    for i in np.arange(0, len(scales)):
        breakpoints = np.arange(1, n[i]+1) * scales[i]
        data0 = signal.detrend(CSdata, bp = breakpoints)
        data0 = np.array_split(data0, breakpoints, axis=-1)
        #we need to remove the last element if too small
        if data0[-1].shape[-1]<data0[0].shape[-1]:
            del data0[-1]
        data0 = np.array(data0)
        F[i] = np.mean(np.std(data0, axis=-1), axis=0)
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

