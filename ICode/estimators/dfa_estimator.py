import os
from ICode.estimators.dfa import hurstexp_dfa
from ICode.estimators import l2_penalization_on_grad
from ICode.estimators import grad_l2_penalization_on_grad
from ICode.estimators import mtvsolver, lipschitz_constant_gradf
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

__all__ = ['dfa_estimator', 'dfa_l2_estimator', 'dfa_tv_estimator', 'dfa_worker']

def dfa_worker(imgs, masker, regu='off', lbda=1, wtype=0, j1=3, j2=7):

    appro = masker.fit_transform(imgs)
    appro = appro.T
    mask = masker.mask_img_.get_data() > 0

    if regu=='off':
        return dfa_estimator(appro, mask, wtype, j1, j2)

    if regu=='tv':
        return dfa_tv_estimator(appro, mask, wtype, j1, j2, lbda)[1]

    if regu=='l2':
        return dfa_l2_estimator(appro, mask, wtype, j1, j2, lbda)[1]

    else:
        #raise ValueError('no regu = %s implemented' %(regu,))
        raise ValueError('no regu =  implemented')


def dfa_estimator(appro, mask=None, wtype=0, j1=3, j2=7):
    """
    This function estimate the Hurst exponent of a signal based
    on Detrend Fluctuation Analysis.
    -----------------------------------------------------------
    Input:
        appro: multidimensionnal signal as an np array (shape X Time)
        mask: if the data are taken from a masked image
        wtype: The kind of weighting used
                    0 -  no weigthing  (ie uniform weights)
                    1 -  1/nj weights  (suitable for fully Gaussian data)
                    2 -  use variance estimates varj
        j1:   the lower limit of the scales chosen,  1<= j1 <= scalemax-1
        j2:   the upper limit of the octaves chosen, 2<= j2 <= scalemax
                    0- no spatial regularisation
                    1- tv regularisation
                    2- l2 regularisation

    Output:
        image (shape) of Hurst coefficient
    """
    if mask is None:
            mask = np.ones(appro.shape[:-1])
    if appro.ndim > 2:
        shape = (reduce(lambda a,b : a*b , (1,) + simulation.shape[:-1]))
        appro = np.reshape(appro, (shape, appro.shape[-1]))
    N = appro.shape[0]
    l = appro.shape[-1]
   
    Hurst_exponent, dico = hurstexp_dfa(appro, j1=j1, j2=j2)

    return Hurst_exponent


def dfa_tv_estimator(appro, mask=None, wtype=0, j1=3, j2=7, lbda=1):
    """
    This function estimate the Hurst exponent of a signal based
    on Daubechies' Wavelet coeffecients.
    It also allow l2 and tv regularisation.
    -----------------------------------------------------------
    Input:
        appro: multidimensionnal signal as an np array (shape X Time)
        mask: if the data are taken from a masked image
        wtype: The kind of weighting used
                    0 -  no weigthing  (ie uniform weights)
                    1 -  1/nj weights  (suitable for fully Gaussian data)
                    2 -  use variance estimates varj
        j1:   the lower limit of the scales chosen,  1<= j1 <= scalemax-1
        j2:   the upper limit of the octaves chosen, 2<= j2 <= scalemax
                    0- no spatial regularisation
                    1- tv regularisation
                    2- l2 regularisation
        lbda: weight used in regularisation

    Output:
        image (shape) of Hurst coefficient
    """
    if mask is None:
            mask = np.ones(appro.shape[:-1])
    if appro.ndim > 2:
        shape = (reduce(lambda a,b : a*b , (1,) + simulation.shape[:-1]))
        appro = np.reshape(appro, (shape, appro.shape[-1]))
   
    Hurst_exponent, dico = hurstexp_dfa(appro, j1=j1, j2=j2)
    Elog = dico['log2F_dfa'].T
    Varlog = dico['Varf'].T
    nj = dico['nj']
    aest = dico['aest']
    lipschitz_constant = lipschitz_constant_gradf(j1,j2,Varlog, nj, wtype)
    l1_ratio = 0
    tv_algo = lambda lbda: mtvsolver(Hurst_exponent, aest,
                               Elog, Varlog,
                               nj, j1, j2,mask,
                               lipschitz_constant=lipschitz_constant,
                               l1_ratio = l1_ratio, l=lbda, verbose=0)

    Hmin = tv_algo(lbda)
    
    return Hurst_exponent, Hmin[0]


def dfa_l2_estimator(appro, mask=None, wtype=0, j1=3, j2=7, lbda=1):
    """
    This function estimate the Hurst exponent of a signal based
    on Detrend Fluctuation Analysis.
    -----------------------------------------------------------
    Input:
        appro: multidimensionnal signal as an np array (shape X Time)
        mask: if the data are taken from a masked image
        wtype: The kind of weighting used
                    0 -  no weigthing  (ie uniform weights)
                    1 -  1/nj weights  (suitable for fully Gaussian data)
                    2 -  use variance estimates varj
        j1:   the lower limit of the scales chosen,  1<= j1 <= scalemax-1
        j2:   the upper limit of the octaves chosen, 2<= j2 <= scalemax
                    0- no spatial regularisation
                    1- tv regularisation
                    2- l2 regularisation
        lbda: weight used in regularisation

    Output:
        image (shape) of Hurst coefficient
    """
    if mask is None:
            mask = np.ones(appro.shape[:-1])
    if appro.ndim > 2:
        shape = (reduce(lambda a,b : a*b , (1,) + simulation.shape[:-1]))
        appro = np.reshape(appro, (shape, appro.shape[-1]))
    N = appro.shape[0]
    l = appro.shape[-1]

    Hurst_exponent, dico = hurstexp_dfa(appro, j1=j1, j2=j2)
    Elog = dico['log2F_dfa'].T
    Varlog = dico['Varf'].T
    nj = dico['nj']
    aest = dico['aest']
    f = lambda x, lbda: l2_penalization_on_grad(x, aest,
                        Elog, Varlog, nj, j1, j2, mask, l=lbda)
    #We set epsilon to 0
    g = lambda x, lbda: grad_l2_penalization_on_grad(x, aest,
                        Elog, Varlog, nj, j1, j2, mask, l=lbda)

    fg = lambda x, lbda, **kwargs: (f(x, lbda), g(x, lbda))
    #For each lambda we use blgs algorithm to find the minimum
    # We start from the
    l2_algo = lambda lbda: fmin_l_bfgs_b(lambda x: fg(x, lbda), Hurst_exponent)
    
    Hmin = l2_algo(lbda)
    
    return Hurst_exponent, Hmin[0]
