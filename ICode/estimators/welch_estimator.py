import os
from ICode.estimators.welch import hurstexp_welchper
from ICode.estimators.welch import welch_squared_loss_l2pen
from ICode.estimators.welch import tv_welch_solver
from ICode.estimators.welch import lipschitz_constant_grad_welch_square_loss
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

__all__ = ['welch_estimator', 'welch_l2_estimator', 'welch_tv_estimator', 'welch_worker']

def welch_worker(imgs, masker, regu='off', lbda=1):

    appro = masker.fit_transform(imgs)
    appro = appro.T
    mask = masker.mask_img_.get_data() > 0

    if regu=='off':
        return welch_estimator(appro, mask, consider_fBm=True)

    if regu=='tv':
        return welch_tv_estimator(appro, mask,lbda, consider_fBm=True)[1]

    if regu=='l2':
        return welch_l2_estimator(appro, mask, wtype, lbda, consider_fBm=True)[1]

    else:
        #raise ValueError('no regu = %s implemented' %(regu,))
        raise ValueError('no regu =  implemented')


def welch_estimator(appro, mask=None, consider_fBm=False):
    """
    This function estimate the Hurst exponent of a signal based
    on Detrend Fluctuation Analysis.
    -----------------------------------------------------------
    Input:
        appro: multidimensionnal signal as an np array (shape X Time)
        mask: if the data are taken from a masked image

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
   
    Hurst_exponent, dico = hurstexp_welchper(appro, consider_fBm=consider_fBm)

    return Hurst_exponent


def welch_tv_estimator(appro, mask=None, lbda=1, consider_fBm=False):
    """
    This function estimate the Hurst exponent of a signal based
    on Daubechies' Wavelet coeffecients.
    It also allow l2 and tv regularisation.
    -----------------------------------------------------------
    Input:
        appro: multidimensionnal signal as an np array (shape X Time)
        mask: if the data are taken from a masked image
        lbda: weight used in regularisation

    Output:
        image (shape) of Hurst coefficient
    """
    if mask is None:
            mask = np.ones(appro.shape[:-1])==1
    if appro.ndim > 2:
        shape = (reduce(lambda a,b : a*b , (1,) + simulation.shape[:-1]))
        appro = np.reshape(appro, (shape, appro.shape[-1]))
   
    Hurst_exponent, dico = hurstexp_welchper(appro, consider_fBm=consider_fBm)
    log2frq = dico['log2frq']
    log2pwr = dico['log2pwr']
    aest = dico['aest']
    lipschitz_constant = lipschitz_constant_grad_welch_square_loss(log2frq, consider_fBm=consider_fBm)
    l1_ratio = 0
    tv_algo = lambda lbda: tv_welch_solver(Hurst_exponent, aest,
                               log2frq, log2pwr, mask,
                               lipschitz_constant=lipschitz_constant,
                               l1_ratio = l1_ratio, l=lbda, verbose=0, consider_fBm=consider_fBm)

    Hmin = tv_algo(lbda)
    
    return Hurst_exponent, Hmin[0]


def welch_l2_estimator(appro, mask=None, lbda=1, consider_fBm=False):
    """
    This function estimate the Hurst exponent of a signal based
    on Detrend Fluctuation Analysis.
    -----------------------------------------------------------
    Input:
        appro: multidimensionnal signal as an np array (shape X Time)
        mask: if the data are taken from a masked image
        lbda: weight used in regularisation

    Output:
        image (shape) of Hurst coefficient
    """
    if mask is None:
            mask = np.ones(appro.shape[:-1])==1
    if appro.ndim > 2:
        shape = (reduce(lambda a,b : a*b , (1,) + simulation.shape[:-1]))
        appro = np.reshape(appro, (shape, appro.shape[-1]))
    N = appro.shape[0]
    l = appro.shape[-1]

    Hurst_exponent, dico = hurstexp_welchper(appro, consider_fBm=consider_fBm)
    log2frq = dico['log2frq']
    log2pwr = dico['log2pwr']
    aest = dico['aest']

    fg = lambda x, lbda, **kwargs: welch_squared_loss_l2pen(x, aest, log2frq, log2pwr, mask, lbda, consider_fBm=consider_fBm)

    l2_algo = lambda lbda: fmin_l_bfgs_b(lambda x: fg(x, lbda), Hurst_exponent)
    
    Hmin = l2_algo(lbda)
    
    return Hurst_exponent, Hmin[0]