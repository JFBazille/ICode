import os
from ICode.estimators.wavelet import hdw_p, lipschitz_constant_gradf
from ICode.estimators.wavelet import wl_l2_penalization_on_grad
from ICode.estimators.wavelet import grad_wl_l2_penalization_on_grad
from ICode.estimators.wavelet import mtvsolver
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

__all__ = ['wavelet_estimator', 'wavelet_l2_estimator', 'wavelet_tv_estimator', 'wavelet_worker']

def wavelet_worker(imgs, masker, regu='off', lbda=1, nb_vanishmoment=2, norm=1,
                      q=np.array(2), nbvoies=None,
                      distn=1, wtype=1, j1=2, j2=8):

    appro = masker.fit_transform(imgs)
    appro = appro.T
    mask = masker.mask_img_.get_data() > 0

    if regu=='off':
        return wavelet_estimator(appro, mask, nb_vanishmoment,
                                 norm, q, nbvoies,
                                 distn, wtype, j1, j2)

    if regu=='tv':
        return wavelet_tv_estimator(appro, mask, nb_vanishmoment,
                                 norm, q, nbvoies,
                                 distn, wtype, j1, j2, lbda)[1]

    if regu=='l2':
        return wavelet_l2_estimator(appro, mask, nb_vanishmoment,
                                 norm, q, nbvoies,
                                 distn, wtype, j1, j2, lbda)[1]

    else:
        #raise ValueError('no regu = %s implemented' %(regu,))
        raise ValueError('no regu =  implemented')


def wavelet_estimator(appro, mask=None, nb_vanishmoment=2, norm=1, q=np.array(2), nbvoies=None,
                      distn=1, wtype=1, j1=2, j2=8):
    """
    This function estimate the Hurst exponent of a signal based
    on Daubechies' Wavelet coeffecients.
    It also allow l2 and tv regularisation.
    -----------------------------------------------------------
    Input:
        appro: multidimensionnal signal as an np array (shape X Time)
        mask: if the data are taken from a masked image
        q:  the order of the statistic used
        nb_vanishmoment:  the number of vanishing moments of the wavelet
            ((here only used in the plot title)
        nbvoies:    number of octaves
        distn:    Hypothesis on the data distribution or `singal type':
                    0: Gauss,  1: Non-G finite va
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
    if nbvoies is None:
        nbvoies = min(int(np.log2(l / (2 * nb_vanishmoment + 1))), int(np.log2(l)))

    dico = hdw_p(appro, nb_vanishmoment, norm, q, nbvoies, distn, wtype, j1, j2, 0)
    Hurst_exponent = dico['Zeta'] / 2.

    return Hurst_exponent


def wavelet_tv_estimator(appro, mask=None, nb_vanishmoment=2, norm=1, q=np.array(2), nbvoies=None,
                      distn=1, wtype=1, j1=2, j2=8, lbda=1):
    """
    This function estimate the Hurst exponent of a signal based
    on Daubechies' Wavelet coeffecients.
    It also allow l2 and tv regularisation.
    -----------------------------------------------------------
    Input:
        appro: multidimensionnal signal as an np array (shape X Time)
        mask: if the data are taken from a masked image
        q:  the order of the statistic used
        nb_vanishmoment:  the number of vanishing moments of the wavelet
            ((here only used in the plot title)
        nbvoies:    number of octaves
        distn:    Hypothesis on the data distribution or `singal type':
                    0: Gauss,  1: Non-G finite va
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
    if nbvoies is None:
        nbvoies = min(int(np.log2(l / (2 * nb_vanishmoment + 1))), int(np.log2(l)))

    dico = hdw_p(appro, nb_vanishmoment, norm, q, nbvoies, distn, wtype, j1, j2, 0)
    Elog = dico['Elogmuqj'][:, 0]
    Varlog = dico['Varlogmuqj'][:, 0]
    nj = dico['nj']
    estimate = dico['Zeta'] / 2.
    aest = dico['aest']
    lipschitz_constant = lipschitz_constant_gradf(j1,j2,Varlog, nj, wtype)
    l1_ratio = 0
    tv_algo = lambda lbda: mtvsolver(estimate, aest,
                               Elog, Varlog,
                               nj, j1, j2,mask,
                               lipschitz_constant=lipschitz_constant,
                               l1_ratio = l1_ratio, l=lbda, verbose=0)

    Hmin = tv_algo(lbda)
    
    return estimate, Hmin[0]


def wavelet_l2_estimator(appro, mask=None, nb_vanishmoment=2, norm=1, q=np.array(2), nbvoies=None,
                      distn=1, wtype=1, j1=2, j2=8, lbda=1):
    """
    This function estimate the Hurst exponent of a signal based
    on Daubechies' Wavelet coeffecients.
    It also allow l2 and tv regularisation.
    -----------------------------------------------------------
    Input:
        appro: multidimensionnal signal as an np array (shape X Time)
        mask: if the data are taken from a masked image
        q:  the order of the statistic used
        nb_vanishmoment:  the number of vanishing moments of the wavelet
            ((here only used in the plot title)
        nbvoies:    number of octaves
        distn:    Hypothesis on the data distribution or `singal type':
                    0: Gauss,  1: Non-G finite va
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
    if nbvoies is None:
        nbvoies = min(int(np.log2(l / (2 * nb_vanishmoment + 1))), int(np.log2(l)))

    dico = hdw_p(appro, nb_vanishmoment, norm, q, nbvoies, distn, wtype, j1, j2, 0)
    Elog = dico['Elogmuqj'][:, 0]
    Varlog = dico['Varlogmuqj'][:, 0]
    nj = dico['nj']
    estimate = dico['Zeta'] / 2.
    aest = dico['aest']
    f = lambda x, lbda: wl_l2_penalization_on_grad(x, aest,
                        Elog, Varlog, nj, j1, j2, mask, l=lbda)
    #We set epsilon to 0
    g = lambda x, lbda: grad_wl_loss_l2_penalization_on_grad(x, aest,
                        Elog, Varlog, nj, j1, j2, mask, l=lbda)

    fg = lambda x, lbda, **kwargs: (f(x, lbda), g(x, lbda))
    #For each lambda we use blgs algorithm to find the minimum
    # We start from the
    l2_algo = lambda lbda: fmin_l_bfgs_b(lambda x: fg(x, lbda), estimate)
    
    Hmin = l2_algo(lbda)
    
    return estimate, Hmin[0]
