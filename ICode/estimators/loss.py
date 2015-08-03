import numpy as np
import ICode.optimize as o
from ICode.optimize.objective_functions import _unmask
from ICode.optimize import tv
from ICode.optimize.fista import mfista
from ICode.optimize.proximal_operator import _prox_tvl1_with_intercept, _prox_tvl1

__all__ = ['l2_loss', 'grad_l2_loss', 'mtvsolver', 'l2_penalization_on_grad',
           'grad_l2_penalization_on_grad', 'lipschitz_constant_gradf']

def l2_loss(H, aest, yij, varyj, nj, j1, j2, wtype=1):
    """
    f(Hi) is the L2 norme of the difference
    between the estimated coefficient H
    and the data
    usually yij = Elogmuqj for all i but we write yij it is simpler and shorter
    we allow different possible weight but usually we will use the nj as weigth
    """
    j2 = np.min((j2, len(nj)))
    j1j2 = np.arange(j1 - 1, j2)
    njj = nj[j1j2]
    djh = 2 * np.outer(H, (j1j2 + 1)) + np.outer(aest, np.ones(len(j1j2)))
    S = (yij[:, j1j2] - djh) ** 2
    N = sum(njj)
    #  uniform weights
    if wtype == 0:
        wvarjj = np.ones(len(j1j2))
        #wstr = 'Uniform'
    # Gaussian type weights
    elif wtype == 1:
        wvarjj = njj / N
        #wstr = 'Gaussian'
    #   weights from data
    elif  wtype == 2:
        wvarjj = 1 / varyj
        #wstr = 'Estimated'
    # all other cases
    else:
        print '** Weight option not recognised, using uniform weights\n'
        wvarjj = np.ones(1, len(j1j2))
        #wstr = 'Uniform'

    #then we multiply S by the pounderation
    S = S.dot(wvarjj)

    return np.sum(S)


def grad_l2_loss(H, aest, yij, varyj, nj, j1, j2, wtype=1):
    """This function is the gradient of the preceding function f
    """
    j2 = np.min((j2, len(nj)))
    j1j2 = np.arange(j1 - 1, j2)
    njj = nj[j1j2]

    #djh pour 2jH
    djh = 2 * np.outer(H, j1j2 + 1) + np.outer(aest, np.ones(len(j1j2)))
    #  uniform weights
    if  wtype == 0:
        wvarjj = np.ones(len(j1j2))
        #wstr = 'Uniform'
    # Gaussian type weights
    elif wtype == 1:
        wvarjj = njj / sum(njj)
        #wstr = 'Gaussian'
    #   weights from data
    elif  wtype == 2:
        wvarjj = 1 / varyj
        #wstr = 'Estimated'
    #all other cases
    else:
        print '** Weight option not recognised, using uniform weights\n'
        wvarjj = np.ones(1, len(j1j2))
        #wstr = 'Uniform'

    S = - 4 * (yij[:, j1j2] - djh).dot((j1j2 + 1) * wvarjj)
    return S


def l2_penalization_on_grad(H, aest, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    grad = o.grad_for_masked_data(H,mask)
    return l2_loss(H, aest, yij, varyj, nj, j1, j2, wtype) + l * np.sum(grad**2)


def grad_l2_penalization_on_grad(H, aest, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    grad = o.grad_for_masked_data(H,mask)
    div = o.div(grad)[mask]
    return grad_l2_loss(H, aest, yij, varyj, nj, j1, j2,
                 wtype) - 2 * l * div


def lipschitz_constant_gradf(j1,j2,varyj, nj, wtype):
    j2 = np.min((j2, len(nj)))
    j1j2 = np.arange(j1 - 1, j2)
    njj = nj[j1j2]
    N = sum(njj)

    #  uniform weights
    if     wtype == 0:
        wvarjj = np.ones(len(j1j2))
        #wstr = 'Uniform'
    #  Gaussian type weights
    elif wtype == 1:
        wvarjj = njj / N
        #wstr = 'Gaussian'
    #   % weights from data
    elif  wtype == 2:
        wvarjj = 1 / varyj
        #wstr = 'Estimated'
    # all other cases
    else:
        print '** Weight option not recognised, using uniform weights\n'
        wvarjj = np.ones(1, len(j1j2))
        #wstr = 'Uniform'

    return np.sum(8 * ((j1j2 + 1) ** 2) * wvarjj)

def mtvsolver(Hurst_init, aest, yij, varyj, nj, j1, j2,
              mask, max_iter=100, init=None,
              prox_max_iter=5000, tol=1e-4, call_back=None, verbose=1,
              l=1, l1_ratio=0,lipschitz_constant=0, wtype=1):
    """
    This function yields the total energy of the Tv penalyzation
    
    """

    # shape of image box
    
    alpha = 1

    flat_mask = mask.ravel()
    volume_shape = mask.shape
    H_size = len(Hurst_init)

    if lipschitz_constant == 0:
        lipschitz_constant = lipschitz_constant_gradf(j1,j2,varyj, nj, wtype)

    #init["z"] = Hurst_init.copy()
    #init["t"] = 1
    #ini["stepsize"] = 1 / lipschitz_constant

    def total_energy(x):
        return l2_loss(x, aest, yij, varyj, nj, j1, j2, wtype) + l * tv(_unmask(x,mask))

    def unmaskvec(w):
        return _unmask(w, mask)

    def maskvec(w):
        return w[flat_mask]

    def f1_grad(x):
        return grad_l2_loss(x, aest, yij, varyj, nj, j1, j2, wtype)

    def f2_prox(w, stepsize, dgap_tol, init):
        out, info = _prox_tvl1(unmaskvec(w),
                weight= (l + 1.e-6) * stepsize, l1_ratio=l1_ratio,
                dgap_tol=dgap_tol, init=unmaskvec(init),
                max_iter=prox_max_iter, fista = False,
                verbose=verbose)

        return maskvec(out.ravel()), info

    w, obj, init = mfista(
        f1_grad, f2_prox, total_energy, lipschitz_constant, H_size,
        dgap_factor=(.1 + l1_ratio) ** 2, tol=tol, init=init, verbose=verbose,
        max_iter=max_iter, callback=None)
    
    return w, obj, init