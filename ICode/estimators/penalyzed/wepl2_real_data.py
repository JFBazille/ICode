import ICode.Optimize as o
import numpy as np
from ICode.optimize.objective_functions import _unmask

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
## Same function but allow to use a mask !
## The H and aest should have the same shape
## the whole figure (masked part + unmasked part)
## But yij, varyij and nj should have a shape compatible
## with the masked part (the part where mask is true !)
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #


def _fmask(H, aest, yij, varyj, nj, j1, j2, mask, wtype=1):
    """Same function but allow to use a mask !
    The H and aest should have the same shape
    the whole figure (masked part + unmasked part)
    But yij, varyij and nj should have a shape compatible
    with the masked part (the part where mask is true !)
    """

    j2 = np.min((j2, len(nj)))
    j1j2 = np.arange(j1 - 1, j2)
    njj = nj[j1j2]
    djh = 2 * np.outer(H, j1j2) + np.outer(aest, np.ones(len(j1j2)))
    S = (yij[:, j1j2] - djh) ** 2
    N = np.sum(njj)
    #  uniform weights
    if  wtype == 0:
        wvarjj = np.ones(len(j1j2))
        #wstr = 'Uniform'
    # Gaussian type weights
    elif wtype == 1:
        wvarjj = njj / N
        #wstr = 'Gaussian'
    #   % weights from data
    elif  wtype == 2:
        wvarjj = 1 / varyj
        #wstr = 'Estimated'
    #% all other cases
    else:
        print '** Weight option not recognised, using uniform weights\n'
        wvarjj = np.ones(1, len(j1j2))
        #wstr = 'Uniform'

    #then we multiply S by the pounderation
    S = S.dot(wvarjj)

    return np.sum(S)


def _gradfmask(H, aest, yij, varyj, nj, j1, j2, mask, wtype=1):
    """This function is the gradient of the preceding function f
    """

    l = H.shape[0]
    S = np.zeros(l)
    j2 = np.min((j2, len(nj)))
    j1j2 = np.arange(j1 - 1, j2)
    njj = nj[j1j2]
    N = sum(njj)
    #djh pour 2jH
    djh = 2 * np.outer(H, j1j2) + np.outer(aest, np.ones(len(j1j2)))

    #  uniform weights
    if     wtype == 0:
        wvarjj = np.ones(len(j1j2))
        #"wstr = 'Uniform'
    #  % Gaussian type weights
    elif wtype == 1:
        wvarjj = njj / N
        #wstr = 'Gaussian'
    # weights from data
    elif  wtype == 2:
        wvarjj = 1 / varyj
        #wstr = 'Estimated'
    # all other cases
    else:
        print '** Weight option not recognised, using uniform weights\n'
        wvarjj = np.ones(1, len(j1j2))
        #wstr = 'Uniform'

    S[np.reshape(mask, l)] = - 4 * (yij[:, j1j2] - djh).dot(j1j2 * wvarjj)
    return S


def _fmaskbis(Haest, yij, varyj, nj, j1, j2, mask, wtype=1, pfunc=None):
    """Same function but allow to use a mask !
    The H and aest should have the same shape
    the whole figure (masked part + unmasked part)
    But yij, varyij and nj should have a shape compatible
    with the masked part (the part where mask is true !)
    """
    #the f(H) part
    j2 = np.min((j2, len(nj)))
    j1j2 = np.arange(j1 - 1, j2)
    njj = nj[j1j2]
    n = len(Haest) / 2
    H = Haest[:n]
    aest = Haest[n:]
    djh = 2 * np.outer(H, j1j2) + np.outer(aest, np.ones(len(j1j2)))
    S = (yij[:, j1j2] - djh) ** 2
    N = sum(njj)

    #  uniform weights
    if     wtype == 0:
        wvarjj = np.ones(len(j1j2))
        #wstr = 'Uniform'
    #   Gaussian type weights
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
    if pfunc is None:
        return np.sum(S)
    return np.sum(S) + pfunc(H)


def _gradfmaskbis(Haest, yij, varyj, nj, j1, j2, mask, wtype=1, Gpfunc=None):
    """Gradient of the preceding function
    """
    shape = mask.shape
    j2 = np.min((j2, len(nj)))
    j1j2 = np.arange(j1 - 1, j2)
    njj = nj[j1j2]
    N = sum(njj)
    n = len(Haest) / 2
    H = Haest[:n]
    aest = Haest[n:]
    Gaest = np.zeros(n)
    GH = np.zeros(n)

    #djh for 2jH
    djh = 2 * np.outer(H, j1j2) + np.outer(aest, np.ones(len(j1j2)))

    #  uniform weights
    if wtype == 0:
        wvarjj = np.ones(len(j1j2))
        #wstr = 'Uniform'
    #  Gaussian type weights
    elif wtype == 1:
        wvarjj = njj / N
        #wstr = 'Gaussian'
    #    weights from data
    elif  wtype == 2:
        wvarjj = 1 / varyj
        #wstr = 'Estimated'
    # all other cases
    else:
        print '** Weight option not recognised, using uniform weights\n'
        wvarjj = np.ones(1, len(j1j2))
        #wstr = 'Uniform'

    G = - 2 * (yij[:, j1j2] - djh)
    Gaest = G.dot(wvarjj)

    S = 2 * G.dot(j1j2 * wvarjj)
    GH = S
    if not (Gpfunc is None):
        GH += Gpfunc(H)
    #return np.reshape((S + 2*l*H),(1,len(H)))
    return np.concatenate((GH, Gaest))


def jhem(H, aest, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    return fmask(H, aest, yij, varyj, nj, j1, j2, mask, wtype) + l * np.sum(
           o.gradient(np.reshape(H,mask.shape)) ** 2)


def gradjhem(H, aest, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    return gradfmask(H, aest, yij, varyj, nj, j1, j2, mask,
            wtype) - 2 * l * np.reshape(
           o.div(o.gradient(np.reshape(H,mask.shape))), H.shape)


def jhembis(Haest, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    """
    Haest is the concatenation of H and aest,
    shape is the shape of the original image
    """
    return fmaskbis(Haest, yij, varyj, nj, j1, j2, mask, wtype,
              pfunc=lambda x: l * np.sum(
              o.gradient(np.reshape(x,mask.shape)) ** 2))


def gradjhembis(Haest, yij, varyj, nj, j1, j2, mask, l=0, wtype=1):
    return gradfmaskbis(Haest, yij, varyj, nj, j1, j2, mask, wtype,
                  Gpfunc=lambda x: - 2 * l * np.reshape(
                  o.div(o.gradient(np.reshape(x,mask.shape))), x.shape))