import numpy as np
__all__ = ['f', 'graf', 'fbis', 'gradfbis']
def f(H, aest, yij, varyj, nj, j1, j2, wtype=1):
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


def gradf(H, aest, yij, varyj, nj, j1, j2, wtype=1):
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


def fbis(Haest, yij, varyj, nj, j1, j2, wtype=1, pfunc=None):
    """
    f(Hi) is the norme L2 of the difference
    between the estimated coefficient H
    but Haest is the concatenation of H and aest. it should allow computation
    where aest can variate (see example for more details)
    and the data
    usually yij = Elogmuqj for all i but we write yij it is simpler and shorter
    we allow different possible weight but usually we will use the nj as weigth
    """
    j2 = np.min((j2, len(nj)))
    j1j2 = np.arange(j1 - 1, j2)
    njj = nj[j1j2]
    n = len(Haest)
    H = Haest[:n / 2]
    aest = Haest[n / 2:]
    djh = 2 * np.outer(H, j1j2 + 1) + np.outer(aest, np.ones(len(j1j2)))
    S = (yij[:, j1j2] - djh) ** 2
    N = sum(njj)
    #  uniform weights
    if wtype == 0:
        wvarjj = np.ones(len(j1j2))
        #wstr = 'Uniform'
    #  Gaussian type weights
    elif wtype == 1:
        wvarjj = njj / N
        #wstr = 'Gaussian'
    #  weights from data
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
    if pfunc is None:
        return sum(S)

    return sum(S) + pfunc(H)


def gradfbis(Haest, yij, varyj, nj, j1, j2, wtype=1, Gpfunc=None):
    """this function is allow to compute the gradient of f bis
       and add to the H part (the first part of the gradient)
       the part of the gradient linked to the penalization
       See below for example
    """
    j2 = np.min((j2, len(nj)))
    j1j2 = np.arange(j1 - 1, j2)
    njj = nj[j1j2]
    N = sum(njj)
    n = len(Haest)
    H = Haest[:n / 2]
    aest = Haest[n / 2:]
    #djh pour 2jH
    djh = 2 * np.outer(H, j1j2 + 1) + np.outer(aest, np.ones(len(j1j2)))
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

    #Gbuff is a buffer
    Gbuff = - 2 * (yij[:, j1j2] - djh)
    #the aest par of the gradient
    Gaest = Gbuff.dot(wvarjj)
    #we reuse G for
    S = 2 * Gbuff.dot((j1j2 + 1) * wvarjj)
    GradH = S
    if not (Gpfunc is None):
        GradH += Gpfunc(H)
    #return np.reshape((S + 2*l*H),(1,len(H)))
    return np.concatenate((GradH, Gaest))


