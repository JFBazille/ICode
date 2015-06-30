#Wavelet estimator penalisation L2 first and simpler penalization
import numpy as np
from scipy.ndimage.filters import laplace
import ICode.optimize as o
"""
this functions are objective functions of type
J = sum(f(Hi)) + lambda x penalyzation(H)

f(Hi) is the norme L2 of the difference
between the estimated coefficient H

The penalization function are variate you can see below a lot
of different example


the data :
    usually yij = Elogmuqj for all i but we write yij it is simpler and shorter
    we allow different possible weight but usually we will use the nj as weigth
"""


def f(H, aest, yij, varyj, nj, j1, j2, wtype=1):
    """
    f(Hi) is the norme L2 of the difference
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
        return S

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

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
## Same function but allow to use a mask !
## The H and aest should have the same shape
## the whole figure (masked part + unmasked part)
## But yij, varyij and nj should have a shape compatible
## with the masked part (the part where mask is true !)
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #


def fmask(H, aest, yij, varyj, nj, j1, j2, mask, wtype=1):
    """Same function but allow to use a mask !
    The H and aest should have the same shape
    the whole figure (masked part + unmasked part)
    But yij, varyij and nj should have a shape compatible
    with the masked part (the part where mask is true !)
    """
    shape = mask.shape
    H = np.reshape(H, shape)[mask]
    aest = np.reshape(aest, shape)[mask]
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


def gradfmask(H, aest, yij, varyj, nj, j1, j2, mask, wtype=1):
    """This function is the gradient of the preceding function f
    """
    shape = mask.shape
    l = H.shape[0]
    S = np.zeros(l)
    H = np.reshape(H, shape)[mask]
    aest = np.reshape(aest, shape)[mask]
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


def fmaskbis(Haest, yij, varyj, nj, j1, j2, mask, wtype=1, pfunc=None):
    """Same function but allow to use a mask !
    The H and aest should have the same shape
    the whole figure (masked part + unmasked part)
    But yij, varyij and nj should have a shape compatible
    with the masked part (the part where mask is true !)
    """
    shape = mask.shape
    #the f(H) part
    j2 = np.min((j2, len(nj)))
    j1j2 = np.arange(j1 - 1, j2)
    njj = nj[j1j2]
    n = len(Haest) / 2
    H = Haest[:n]
    aest = Haest[n:]
    Hm = np.reshape(H, shape)[mask]
    aest = np.reshape(aest, shape)[mask]
    djh = 2 * np.outer(Hm, j1j2) + np.outer(aest, np.ones(len(j1j2)))
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


def gradfmaskbis(Haest, yij, varyj, nj, j1, j2, mask, wtype=1, Gpfunc=None):
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
    mmask = np.reshape(mask, n)
    Hm = np.reshape(H, shape)[mask]
    aest = np.reshape(aest, shape)[mask]
    #djh for 2jH
    djh = 2 * np.outer(Hm, j1j2) + np.outer(aest, np.ones(len(j1j2)))

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
    Gaest[mmask] = G.dot(wvarjj)

    S = 2 * G.dot(j1j2 * wvarjj)
    GH[mmask] = S
    if not (Gpfunc is None):
        GH += Gpfunc(H)
    #return np.reshape((S + 2*l*H),(1,len(H)))
    return np.concatenate((GH, Gaest))


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#this function is the objective function
#J = sum(f(Hi)) + normel2(H)^2


def j2(H, aest, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return f(H, aest, yij, varyj, nj, j1, j2, wtype=1) + l * np.sum(H ** 2)


def gradj2(H, aest, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return gradf(H, aest, yij, varyj, nj, j1, j2, wtype=1) + 2 * l * H


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#this function is the same objective function
#but we use also aest as a variable ! and see if it yields a different result
#Haest is a 2n array, the n first term are H and the n last aest,
#it is as simple as that !
def j2bis(Haest, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return fbis(Haest, yij, varyj, nj, j1, j2,
              pfunc=lambda x: l * np.sum(x ** 2))


def gradj2bis(Haest, yij, varyj, nj, j1, j2, l=0, wtype=1):
    return gradfbis(Haest, yij, varyj, nj, j1, j2, wtype,
                  Gpfunc=lambda x: 2 * l * x)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#this function is the objective function
#J = sum(f(Hi)) + normel2(Gradient(H))^2
#where f(Hi) is the norme L2 of the difference between
#the estimated coefficient H and the data
#usually yij = Elogmuqj for all i but we write yij it is simpler and shorter
#we allow different possible weight but usually we will use the nj as weigth
#I call it JH2 because the norm remind me of the H2 norm
def jh2(H, aest, shape, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return f(H, aest, yij, varyj, nj, j1, j2, wtype) + l * np.sum(
           np.array(np.gradient(np.reshape(H, shape))) ** 2)


def gradjh2(H, aest, shape, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return gradf(H, aest, yij, varyj, nj, j1, j2, wtype) - 2 * l * np.reshape(
           laplace(np.reshape(H, shape), mode='reflect'), H.shape)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#this function is the same objective function
#but we use also aest as a variable ! and see if it yields a different result
#Haest is a 2n array, the n first term are H and the n last aest,
#it is as simple as that !
def jh2bis(Haest, shape, yij, varyj, nj, j1, j2, l=1, wtype=1):
    """
    Haest is the concatenation of H and aest,
    shape is the shape of the original image
    """
    return fbis(Haest, yij, varyj, nj, j1, j2, wtype,
              pfunc=lambda x: l * np.sum(
              np.array(np.gradient(np.reshape(x, shape))) ** 2))


def gradjh2bis(Haest, shape, yij, varyj, nj, j1, j2, l=0, wtype=1):
    return gradfbis(Haest, yij, varyj, nj, j1, j2, wtype,
                  Gpfunc=lambda x: - 2 * l * np.reshape(
                  laplace(np.reshape(x, shape), mode='reflect'), x.shape))


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#this function is the objective function
#J = sum(f(Hi)) + normel2(Gradient(H))^2
#where f(Hi) is the norme L2 of the difference between
#the estimated coefficient H and the data
#usually yij = Elogmuqj for all i but we write yij it is simpler and shorter
#we allow different possible weight but usually we will use the nj as weigth
#I implement my own norm of the gradient and Laplacian
def jh22(H, aest, shape, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return f(H, aest, yij, varyj, nj, j1, j2, wtype) + l * np.sum(
             np.array(o.hgrad(np.reshape(H, shape))) ** 2)


def gradjh22(H, aest, shape, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return gradf(H, aest, yij, varyj, nj, j1, j2, wtype) - 2 * l * np.reshape(
                o.hlaplacian(np.reshape(H, shape)), H.shape)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#this function is the same objective function
#but we use also aest as a variable ! and see if it yields a different result
#Haest is a 2n array, the n first term are H and the n last aest,
#it is as simple as that !
def jh22bis(Haest, shape, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return fbis(Haest, yij, varyj, nj, j1, j2, wtype,
                pfunc=lambda x: l * np.sum(
                np.array(o.hgrad(np.reshape(x, shape))) ** 2))


def gradjh22bis(Haest, shape, yij, varyj, nj, j1, j2, l=0, wtype=1):
    return gradfbis(Haest, yij, varyj, nj, j1, j2, wtype,
                    Gpfunc=lambda x: - 2 * l * np.reshape(
                    o.hlaplacian(np.reshape(x, shape)), x.shape))


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
def jhu(H, aest, shape, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return f(H, aest, yij, varyj, nj, j1, j2, wtype) + l * np.sum(
         np.array(o.hgrad(np.reshape(H, shape))) ** 2)


def gradjhu(H, aest, shape, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return gradf(H, aest, yij, varyj, nj, j1, j2, wtype) + 2 * l * np.reshape(
         o.hflap(np.reshape(H, shape)), H.shape)


def gradjhud(H, aest, shape, yij, varyj, nj, j1, j2, epsilon=1, l=1, wtype=1):
    return gradf(H, aest, yij, varyj, nj, j1, j2, wtype) + l * np.reshape(
           o.hflapd(np.reshape(H, shape), epsilon=epsilon), H.shape)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#this function is the same objective function
#but we use also aest as a variable ! and see if it yields a different result
#Haest is a 2n array, the n first term are H and the n last aest
def jhubis(Haest, shape, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return fbis(Haest, yij, varyj, nj, j1, j2, wtype,
                pfunc=lambda x: l * np.sum(
                np.array(o.hgrad(np.reshape(x, shape))) ** 2))


def gradjhubis(Haest, shape, yij, varyj, nj, j1, j2, l=0, wtype=1):
    return gradfbis(Haest, yij, varyj, nj, j1, j2, wtype,
                    Gpfunc=lambda x: l * np.reshape(
                    o.hflap(np.reshape(x, shape)), x.shape))


def gradjhudbis(Haest, shape, yij, varyj, nj, j1, j2, epsilon=1, l=0, wtype=1):
    return gradfbis(Haest, yij, varyj, nj, j1, j2, wtype,
                    Gpfunc=lambda x: l * np.reshape(
                    o.hflapd(np.reshape(x, shape), epsilon=epsilon), x.shape))


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
def jhumask(H, aest, shape, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    return f(H, aest, yij, varyj, nj, j1, j2, wtype) + l * np.sum(
            (np.array(o.hgradmask(np.reshape(H, shape), mask)) ** 2))


def gradjhudmask(H, aest, shape, yij, varyj, nj, j1, j2, mask, epsilon=1, l=1,
                 wtype=1):
    return gradf(H, aest, yij, varyj, nj, j1, j2, wtype) + l * np.reshape(
           o.hflapdmask(np.reshape(H, shape), mask, epsilon=epsilon), H.shape)


def jhumaskbis(Haest, shape, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    return fbis(Haest, yij, varyj, nj, j1, j2, wtype,
                pfunc=lambda x: l * np.sum(
                (np.array(o.hgradmask(np.reshape(x, shape), mask)) ** 2)))


def gradjhumaskbis(Haest, shape, yij, varyj, nj, j1, j2, mask, l=0, wtype=1):
    return gradfbis(Haest, yij, varyj, nj, j1, j2, wtype,
                    Gpfunc=lambda x: l * np.reshape(
                    o.hflapdmask(np.reshape(x, shape), mask, epsilon=0),
                    x.shape))


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# This are the functions that should be tested on the real datas
def jhm(H, aest, shape, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    return fmask(H, aest, yij, varyj, nj, j1, j2, mask, wtype) + l * np.sum(
           (np.array(o.hgradmask(np.reshape(H, shape), mask)) ** 2))


def gradjhm(H, aest, shape, yij, varyj, nj, j1, j2, mask,
            epsilon=1, l=1, wtype=1):
    return gradfmask(H, aest, yij, varyj, nj, j1, j2, mask,
                     wtype) + l * np.reshape(o.hflapdmask(np.reshape(
                     H, shape), mask, epsilon=epsilon), H.shape)


def jhmbis(Haest, shape, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    return fmaskbis(Haest, yij, varyj, nj, j1, j2, mask, wtype,
        pfunc=lambda x: l * np.sum((np.array(o.hgradmask(
        np.reshape(x, shape), mask)) ** 2)))


def gradjhmbis(Haest, shape, yij, varyj, nj, j1, j2, mask, epsilon=0, l=0,
               wtype=1):
    return gradfmaskbis(Haest, yij, varyj, nj, j1, j2, mask, wtype,
                     Gpfunc=lambda x: l * np.reshape(
                     o.hflapdmask(np.reshape(x, shape),
                     mask, epsilon=epsilon), x.shape))
