#Wavelet estimator penalisation L2 first and simpler penalization
import numpy as np
from scipy.ndimage.filters import laplace
import ICode.optimize as o
from .utils import *
from ICode.optimize.objective_functions import _unmask
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


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#This are the same function implemented with elvis gradient and laplacian
def jhe(H, aest, shape, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return f(H, aest, yij, varyj, nj, j1, j2, wtype) + l * np.sum(
           o.gradient(np.reshape(H,shape)) ** 2)


def gradjhe(H, aest, shape, yij, varyj, nj, j1, j2, l=1, wtype=1):
    return gradf(H, aest, yij, varyj, nj, j1, j2, wtype) - 2 * l * np.reshape(
           o.div(o.gradient(np.reshape(H, shape))), H.shape)


def jhebis(Haest, shape, yij, varyj, nj, j1, j2, l=1, wtype=1):
    """
    Haest is the concatenation of H and aest,
    shape is the shape of the original image
    """
    return fbis(Haest, yij, varyj, nj, j1, j2, wtype,
              pfunc=lambda x: l * np.sum(
              o.gradient(np.reshape(x, shape)) ** 2))


def gradjhebis(Haest, shape, yij, varyj, nj, j1, j2, l=0, wtype=1):
    return gradfbis(Haest, yij, varyj, nj, j1, j2, wtype,
                  Gpfunc=lambda x: - 2 * l * np.reshape(
                  o.div(o.gradient(np.reshape(x, shape))), x.shape))

##Same but these function are able to deal with a mask
##Usefull in my case with real datas !
def jhem(H, aest, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    return f(H, aest, yij, varyj, nj, j1, j2, wtype) + l * np.sum(
           o.gradient(_unmask(H,mask)) ** 2)


def gradjhem(H, aest, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    return gradf(H, aest, yij, varyj, nj, j1, j2,
            wtype) - 2 * l * o.div(o.gradient(_unmask(H,mask)))[mask]


def jhembis(Haest, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    """
    Haest is the concatenation of H and aest,
    shape is the shape of the original image
    """
    return fbis(Haest, yij, varyj, nj, j1, j2, wtype,
              pfunc=lambda x: l * np.sum(
              o.gradient(_unmask(x,mask)) ** 2))


def gradjhembis(Haest, yij, varyj, nj, j1, j2, mask, l=0, wtype=1):
    return gradfbis(Haest, yij, varyj, nj, j1, j2, wtype,
           Gpfunc=lambda x: - 2 * l * o.div(o.gradient(_unmask(x,mask)))[mask])
 