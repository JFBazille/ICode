import numpy as np
import ICode.optimize as o
from .utils import f,gradf
from ICode.optimize.objective_functions import _unmask
"""
this functions are objective functions of type
J = sum(f(Hi)) + lambda x penalyzation(H)

f(Hi) is the norme L2 of the difference
between the estimated coefficient H

The penalization function is l2 norm of the gradient

the data :
    usually yij = Elogmuqj for all i but we write yij it is simpler and shorter
    we allow different possible weight but usually we will use the nj as weigth
"""
##Same but these function are able to deal with a mask
##Usefull in my case with real datas !
__all__ = ['loss_l2_penalization_on_grad', 'grad_loss_l2_penalization_on_grad']
def loss_l2_penalization_on_grad(H, aest, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    grad = o.grad_for_masked_data(H,mask)
    return f(H, aest, yij, varyj, nj, j1, j2, wtype) + l * np.sum(grad**2)


def grad_loss_l2_penalization_on_grad(H, aest, yij, varyj, nj, j1, j2, mask, l=1, wtype=1):
    grad = o.grad_for_masked_data(H,mask)
    div = o.div(grad)[mask]
    return gradf(H, aest, yij, varyj, nj, j1, j2,
            wtype) - 2 * l * div