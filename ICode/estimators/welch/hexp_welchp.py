"""
These functions compute the Hurst exponent of a signal using the
Welch periodogram
We use here the nitime estimator of welch periodogram
"""
import ICode.optimize as o
from ICode.optimize.objective_functions import _unmask
from ICode.optimize import tv
from ICode.optimize.fista import mfista
from ICode.optimize.proximal_operator import _prox_tvl1_with_intercept, _prox_tvl1
from nitime.analysis.spectral import SpectralAnalyzer
from nitime.timeseries import TimeSeries
import numpy as np

__all__ = ["hurstexp_welchper", "welch_squared_loss"]

def hurstexp_welchper(data, samp=1.05, f_max=0):
    """
    These functions compute the Hurst exponent of a signal using the
    Welch periodogram
    data : your signal
    samp : sampling rate in Hz 1 for an fMRI series
    f_max: the higher frequency you want to take into account
    """
    #data could be two dimensional(but no more...) in that cas time should
    #be on second position
    time_series = TimeSeries(data=data, sampling_rate=samp)
    spectral_analysis = SpectralAnalyzer(time_series)
    frq, pwr = spectral_analysis.psd
    #We need to take only the small frequency, but the exact choice is a
    #bit arbitrary we need to have alpha between 0 and 1
    if f_max==0:
        masker = frq > 0
    else:
        masker = np.all([(frq > 0), (frq < f_max)], axis=0)
    log2frq = np.log2(frq[masker])
    log2pwr = np.log2(pwr.T[masker])
    tmp = np.polyfit(log2frq, log2pwr, deg=1)
    return (1 - tmp[0]) / 2, {'aest': tmp[1], 'log2frq': log2frq, 'log2pwr': log2pwr}


def welch_squared_loss(log2frq, log2pwr, H, aest, compute_energy=True, compute_grad=False):
    """Compute the MSE error, and optionally, its gradient too.
    The cost / energy function is
        MSE =  ||log2(pwr) - [(1 -2H)log2(frq) + aest]||^2
    A (1 / n_samples) factor is applied to the MSE.
    Parameters
    ----------

    compute_energy : bool, optional (default True)
        If set then energy is computed, otherwise only gradient is computed.
    compute_grad : bool, optional (default True)
        If set then gradient is computed, otherwise only energy is computed.
    Returns
    -------
    energy : float
        Energy (returned if `compute_energy` is set).
    gradient : ndarray, shape (n_features,)
        Gradient of energy (returned if `compute_grad` is set).
    """
    if not (compute_energy or compute_grad):
        raise RuntimeError(
            "At least one of compute_energy or compute_grad must be True.")

    residual = log2pwr - np.outer((1 - 2 * H), log2frq)
    residual -= np.outer(aest, np.ones(log2frq.shape))

    # compute energy
    if compute_energy:
        energy = np.dot(residual, residual)
        if not compute_grad:
            return energy

    grad = 4 * np.dot(residual, log2frq)

    if not compute_energy:
        return grad

    return energy, grad


def welch_squared_loss_l2pen(log2frq, log2pwr, H, aest, mask, l=1):
    loss, grad_loss = welch_squared_loss(log2frq, log2pwr, H, aest, compute_energy=True, compute_grad=False)
    grad = o.grad_for_masked_data(H,mask)
    div = o.div(grad)[mask]
    return loss + grad, grad_loss - 2 * l * div


def lipschitz_constant_grad_welch_square_loss(log2frq):
    return 8 * np.abs(np.sum(log2frq))


def tv_welch_solver(Hurst_init, aest, log2frq, log2pwr,
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

    if lipschitz_constant==0:
        lipschitz_constant = lipschitz_constant_grad_welch_square_loss(log2frq)

    def total_energy(x):
        return welch_squared_loss(log2frq, log2pwr, Hurst_init, aest, compute_energy=True, compute_grad=False)

    def unmaskvec(w):
        return _unmask(w, mask)

    def maskvec(w):
        return w[flat_mask]

    def f1_grad(x):
        return welch_squared_loss(log2frq, log2pwr, Hurst_init, aest, compute_energy=False, compute_grad=True)

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