# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from nilearn.input_data import NiftiMasker
from joblib import Parallel, delayed
import pickle
from ICode.estimators import wavelet_worker, dfa_worker, welch_worker


__all__ = ["Hurst_Estimator"]

class Hurst_Estimator(BaseEstimator, TransformerMixin):
    """This class makes Hurst coefficient estimation for Niftii file
    easier.
    First one should initialise the Niftii Masker with the corresponding
    elements:
        detrend
        low_pass
        high_pass
        t_r
        smoothing_fwhm        
        memory
        memory_level
        See nilearn.input_data.NiftiMasker for more details
    One must choose the metric and the regulation:
    metric:  'wavelet', 'dfa' or 'welch'
    regu:    'tv', 'l2', 'off'
    lambda:  the ponderation for the regulation cost function
    
    Than one can use the fit function and compute the Hurst Exponent load_map
    for each signal contained in imgs niftii file.
    """
    
    def __init__(self, mask=None, metric='wavelet', regu='tv', lbda=1, detrend=True,
                 low_pass=.1, high_pass=.01, t_r=1.05, smoothing_fwhm=6.,
                 memory='', memory_level=0, n_jobs=1, nb_vanishmoment=2,
                 norm=1, q=np.array(2), nbvoies=None,
                 distn=1, wtype=1, j1=2, j2=8):
        self.metric = metric
        self.mask = mask
        self.n_jobs = n_jobs
        self.nb_vanishmoment = nb_vanishmoment
        self.norm = norm
        self.q = q
        self.nbvoies = nbvoies
        self.distn = distn
        self.wtype = wtype
        self.j1 = j1
        self.j2 = j2
        self.regu = regu
        self.lbda = lbda
        
        if self.mask is None:
                self.masker = NiftiMasker(detrend=detrend,
                                    low_pass=low_pass,
                                    high_pass=high_pass,
                                    t_r=t_r,
                                    smoothing_fwhm=smoothing_fwhm,
                                    standardize=False,
                                    memory_level=memory_level,
                                    verbose=0)
        else:
            self.masker = NiftiMasker(mask_img=self.mask,
                                    detrend=detrend,
                                    low_pass=low_pass,
                                    high_pass=high_pass,
                                    t_r=t_r,
                                    smoothing_fwhm=smoothing_fwhm,
                                    standardize=False,
                                    memory_level=memory_level,
                                    verbose=0)
            self.masker.fit(self.mask)


    def fit(self, imgs):
        """ compute connectivities
        """

        if self.metric == 'wavelet':
            jobs = (delayed(wavelet_worker)(img, self.masker, self.regu, self.lbda,
                                                    self.nb_vanishmoment, self.norm,
                                                    self.q, self.nbvoies,
                                                    self.distn, self.wtype,
                                                    self.j1, self.j2) for img in imgs)

        elif self.metric == 'dfa':
            jobs = (delayed(dfa_worker)(img, self.masker, self.regu, self.lbda,
                                                    self.wtype,
                                                    self.j1, self.j2) for img in imgs)
        elif self.metric=='welch':
            jobs = (delayed(welch_worker)(img, self.masker, self.regu, self.lbda,
                                                    ) for img in imgs)
        else:
            raise ValueError("the metric dico = %s is not yet implemented"
                % (self.metric,))

        ts = Parallel(n_jobs=5, verbose=5)(jobs)

        self.hurst = ts
        return self.hurst
    
    def save(self, save_path='',
             save_file=None):

        if not 'hurst' in dir(self):
            os.write(1, 'Nothing to save !!')
            return

        if save_file is None:
            save_file = 'hurstmap_metric_' + self.metric +'_regu_'+ self.regu
        save_file = os.path.join(save_path, save_file)
        with open(save_file,'wb') as myfile:
            monpickler = pickle.Pickler(myfile)
            monpickler.dump(self.hurst)

    def load_map(self, INPUT_PATH='', save_file=None):
        if save_file is None:
            save_file = 'hurstmap_metric_' + self.metric +'_regu_'+ self.regu
        save_file = os.path.join(INPUT_PATH, save_file)
        with open(save_file,'rb') as myfile:
            monunpickler = pickle.Unpickler(myfile)
            self.hurst = monunpickler.load()