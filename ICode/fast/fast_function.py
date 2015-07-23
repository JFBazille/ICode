#! /usr/bin/python3.4
#fast has been implemented to test Hurstexp_Welchper
#(Hurst exponent estmator using Wlech periodogramme)
#on real datas
import os
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
import numpy as np
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiMasker
from ICode.estimators.wavelet import *
from ICode.progressbar import ProgressBar
from ICode.extractsignals import signal_extractor
from ICode.estimators.penalized import l2regu
from scipy.optimize import fmin_l_bfgs_b
from ICode.loader import load_dynacomp
import itertools


def test_real_data(nbsubjects=1,lbda=1, title_prefix='', nbvanishmoment=2, j1=2, j2=6):
    OUTPUT_PATH = os.path.join('/volatile', 'hubert', 'beamer', 'multivariate_analysis6')
    idx_subject = 0
    ##loading data
    dataset = load_dynacomp(preprocessing_folder='pipeline_1', prefix='wr')
    extractor = signal_extractor(dataset)
    extractor = extractor.extract()
    extractor = itertools.islice(extractor,idx_subject, nbsubjects)

    nbvanishmoment = 2
    j1 = 2
    j2 = 6
    wtype = 1
    mlist = list()

    for x,masker in extractor:
        x = x.T
        N = x.shape[0]
        l = x.shape[1]
        Bar = ProgressBar(N, 60, 'Work in progress')
        estimate = np.zeros(N)
        aest = np.zeros(N)
        nbvoies = min(int(np.log2(l / (2 * nbvanishmoment + 1))), int(np.log2(l)))

        dico = wtspecq_statlog32(x, nbvanishmoment, 1, np.array(2), int(np.log2(x.shape[1])), 0, 0)
        Elog = dico['Elogmuqj'][:, 0]
        Varlog = dico['Varlogmuqj'][:, 0]
        nj = dico['nj']
        del dico

        for j in np.arange(0, N):
            sortie = regrespond_det2(Elog[j], Varlog[j], 2, nj, j1, j2, wtype)
            estimate[j] = sortie['Zeta'] / 2.
            aest[j] = sortie['aest']
            Bar.update(j)
        del sortie

        #we create the image with the appropriate function
        imgHurst = masker.inverse_transform(estimate)
        output_file = os.path.join(OUTPUT_PATH, 'brain_univariate_subject' + str(idx_subject) + '.pdf')
        p = plot_stat_map(imgHurst, output_file=output_file)
        #all is plugged in the appropriate plotting function
        #VERY IMPORTANT DO NOT TRY DO DO SO WITH NON BOOLEAN VALUE !
        mask = masker.mask_img_.get_data() > 0
        shape = mask.shape
        H = imgHurst.get_data()[mask]
        imgaest = masker.inverse_transform(aest)
        aest = imgaest.get_data()[mask]

        f = lambda x, lbda: l2regu.loss_l2_penalization_on_grad(x, aest,
                          Elog, Varlog, nj, j1, j2, mask, l=lbda)
        #We set epsilon to 0
        g = lambda x, lbda: l2regu.grad_loss_l2_penalization_on_grad(x, aest, Elog, Varlog, nj, j1, j2, mask,
                                l=lbda)

        title = title_prefix + 'loss_l2_penalisation_on_grad'+ str(idx_subject) + '.pdf'

        fg = lambda x, lbda, **kwargs: (f(x, lbda), g(x, lbda))
        #For each lambda we use blgs algorithm to find the minimum
        # We start from the
        fmin = lambda lbda: fmin_l_bfgs_b(lambda x: fg(x, lbda), H)

        monmin = fmin(lbda)
        img = masker.inverse_transform(monmin[0])
        output_file = os.path.join(OUTPUT_PATH, title)

        p = plot_stat_map(img, output_file=output_file)
        mlist.append(monmin[0])
        idx_subject += 1

    plt.show()
