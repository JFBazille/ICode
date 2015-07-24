#! /usr/bin/python3.4
#fast has been implemented to test Hurstexp_Welchper
#(Hurst exponent estmator using Wlech periodogramme)
#on real datas

from ICode.estimators.wavelet import *
from ICode.progressbar import ProgressBar
from ICode.extractsignals import signal_extractor
from ICode.estimators.penalized import l2regu, wetvp
from ICode.loader import load_dynacomp

import numpy as np
import os
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import ttest_1samp as ttest
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiMasker


import itertools


def test_real_data_l2min(nbsubjects=1,lbda=1, title_prefix='test_real_data_l2min', nbvanishmoment=2, j1=2, j2=6):
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

        title = title_prefix + str(idx_subject) + '.pdf'

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


def test_real_data_tvmin(nbsubjects=1,lbda=1, title_prefix='test_real_data_tvmin', nbvanishmoment=2, j1=2, j2=6):
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

        l1_ratio = 0
        j22 = np.min((j2, len(nj)))
        j1j2 = np.arange(j1 - 1, j22)
        njj = nj[j1j2]
        N = sum(njj)
        wvarjj = njj / N
        lipschitz_constant =  np.sum(8 * ((j1j2 + 1) ** 2) * wvarjj)
        fmin = lambda lbda: mtvsolver(H, aest,
                                Elog, Varlog,
                                nj, j1, j2,mask,
                                lipschitz_constant=lipschitz_constant,
                                l1_ratio = l1_ratio, l=lbda)

        title = title_prefix + 'loss_tv_penalisation_on_grad'+ str(idx_subject) + '.pdf'

        monmin = fmin(lbda)
        img = masker.inverse_transform(monmin[0])
        output_file = os.path.join(OUTPUT_PATH, title)

        p = plot_stat_map(img, output_file=output_file)
        mlist.append(monmin[0])
        idx_subject += 1

    plt.show()


def test_real_data_multivariate_analysis_ttest(nbsubjects=1,lbda=1, title_prefix='test_real_data_tvmin', nbvanishmoment=2, j1=2, j2=6, OUTPUT_PATH=None):
    if OUTPUT_PATH is None:
        OUTPUT_PATH = os.path.join('/volatile', 'hubert', 'beamer', 'graphics', 'multivariate_analysis_ttest')
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
    tv_dict = dict()
    l2_dict = dict()
    univariate_list = list()

    for x,masker in extractor:
        x = x.T
        N = x.shape[0]
        l = x.shape[1]
        nbvoies = min(int(np.log2(l / (2 * nbvanishmoment + 1))), int(np.log2(l)))

        dico = hdw_p(x, nbvanishmoment, norm=1, q=np.array(2), nbvoies=nbvoies, distn=0, wtype=wtype, j1=j1, j2=j2, printout=0)
        Elog = dico['Elogmuqj'][:, 0]
        Varlog = dico['Varlogmuqj'][:, 0]
        nj = dico['nj']
        estimate = dico['Zeta'] / 2.
        aest = dico['aest']
        min_value = np.min(estimate)
        max_value = np.max(estimate)
        univariate_list.append(estimate)

        #we create the image with the appropriate function
        imgHurst = masker.inverse_transform(estimate)
        output_file = os.path.join(OUTPUT_PATH,'brain_univariate_subject' + str(idx_subject) + '.pdf')
        p = plot_stat_map(imgHurst, output_file=output_file)
        #all is plugged in the appropriate plotting function
        #VERY IMPORTANT DO NOT TRY DO DO SO WITH NON BOOLEAN VALUE !
        mask = masker.mask_img_.get_data() > 0
        shape = mask.shape
        H = imgHurst.get_data()[mask]
        imgaest = masker.inverse_transform(aest)
        aest = imgaest.get_data()[mask]

        #######################################################################

        f = lambda x, lbda: l2regu.loss_l2_penalization_on_grad(x, aest,
                            Elog, Varlog, nj, j1, j2, mask, l=lbda)
        #We set epsilon to 0
        g = lambda x, lbda: l2regu.grad_loss_l2_penalization_on_grad(x, aest,
                            Elog, Varlog, nj, j1, j2, mask, l=lbda)

        l2_title = title_prefix + 'loss_l2_penalisation_on_grad_subject' + str(idx_subject) 

        fg = lambda x, lbda, **kwargs: (f(x, lbda), g(x, lbda))
        #For each lambda we use blgs algorithm to find the minimum
        # We start from the
        l2_algo = lambda lbda: fmin_l_bfgs_b(lambda x: fg(x, lbda), estimate)

        #######################################################################

        lipschitz_constant = wetvp.lipschitz_constant_gradf(j1,j2,Varlog, nj, wtype)
        l1_ratio = 0
        tv_algo = lambda lbda: wetvp.mtvsolver(estimate, aest,
                                            Elog, Varlog,
                                            nj, j1, j2,mask,
                                            lipschitz_constant=lipschitz_constant,
                                            l1_ratio = l1_ratio, l=lbda)
        tv_title = title_prefix + 'wetvp_subject' + str(idx_subject)

        ######################################################################

        if not isinstance(lbda, np.ndarray):
            lbda = lbda
            l2_dict[lbda] = list()
            monmin = l2_algo(lbda)
            img = masker.inverse_transform(monmin[0])

            output_file = os.path.join(OUTPUT_PATH, 'multivariate', l2_title + 'lambda' + str(lbda) + '.pdf')
            p = plot_stat_map(img, output_file=output_file)
            l2_dict[lbda] = list()
            l2_dict[lbda].append(monmin[0])

            output_file = os.path.join(OUTPUT_PATH, 'multivariate_normalized_color', l2_title + 'lambda' + str(lbda) + '.pdf')
            p = plot_stat_map(img, vmax=max_value, output_file=output_file)
            l2_dict[lbda] = list()
            l2_dict[lbda].append(monmin[0])

            monmin = tv_algo(lbda)
            img = masker.inverse_transform(monmin[0])
            output_file = os.path.join(OUTPUT_PATH, 'multivariate', tv_title + 'lambda' + str(lbda) + '.pdf')

            p = plot_stat_map(img, output_file=output_file)
            tv_dict[lbda] = list()
            tv_dict[lbda].append(monmin[0])

            output_file = os.path.join(OUTPUT_PATH, 'multivariate_normalized_color', tv_title + 'lambda' + str(lbda) + '.pdf')
            p = plot_stat_map(img, vmax=max_value, output_file=output_file)
            l2_dict[lbda] = list()
            l2_dict[lbda].append(monmin[0])

        else:
            for lbda_loop in lbda:
                l2_dict[lbda] = list()
                monmin = l2_algo(lbda_loop[0])
                img = masker.inverse_transform(monmin[0])

                output_file = os.path.join(OUTPUT_PATH, 'multivariate', l2_title + 'lambda' + str(lbda_loop) + '.pdf')
                p = plot_stat_map(img, output_file=output_file)
                l2_dict[lbda_loop] = list()
                l2_dict[lbda_loop].append(monmin[0])

                output_file = os.path.join(OUTPUT_PATH, 'multivariate_normalized_color', l2_title + 'lambda' + str(lbda_loop) + '.pdf')
                p = plot_stat_map(img, vmax=max_value, output_file=output_file)
                l2_dict[lbda_loop] = list()
                l2_dict[lbda_loop].append(monmin[0])

                monmin = tv_algo(lbda[0])
                img = masker.inverse_transform(monmin[0])
                output_file = os.path.join(OUTPUT_PATH, 'multivariate', tv_title + 'lambda' + str(lbda_loop) + '.pdf')

                p = plot_stat_map(img, output_file=output_file)
                tv_dict[lbda_loop] = list()
                tv_dict[lbda_loop].append(monmin[0])

                output_file = os.path.join(OUTPUT_PATH, 'multivariate_normalized_color', tv_title + 'lambda' + str(lbda_loop) + '.pdf')
                p = plot_stat_map(img, vmax=max_value, output_file=output_file)
                l2_dict[lbda_loop] = list()
                l2_dict[lbda_loop].append(monmin[0])
        idx_subject += 1

    ## ## ## ## ## ttest part  ## ## ## ## ##
    t, proba = ttest(univariate_list)
    logproba = - np.log10(proba)
    img = masker.inverse_transform(logproba)
    output_file = os.path.join(OUTPUT_PATH, 'ttest', 'univariate_test' + '.pdf')
    p = plot_stat_map(img, output_file=output_file)

    img = masker.inverse_transform(logproba * np.array(logproba > 5, dtype=int))
    output_file = os.path.join(OUTPUT_PATH, 'ttest', 'univariate_test_threshold' + '.pdf')
    p = plot_stat_map(img, output_file=output_file)

    if not isinstance(lbda, np.ndarray):
        t, proba = ttest(l2_dict[lbda])
        logproba = - np.log10(proba)
        img = masker.inverse_transform(logproba)
        output_file = os.path.join(OUTPUT_PATH, 'ttest', 'multivariate_lambda' + str(lbda) + '.pdf')
        p = plot_stat_map(img, output_file=output_file)

        img = masker.inverse_transform(logproba * np.array(logproba > 5, dtype=int))
        output_file = os.path.join(OUTPUT_PATH, 'ttest', 'multivariate_threshold_lambda' + str(lbda) + '.pdf')
        p = plot_stat_map(img, output_file=output_file)

    else:
        for lbda_loop in lbda:
            t, proba = ttest(l2_dict[lbda_loop])
            logproba = - np.log10(proba)
            img = masker.inverse_transform(logproba)
            output_file = os.path.join(OUTPUT_PATH, 'ttest', 'multivariate_lambda' + str(lbda_loop) + '.pdf')
            p = plot_stat_map(img, output_file=output_file)

            img = masker.inverse_transform(logproba * np.array(logproba > 5, dtype=int))
            output_file = os.path.join(OUTPUT_PATH, 'ttest', 'multivariate_threshold_lambda' + str(lbda_loop) + '.pdf')
            p = plot_stat_map(img, output_file=output_file)
    plt.show()


