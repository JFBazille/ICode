#! /usr/bin/python3.4
#fast has been implemented to test Hurstexp_Welchper
#(Hurst exponent estmator using Wlech periodogramme)
#on real datas
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
import numpy as np
from ICode.extractsignals import signal_extractor
from ICode.estimators.wavelet import *
from ICode.progressbar import ProgressBar
from ICode.loader import load_dynacomp
from ICode.estimators.penalyzed import mtvsolver, mtvsolverbis
import itertools
from scipy.optimize import fmin_l_bfgs_b
import os

statlog = 0
OUTPUT_PATH = os.path.join('/volatile', 'hubert', 'beamer', 'multivariate_analysis5')
idx_subject = 0
max_idx_subject = 15
##loading data
dataset = load_dynacomp(preprocessing_folder='pipeline_1', prefix='wr')
extractor = signal_extractor(dataset)
extractor = extractor.extract()
extractor = itertools.islice(extractor,idx_subject, max_idx_subject)

nbvanishmoment = 4
j1 = 2
j2 = 6
wtype = 1
#lbda = 1
mlist = list()

for x,masker in extractor:
    x = x.T
    N = x.shape[0]
    l = x.shape[1]
    Bar = ProgressBar(N, 60, 'Work in progress')
    estimate = np.zeros(N)
    aest = np.zeros(N)
    nbvoies = min(int(np.log2(l / (2 * nbvanishmoment + 1))), int(np.log2(l)))
    if statlog == 1:
        Bar = ProgressBar(N, 60, 'Work in progress')
        Elog = np.zeros((N, nbvoies))
        Varlog = np.zeros((N, nbvoies))

        for j in np.arange(0, N):
            Bar.update(j)
            dico = wtspecq_statlog3(x[j], nbvanishmoment, 1, np.array(2),
                                nbvoies, 0, 0)
            sortie = regrespond_det2(dico['Elogmuqj'][0], dico['Varlogmuqj'][0], 2,
                                dico['nj'], j1, j2, wtype)
            estimate[j] = sortie['Zeta'] / 2.
            aest[j] = sortie['aest']
            Elog[j] = dico['Elogmuqj'][0]
            Varlog[j] = dico['Varlogmuqj'][0]
            if j==0:
                nj = dico['nj']
            Bar.update(j + 1)
    else:
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
    f = lambda lbda: mtvsolver(H, aest,
                               Elog, Varlog,
                               nj, j1, j2,mask,
                               lipschitz_constant=0,
                               l1_ratio = l1_ratio, l=lbda)
    title = 'brain_wetvp_lbda1_subject' + str(idx_subject) + '.pdf'

    monmin = f(lbda)
    img = masker.inverse_transform(monmin[0])
    output_file = os.path.join(OUTPUT_PATH, title)

    p = plot_stat_map(img, output_file=output_file)
    mlist.append(monmin[0])
    idx_subject += 1

plt.show()
