#! /usr/bin/python3.4
#fast has been implemented to test Hurstexp_Welchper
#(Hurst exponent estmator using Wlech periodogramme)
#on real datas
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
import numpy as np
from ICode.extractsignals import extract_one_vpv_signal
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
max_idx_subject = 10
##loading data
dataset = load_dynacomp(preprocessing_folder='pipeline_1', prefix='wr')
[x, masker] = extract_one_vpv_signal(dataset)


j1 = 2
wtype = 1

x = x.T
N = x.shape[0]
l = x.shape[1]
Bar = ProgressBar(N, 60, 'Work in progress')
estimate = np.zeros(N)
aest = np.zeros(N)

for nbvanishmoment  in np.arange(2,5):
    nbvoies = min(int(np.log2(l / (2 * nbvanishmoment + 1))), int(np.log2(l)))
    Bar = ProgressBar(N, 60, 'Work in progress')
    #Elog = np.zeros((N, nbvoies))
    #Varlog = np.zeros((N, nbvoies))
    #for j2 in np.arange(4,7):
        #for i in np.arange(0, N):
            #Bar.update(i)
            #dico = wtspecq_statlog3(x[i], nbvanishmoment, 1, np.array(2),
                                #nbvoies, 0, 0)
            #sortie = regrespond_det2(dico['Elogmuqj'][0], dico['Varlogmuqj'][0], 2,
                                #dico['nj'], j1, j2, wtype)
            #estimate[i] = sortie['Zeta'] / 2.
            #aest[i] = sortie['aest']
            #Elog[i] = dico['Elogmuqj'][0]
            #Varlog[i] = dico['Varlogmuqj'][0]
            #if i==0:
                #nj = dico['nj']
            #Bar.update(i + 1)
        #img = masker.inverse_transform(estimate)
        #output_file = ('/volatile/hubert/beamer/comparison/brain_nbvanishmoment%d_j2%d_statlog3_nonsmoothed.pdf' %(nbvanishmoment,j2))
        #p = plot_stat_map(img, output_file=output_file)

    dico = wtspecq_statlog32(x, nbvanishmoment, 1, np.array(2), int(np.log2(x.shape[1])), 0, 0)
    Elog = dico['Elogmuqj'][:, 0]
    Varlog = dico['Varlogmuqj'][:, 0]
    nj = dico['nj']
    del dico
    for j2 in np.arange(4,7):
        for i in np.arange(0, N):
            sortie = regrespond_det2(Elog[i], Varlog[i], 2, nj, j1, j2, wtype)
            estimate[i] = sortie['Zeta'] / 2.
            aest[i] = sortie['aest']
            Bar.update(i)
        img = masker.inverse_transform(estimate)
        output_file = ('/volatile/hubert/beamer/comparison/brain_nbvanishmoment%d_j2%d_statlog32_nonsmoothed.pdf' %(nbvanishmoment,j2))
        p = plot_stat_map(img, output_file=output_file)


plt.show()
