#! /usr/bin/python3.4
#fast has been implemented to test Hurstexp_Welchper
#(Hurst exponent estmator using Wlech periodogramme)
#on real datas
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
import numpy as np
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiMasker

from ICode.extractsignals.extract import extract_one_vpv_signal
from ICode.estimators.wavelet import *
from ICode.progressbar import ProgressBar
from ICode.loader import load_dynacomp
from ICode.estimators.penalyzed import jhm, gradjhm, jhmbis, gradjhmbis

from scipy.optimize import fmin_l_bfgs_b  #, check_grad
##loading data
dataset2 = load_dynacomp(preprocessing_folder='pipeline_1', prefix='swr')

print 'chargement des donnees'

#we take out the signal
[x, masker] = extract_one_vpv_signal(dataset2)
x = x.T
N = x.shape[0]
Bar = ProgressBar(N, 60, 'Work in progress')
j1 = 2
j2 = 7
wtype = 1
##un Autre un peu moins precise mais tres rapide
dico = wtspecq_statlog32(x, 2, 1, np.array(2), int(np.log2(x.shape[1])), 0, 0)
Elog = dico['Elogmuqj'][:, 0]
Varlog = dico['Varlogmuqj'][:, 0]
nj = dico['nj']
del dico
estimate = np.zeros(N)
aest = np.zeros(N)
for j in np.arange(0, N):
  sortie = regrespond_det2(Elog[j], Varlog[j], 2, nj, j1, j2, wtype)
  estimate[j]=sortie['Zeta'] / 2. #normalement Zeta
  aest[j]  = sortie['aest']
  Bar.update(j)

del sortie
## ## ## ## ##


#we create the image with the appropriate function
img8 = masker.inverse_transform(estimate)
#all is plugged in the appropriate plotting function
plt.figure(1)
p = plot_stat_map(img8)
#VERY IMPORTANT DO NOT TRY DO DO SO WITH NON BOOLEAN VALUE !
mask = masker.mask_img_.get_data() > 0
shape = mask.shape
H = img8.get_data()
del img8
img9 = masker.inverse_transform(aest)
aest = img9.get_data()
del img9

choice = 20

if choice == 20:
    f = lambda x, lbda: jhmbis(x, shape,
    Elog, Varlog, nj, j1, j2, mask, l=lbda)
    g = lambda x, lbda: gradjhmbis(x, shape, Elog, Varlog,
                                             nj, j1, j2, mask, epsilon=0,
                                             l=lbda)
    H = np.concatenate((H, aest))
    title = 'JHumaskbis GradJhudmaskbis'

if choice == 21:
    f = lambda x, lbda: jhm(x, aest, shape,
                          Elog, Varlog, nj, j1, j2, mask, l=lbda)
    #We set epsilon to 0
    g = lambda x, lbda: gradjhm(x, aest, shape, Elog, Varlog, nj, j1, j2, mask,
                              epsilon=0, l=lbda)
    title = 'JHm GradJHm'

fg = lambda x, lbda, **kwargs: (f(x, lbda), g(x, lbda))
#For each lambda we use blgs algorithm to find the minimum
# We start from the
fmin = lambda lbda: fmin_l_bfgs_b(lambda x: fg(x, lbda), H)
ckgrad = lambda lbda: check_grad(lambda x: f(x, lbda), lambda x: g(x, lbda), H)
                                                               #, epsilon = )

lmax = 15
#minimiseurs = list()
#cg = np.zeros(lmax)
r = np.arange(lmax)
lbda = np.array((0,) + tuple(1.5 ** r[:-1]))


if choice % 2 == 1:
    for idx in r:
        monmin = fmin(lbda[idx])
        #cg[idx] = ckgrad(lbda[idx])
        img = masker.inverse_transform(np.reshape(monmin[0], mask.shape)[mask])
        p = plot_stat_map(img)
        p.title('Lambda = %.3f' % lbda[idx])
        #minimiseurs.append(img)
else:
    for idx in r:
        monmin = fmin(lbda[idx])
        #cg[idx] = ckgrad(lbda[idx])
        img = masker.inverse_transform(np.reshape(
                monmin[0][:monmin[0].shape[0] / 2], mask.shape)[mask])
        p = plot_stat_map(img)
        p.title('Lambda = %.3f' % lbda[idx])
        #minimiseurs.append(img)

plt.show()
